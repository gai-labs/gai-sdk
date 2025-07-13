import os
import re
import json
import time
import tempfile
import shutil
from typing import Any, Optional, Union
from gai.lib.constants import DEFAULT_GUID
from gai.lib.logging import getLogger
from gai.messages.typing import (
    MessagePydantic,
    MonologueBodyPydantic,
    MessageHeaderPydantic,
)
from gai.messages import message_helper
from gai.messages import MessageStore

logger = getLogger(__file__)

class Monologue:
    def __init__(
        self,
        agent_name: str = "Assistant",
        messages: Optional[Union["Monologue", list[MessagePydantic]]] = None,
        limit: int = 600000,        
    ):
        self.agent_name = agent_name
        self.limit = limit  # Character limit for messages
        self._messages = messages
        if self._messages:
            if isinstance(self._messages, Monologue):
                self._messages = self._messages.list_messages().copy()
        else:
            self._messages = []
    
    def get_total_size(self, new_message: Optional[dict] = None):
        chat_messages = self.list_chat_messages()
        total_size = len(json.dumps(chat_messages))
        if new_message:
            total_size += len(json.dumps(new_message))
        return total_size
    
    def list_chat_messages(self) -> list[dict[str, Any]]:
        """
        Returns the list of chat messages in the monologue.
        """
        return message_helper.convert_to_chat_messages(self._messages)

    def is_new(self):
        """
        Check if the monologue is new.
        A monologue is considered new if it has no messages or only contains a system message.
        """
        if not self._messages:
            return True
        return False

    def add_user_message(self, content: Any, state=None):
        state_name = ""
        step_no = -1
        if state:
            state_name = state.title
            step_no = state.input["step"]

        message = MessagePydantic(
            header=MessageHeaderPydantic(sender="User", recipient=self.agent_name),
            body=MonologueBodyPydantic(
                state_name=state_name,
                step_no=step_no,
                role="user",
                content=content,
            ),
        )

        try:
            while (
                self.get_total_size({"role": "user", "content": content}) > self.limit
            ):
                # Remove second and third oldest message (last being the original user message)
                if len(self._messages) > 2:
                    self._messages.pop(1)
                    self._messages.pop(1)
                else:
                    raise Exception(
                        "add_user_message: content size is bigger than 600,000 char. Are you sending an image?"
                    )
        except Exception as e:
            logger.error(f"add_user_message: error={str(e)}")
            raise e

        self._messages.append(message)
        return self

    def add_assistant_message(self, content: Any, state=None):
        state_name = ""
        step_no = -1
        if state:
            state_name = state.title
            step_no = state.input["step"]

        message = MessagePydantic(
            header=MessageHeaderPydantic(sender=self.agent_name, recipient="User"),
            body=MonologueBodyPydantic(
                state_name=state_name,
                step_no=step_no,
                role="assistant",
                content=content,
            ),
        )

        try:
            while (
                self.get_total_size({"role": "assistant", "content": content})
                > self.limit
            ):
                # Remove second and third oldest message (last being the original user message)
                if len(self._messages) > 2:
                    self._messages.pop(1)
                    self._messages.pop(1)
                else:
                    raise Exception(
                        "add_user_message: content size is bigger than 600,000 char. Are you sending an image?"
                    )

        except Exception as e:
            logger.error(f"add_assistant_message: error={str(e)}")
            raise e

        self._messages.append(message)
        return self

    def copy(self):
        """Returns a copy of the monologue."""
        return Monologue(
            agent_name=self.agent_name,
            messages=self._messages.copy()
        )

    def list_messages(self) -> list[MessagePydantic]:
        return self._messages.copy()

    def pop(self):
        """
        pop the last message
        """
        return self._messages.pop()

    def update(self, messages: list[MessagePydantic]):
        """
        Replace the internal list
        """
        self._messages = messages.copy()
        return self._messages

    def reset(self, path: Optional[str] = None):
        self._messages.clear()

    def get_last_toolcalls(self):
        """
        From the last message in the monologue, extract all tool_call content.
        """

        # messages = self.machine.monologue.list_messages()
        last_message = self._messages[-1] if self._messages else None
        if not last_message:
            raise ValueError(
                "AthropicToolUseState: No messages found. Check if 'start' was called."
            )
        if last_message.body.role != "assistant":
            # If last message is not an assistant's message, then something might have gone wrong previously preventing the successful completion of the state.
            logger.warning(
                "AthropicToolUseState: Last message is not from assistant. Removing last message to try again."
            )
            # Remove the last message try again.
            self._messages.pop()
            last_message = self._messages[-1] if self._messages else None
            if not last_message:
                raise ValueError(
                    "AthropicToolUseState: No messages found in ToolUseState. Check the output from ChatState."
                )

        tool_calls = []
        if isinstance(last_message.body.content, list):
            for item in last_message.body.content:
                if isinstance(item, str):
                    continue
                if not isinstance(item, dict):
                    item = item.model_dump()
                if item["type"] == "tool_use":
                    tool_calls.append(
                        {
                            "tool_use_id": item["id"],
                            "tool_name": item["name"],
                            "arguments": item["input"],
                        }
                    )
        return tool_calls

    def is_terminated(self):
        """
        LLM Terminated: If the last message is an assistant message,
        check if it doesn't contain any tool calls (or if it contains "task_completed" tool - this is obsoleted by remain for compatibility).

        User Terminated: If the last message is a user message,
        check if it contains "TERMINATE".
        """
        last_tool_calls = self.get_last_toolcalls()
        if not last_tool_calls:
            return True

        # Note: Use of "task_completed" is obsoleted.
        if last_tool_calls and any(
            result["tool_name"] == "task_completed" for result in last_tool_calls
        ):
            return True

        last_message = self._messages[-1] if self._messages else None
        last_content = last_message.body.content
        if (
            last_message.body.role == "user"
            and isinstance(last_content, str)
            and last_content.upper() == "TERMINATE"
        ):
            return True

        return False

    def is_interrupted(self, user_message: str):
        """
        LLM Interrupted:
        If the last message is an assistant message containing "user_input"
        and no user_message, that means LLM is still pending for user_message
        and should terminate the flow.
        If user_message is present, that means it is no longer pending for user_message
        and so the flow will continue.
        """
        last_tool_calls = self.get_last_toolcalls()
        if not last_tool_calls:
            return True

        if (
            last_tool_calls
            and any(result["tool_name"] == "user_input" for result in last_tool_calls)
            and not user_message
        ):
            return True

        return False

# -----
    
class FileMonologue(Monologue):
    def __init__(
        self,
        agent_name: str = "Assistant",
        messages: Optional[Union["Monologue", "FileMonologue", list[MessagePydantic]]] = None,
        limit: int = 600000,
        file_path: Optional[str] = None,
    ):
        super().__init__(
            agent_name=agent_name, 
            messages=messages, 
            limit=limit
            )

        self.file_path = file_path
        if not self.file_path:
            self.file_path = f"/tmp/{self.agent_name}.json"

        if messages:
            if isinstance(self._messages, FileMonologue):
                self._messages = self._messages.list_messages()
            self._save()

    def _save(self):
        message_store = MessageStore(
            file_path=self.file_path,
            MessagePydantic_cls=MessagePydantic,
        )
        message_store.reset()
        message_store.bulk_insert_messages(self._messages)

    def _load(self, path: Optional[str] = None):
        message_store = MessageStore(
            file_path=path or self.file_path,
            MessagePydantic_cls=MessagePydantic,
        )
        self._messages = message_store.list_messages()

    def copy(self):
        """Returns a copy of the file monologue."""
        return FileMonologue(
            agent_name=self.agent_name,
            messages=self._messages.copy()
        )

    def list_messages(self) -> list[MessagePydantic]:
        """
        Returns the list of messages in the monologue.
        """
        self._load()
        return self._messages

    def list_chat_messages(self) -> list[dict[str, Any]]:
        """
        Returns the list of chat messages in the monologue.
        """
        self._load()
        return message_helper.convert_to_chat_messages(self._messages)

    def reset(self, path: Optional[str] = None):
        self.message_store.reset()
        self._messages.clear()

    def add_user_message(self, content: Any, state=None):
        self._load()
        result = super().add_user_message(content, state)
        self._save()
        return result

    def add_assistant_message(self, content: Any, state=None):
        self._load()
        result = super().add_assistant_message(content, state)
        self._save()
        return result

    def pop(self):
        """
        pop the last message
        """
        self._load()
        popped = super().pop()
        self._save()
        return popped

    def update(self, messages: list[MessagePydantic]):
        """
        Update the internal list
        """
        self._load()
        result = super().update(messages)
        self._save()
        return result

    def get_last_toolcalls(self):
        """
        From the last message in the monologue, extract all tool_call content.
        """
        self._load()
        return super().get_last_toolcalls()

    def is_terminated(self):
        """
        From the last message in the monologue, check if last message contains tool_use
        """
        self._load()
        return super().is_terminated()

    def is_interrupted(self, user_message):
        """
        From the last message in the monologue, check if it contains "user_input" tool_use.
        """
        self._load()
        return super().is_interrupted(user_message=user_message)
