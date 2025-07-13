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
from gai.messages import MessageStore

logger = getLogger(__file__)


class Monologue:
    def __init__(
        self,
        agent_name: str = "Assistant",
        messages: Optional[Union["Monologue", list[MessagePydantic]]] = None,
        dialogue_id: str = DEFAULT_GUID,
        limit: int = 600000,
    ):
        self.limit = limit  # Character limit for messages
        self.dialogue_id = dialogue_id
        self.agent_name = agent_name

        self._messages: list[MessagePydantic] = []
        if isinstance(messages, Monologue):
            self._messages = messages.list_messages()
        else:
            self._messages = messages or []

        self.created_at = int(time.time())
        self.updated_at = int(time.time())

    def get_total_size(self, new_message: Optional[dict] = None):
        chat_messages = self.list_chat_messages()
        total_size = len(json.dumps(chat_messages))
        if new_message:
            total_size += len(json.dumps(new_message))
        return total_size

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
            messages=self._messages.copy(),
            dialogue_id=self.dialogue_id,
        )

    def list_messages(self) -> list[MessagePydantic]:
        return self._messages

    def list_chat_messages(self) -> list[dict[str, Any]]:
        chat_messages = [
            {"role": m.body.role, "content": m.body.content} for m in self._messages
        ]

        # clean up whitespace from system messages
        for message in chat_messages:
            if message["role"] == "system":
                message["content"] = re.sub(r"\s+", " ", message["content"])

        return chat_messages

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
        messages: Optional[Union["Monologue", list[MessagePydantic]]] = None,
        dialogue_id: str = DEFAULT_GUID,
        file_path: Optional[str] = None,
    ):
        super().__init__(agent_name, messages, dialogue_id)
        self.path = f"/tmp/{self.agent_name}.json"
        if file_path:
            self.path = file_path
        self._load(self.path)

    def _save(self, path: Optional[str] = None):
        if not path:
            path = f"/tmp/{self.agent_name}.json"
        if os.path.exists(path):
            with open(path, "w") as f:
                jsoned = json.dumps([m.model_dump() for m in self._messages], indent=4)
                f.write(jsoned)

    def _load(self, path: Optional[str] = None):
        if not path:
            path = f"/tmp/{self.agent_name}.json"
        if not os.path.exists(path):
            # Create empty monologue file if its not available.
            self.reset(path)
        with open(path, "r") as f:
            result = json.load(f)
            self._messages = [MessagePydantic(**m) for m in result]

    def copy(self):
        """Returns a copy of the file monologue."""
        return FileMonologue(
            agent_name=self.agent_name,
            messages=self._messages.copy(),
            dialogue_id=self.dialogue_id,
        )

    def list_messages(self) -> list[MessagePydantic]:
        """
        Returns the list of messages in the monologue.
        """
        self._load(self.path)
        return super().list_messages()

    def list_chat_messages(self) -> list[dict[str, Any]]:
        """
        Returns the list of chat messages in the monologue.
        """
        self._load(self.path)
        return super().list_chat_messages()

    def reset(self, path: Optional[str] = None):
        self._messages.clear()
        if not path:
            path = self.path
        dir_name = os.path.dirname(path)
        with tempfile.NamedTemporaryFile("w", dir=dir_name, delete=False) as tmp:
            tmp.write(json.dumps([]))
            tmp.flush()
            os.fsync(tmp.fileno())
        os.replace(tmp.name, path)
        time.sleep(1)

    def add_user_message(self, content: Any, state=None):
        self._load(self.path)
        result = super().add_user_message(content, state)
        self._save(self.path)
        return result

    def add_assistant_message(self, content: Any, state=None):
        self._load(self.path)
        result = super().add_assistant_message(content, state)
        self._save(self.path)
        return result

    def pop(self):
        """
        pop the last message
        """
        self._load(self.path)
        popped = super().pop()
        self._save(self.path)
        return popped

    def update(self, messages: list[MessagePydantic]):
        """
        Update the internal list
        """
        self._load(self.path)
        result = super().update(messages)
        self._save(self.path)
        return result

    def get_last_toolcalls(self):
        """
        From the last message in the monologue, extract all tool_call content.
        """
        self._load(self.path)
        return super().get_last_toolcalls()

    def is_terminated(self):
        """
        From the last message in the monologue, check if last message contains tool_use
        """
        self._load(self.path)
        return super().is_terminated()

    def is_interrupted(self, user_message):
        """
        From the last message in the monologue, check if it contains "user_input" tool_use.
        """
        self._load(self.path)
        return super().is_interrupted(user_message=user_message)
