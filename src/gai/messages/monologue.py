from functools import wraps
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
        limit: int = 300000,
    ):
        self.agent_name = agent_name
        self.limit = limit  # Character limit for messages
        self._messages: list[MessagePydantic] = messages
        if self._messages:
            if isinstance(self._messages, Monologue):
                self._messages = self._messages.list_messages().copy()
        else:
            self._messages = []

    def list_chat_messages(self) -> list[dict[str, Any]]:
        """
        Returns the list of chat messages in the monologue.
        """
        messages_copy = self._messages.copy()

        messages_copy = self.shrink_messages(messages_copy)
        return message_helper.convert_to_chat_messages(messages_copy)

    def is_new(self):
        """
        Check if the monologue is new.
        A monologue is considered new if it has no messages or only contains a system message.
        """
        if not self._messages:
            return True
        return False

    def get_messages_length(self, messages: Optional[list[MessagePydantic]] = None) -> int:
        """
        Returns the length of the messages in characters.
        If no messages are provided, it uses the internal list of messages.
        """
        if not messages:
            messages = self._messages
        return sum(len(json.dumps(msg.model_dump())) for msg in messages)

    def shrink_messages(self, messages_copy: list[MessagePydantic]) -> list[MessagePydantic]:
        """
        Shrink the messages to fit within the character limit.
        This method will save the earliest user message and the earliest assistant message, then remove
        all messages from the earliest to the current until the total size is within self.limit.
        """

        if len(messages_copy) < 2:
            # No need to shrink if there are less than 2 messages
            return messages_copy

        earliest_user_message = messages_copy.pop(0)
        earliest_user_message_len = len(
            json.dumps(earliest_user_message.model_dump()))
        earliest_assistant_message = messages_copy.pop(0)
        earliest_assistant_message_len = len(
            json.dumps(earliest_assistant_message.model_dump()))

        remaining_length = self.limit - \
            (earliest_user_message_len + earliest_assistant_message_len)
        while self.get_messages_length(messages_copy) > remaining_length and len(messages_copy) > 2:
            # Remove the user and assistant messages in pairs from the earliest
            if len(messages_copy) > 2:
                messages_copy.pop(0)
                messages_copy.pop(0)
            else:
                raise Exception(
                    "Monologue: content size is bigger than 600,000 char. Are you sending an image?"
                )

        # Add the earliest user and assistant messages back
        messages_copy.insert(0, earliest_user_message)
        messages_copy.insert(1, earliest_assistant_message)

        return messages_copy

    def add_user_message(self, content: Any, state=None):
        state_name = ""
        step_no = -1
        if state:
            state_name = state.title
            step_no = state.input["step"]

        message = MessagePydantic(
            header=MessageHeaderPydantic(
                sender="User", recipient=self.agent_name),
            body=MonologueBodyPydantic(
                state_name=state_name,
                step_no=step_no,
                role="user",
                content=content,
            ),
        )
        self._messages.append(message)
        return self

    def add_assistant_message(self, content: Any, state=None):
        state_name = ""
        step_no = -1
        if state:
            state_name = state.title
            step_no = state.input["step"]

        message = MessagePydantic(
            header=MessageHeaderPydantic(
                sender=self.agent_name, recipient="User"),
            body=MonologueBodyPydantic(
                state_name=state_name,
                step_no=step_no,
                role="assistant",
                content=content,
            ),
        )
        self._messages.append(message)
        return self

    def copy(self):
        """Returns a copy of the monologue."""
        return Monologue(agent_name=self.agent_name, messages=self._messages.copy())

    def list_messages(self) -> list[MessagePydantic]:
        messages_copy = self._messages.copy()
        messages_copy = self.shrink_messages(messages_copy)
        return messages_copy

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
        self._messages = []

    def get_last_toolcalls(self):
        """
        This is used for creating tool results.
        The purpose is to extract the tool_use_id from the last
        valid tool call, if any.
        """

        if len(self._messages) <= 1 and self._messages[0].body.role == "user":
            # If this is the first message or its empty, then there are naturally no tool calls.
            # ie. called by ChatState
            return None

        messages_copy = self._messages.copy()
        last_assistant_message = None
        while messages_copy:
            if messages_copy[-1].body.role != "assistant":
                messages_copy.pop()
            else:
                last_assistant_message = messages_copy[-1]
                break
        if not last_assistant_message:
            # If there are no assistant messages, then there are no tool calls.
            return None

        # If a tool call is present in the last message,
        # it will be found in a list of content
        # where type is "tool_use".

        tool_calls = []
        if isinstance(last_assistant_message.body.content, list):
            for item in last_assistant_message.body.content:
                if isinstance(item, str):
                    continue
                if not isinstance(item, dict):
                    item = item.model_dump()
                if "type" in item and item["type"] == "tool_use":
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

        # If there are no tool calls, then it is considered terminated or interrupted.
        last_tool_calls = self.get_last_toolcalls()
        if not last_tool_calls:
            return True

        # obsolete: If last message contains "task_completed" tool call, then it is considered terminated.
        # This is not fool proof since LLM might missed the tool call.
        # The worst case is that we can't tell if it is terminated or interrupted.
        # Which is not a big deal since we can always retry.
        if last_tool_calls and any(
            result["tool_name"] == "task_completed" for result in last_tool_calls
        ):
            return True

        # If user send "TERMINATE", then it is considered terminated.
        last_message = self._messages[-1] if self._messages else None
        last_content = last_message.body.content
        if (
            last_message.body.role == "user"
            and isinstance(last_content, str)
            and last_content.upper() == "TERMINATE"
        ):
            return True

        return False


# -----


def transactional(method):
    """load before, save after."""

    @wraps(method)
    def _wrapped(self, *args, **kwargs):
        self._load()
        result = method(self, *args, **kwargs)
        self._save()
        return result

    return _wrapped


def load_only(method):
    """load before, no save."""

    @wraps(method)
    def _wrapped(self, *args, **kwargs):
        self._load()
        return method(self, *args, **kwargs)

    return _wrapped


class FileMonologue(Monologue):
    def __init__(
        self,
        file_path: str,
        agent_name: str = "Assistant",
        messages: Optional[
            Union["Monologue", "FileMonologue", list[MessagePydantic]]
        ] = None,
        limit: int = 300000,
        force: bool = False,
    ):
        super().__init__(agent_name=agent_name, messages=messages, limit=limit)

        # Use `force` when loading from an incorrect format file

        self.force = force

        # Initialize MessageStore

        self.file_path = file_path
        self.message_store = MessageStore(
            file_path=self.file_path,
            MessagePydantic_cls=MessagePydantic,
        )

        # Initialize messages
        self._messages: list[MessagePydantic] = messages
        if self._messages:
            if isinstance(self._messages, FileMonologue) or isinstance(
                self._messages, Monologue
            ):
                self._messages = self._messages.list_messages()
            elif isinstance(messages, list):
                self._messages = []
                for m in messages:
                    if isinstance(m, MessagePydantic):
                        self._messages.append(m.copy())
                    elif isinstance(m, dict):
                        self._messages.append(MessagePydantic(**m))
                    else:
                        raise ValueError(
                            "FileMonologue: messages should be a list of MessagePydantic or a Monologue/FileMonologue instance."
                        )
            else:
                raise ValueError(
                    "FileMonologue: messages should be a list of MessagePydantic or a Monologue/FileMonologue instance."
                )

            self._save()
        else:
            # Force load from an incompatible file

            try:
                self._load()
            except Exception as e:
                if not self.force:
                    logger.error(
                        f"FileMonologue.__init__: Failed to load. Error={str(e)}. Use force=True to overwrite an empty file."
                    )

    def _save(self):
        self.message_store.reset()
        self.message_store.bulk_insert_messages(self._messages)

    def _load(self, path: Optional[str] = None):
        self._messages = self.message_store.list_messages()

    def reset(self):
        self.message_store.reset()
        return super().reset()

    def copy(self):
        """Returns a copy of the file monologue."""
        return FileMonologue(
            agent_name=self.agent_name,
            messages=self._messages.copy(),
            file_path=self.file_path,
        )

    @load_only
    def list_messages(self) -> list[MessagePydantic]:
        return super().list_messages()

    @load_only
    def list_chat_messages(self) -> list[dict[str, Any]]:
        return super().list_chat_messages()

    @transactional
    def add_user_message(self, content: Any, state=None):
        return super().add_user_message(content, state)

    @transactional
    def add_assistant_message(self, content: Any, state=None):
        return super().add_assistant_message(content, state)

    @transactional
    def pop(self):
        return super().pop()

    @transactional
    def update(self, messages: list[MessagePydantic]):
        return super().update(messages)

    @load_only
    def get_last_toolcalls(self):
        return super().get_last_toolcalls()

    @load_only
    def is_terminated(self):
        return super().is_terminated()
