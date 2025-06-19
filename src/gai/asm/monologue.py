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
    StateBodyPydantic,
    MessageHeaderPydantic,
)

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

    def is_terminated(self):
        """
        Check if the monologue is terminated.
        A monologue is considered terminated if it contains a user message with content="TERMINATE"
        """
        for message in self._messages:
            if (
                message.body.role == "user"
                and isinstance(message.body.content, str)
                and message.body.content.strip().upper() == "TERMINATE"
            ):
                return True
        return False

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
            body=StateBodyPydantic(
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
            body=StateBodyPydantic(
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
        result = super().add_user_message(content, state)
        self._save(self.path)
        return result

    def add_assistant_message(self, content: Any, state=None):
        result = super().add_assistant_message(content, state)
        self._save(self.path)
        return result

    def pop(self):
        """
        pop the last message
        """
        popped = super().pop()
        self._save(self.path)
        return popped

    def update(self, messages: list[MessagePydantic]):
        """
        Update the internal list
        """
        result = super().update(messages)
        self._save(self.path)
        return result
