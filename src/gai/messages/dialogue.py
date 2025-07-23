import os
from functools import wraps
from typing import Any, Iterable, Optional, Union
from gai.lib.logging import getLogger
from gai.lib.constants import DEFAULT_GUID
from .message_store import MessageStore
from gai.messages import message_helper
from .typing import MessagePydantic

logger = getLogger(__name__)


class Dialogue:
    def __init__(
        self,
        agent_name: Optional[str] = None,
        max_recap_size: int = 60000,
        messages: Optional[Iterable[MessagePydantic]] = None,
    ):
        self.dialogue_id = DEFAULT_GUID
        self.agent_name = agent_name or "User"
        self.max_recap_size = max_recap_size
        self._messages = list(messages) if messages else []

    def list_messages(self) -> list[MessagePydantic]:
        return self._messages.copy()

    def list_chat_messages(self) -> list[dict[str, Any]]:
        """
        Returns the list of chat messages in the monologue.
        """
        return message_helper.convert_to_chat_messages(self._messages)

    def reset(self):
        """
        Reset the dialogue by clearing the messages.
        """
        self._messages.clear()

    def add_user_message(self, recipient: str, content: str) -> MessagePydantic:
        # 1) Find round_no
        # User message's round number is the last user message's round number + 1

        round_no = 0
        reversed_messages = reversed(self._messages.copy())
        last_user_message = None
        for msg in reversed_messages:
            if msg.body.role == "user":
                last_user_message = msg
                break
        if last_user_message:
            round_no = last_user_message.body.round_no + 1

        # 2) Find step_no
        # User message's step number is always 0

        step_no = 0

        # 3) Create the user message
        named_content = f"{recipient}, {content}"
        user_message = MessagePydantic(
            **{
                "header": {
                    "sender": "User",
                    "recipient": recipient,
                },
                "body": {
                    "type": "chat.send",
                    "dialogue_id": self.dialogue_id,
                    "round_no": round_no,
                    "step_no": step_no,
                    "role": "user",
                    "content": named_content,
                },
            }
        )

        self._messages.append(user_message)
        return user_message

    def add_assistant_message(
        self, sender: str, chunk: str, content: Optional[str] = None
    ) -> MessagePydantic:
        if not self._messages or len(self._messages) == 0:
            raise ValueError("The first message must be a user message.")

        # 1) Find round_no
        # Agent message's round number is the same as the last user message's round number

        last_message = self._messages[-1]
        round_no = last_message.body.round_no

        # 2) Find step_no
        # Agent message's step number is the last message's step number + 1

        step_no = last_message.body.step_no + 1

        # 3) Find chunk_no
        # If the last message is an assistant message, then chunk_no is the last message's chunk_no + 1
        # Otherwise, chunk_no is 0

        chunk_no = 0
        if last_message.body.role == "assistant":
            chunk_no = last_message.body.chunk_no + 1

        # 4) Validate chunk and content

        # If content is provided, then chunk must be "<eom>"
        if chunk != "<eom>" and content is not None:
            raise ValueError("If content is provided, chunk must be '<eom>'.")

        # If chunk is "<eom>", then content must not be None
        if chunk == "<eom>" and content is None:
            raise ValueError(
                "If chunk is '<eom>', content can be '' but must not be None."
            )

        # 5) Create the assistant message
        assistant_message = MessagePydantic(
            **{
                "header": {
                    "sender": sender,
                    "recipient": "User",
                },
                "body": {
                    "type": "chat.reply",
                    "dialogue_id": self.dialogue_id,
                    "round_no": round_no,
                    "step_no": step_no,
                    "role": "assistant",
                    "chunk": chunk,
                    "chunk_no": chunk_no,
                    "content": content,
                },
            }
        )

        self._messages.append(assistant_message)
        return assistant_message

    def delete_message(self, message_id: str) -> None:
        """
        Delete a message from the dialogue by its ID.
        """
        self._messages = [m for m in self._messages if m.id != message_id]

    def extract_recap(self, last_n: int = 0) -> str:
        return message_helper.extract_recap(
            self._messages, last_n=last_n, max_recap_size=self.max_recap_size
        )


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


class FileDialogue(Dialogue):
    """
    Dialogue class that extends MessageStore to handle dialogue-specific operations.
    This class is used to manage dialogues in a structured way, inheriting from MessageStore.
    """

    def __init__(
        self,
        dialogue_id: Optional[str] = None,
        agent_name: Optional[str] = None,
        messages: Optional[
            Union["Dialogue", "FileDialogue", list[MessagePydantic]]
        ] = None,
        max_recap_size: int = 60000,
        file_path: Optional[str] = None,
    ):
        """
        Create or load a dialogue.
        """

        # Initialize messages

        if messages:
            if isinstance(messages, Dialogue) or isinstance(messages, FileDialogue):
                messages = messages.list_messages()
            elif isinstance(messages, list):
                messages = messages.copy()
            else:
                raise ValueError(
                    "FileDialogue: messages should be a list of MessagePydantic or a Dialogue instance."
                )

        super().__init__(
            agent_name=agent_name, max_recap_size=max_recap_size, messages=messages
        )

        # Initialize MessageStore

        self.caller_id = DEFAULT_GUID
        self.dialogue_id = dialogue_id or DEFAULT_GUID
        file_path = (
            file_path
            or f"~/.gai/data/{self.caller_id}/User/dialogues/{self.dialogue_id}.json"
        )
        self.file_path = os.path.expanduser(file_path)
        self.message_store = MessageStore[MessagePydantic](
            file_path=self.file_path, MessagePydantic_cls=MessagePydantic
        )
        if self._messages:
            self.message_store.reset()
            self.message_store.bulk_insert_messages(self._messages)

        # if self._messages:
        #     self._save()
        # else:
        #     self._load()

    def _save(self):
        self.message_store.reset()
        self.message_store.bulk_insert_messages(self._messages)

    def _load(self):
        self._messages = self.message_store.list_messages()

    def reset(self):
        self.message_store.reset()
        return super().reset()

    @load_only
    def list_messages(self) -> list[MessagePydantic]:
        return super().list_messages()

    @load_only
    def list_chat_messages(self) -> list[dict[str, Any]]:
        return super().list_chat_messages()

    @transactional
    def add_user_message(self, recipient: str, content: str) -> MessagePydantic:
        return super().add_user_message(recipient, content)

    @transactional
    def add_assistant_message(
        self, sender: str, chunk: str, content: Optional[str] = None
    ) -> MessagePydantic:
        return super().add_assistant_message(sender, chunk, content)

    # @transactional
    # def insert_message(self, message: MessagePydantic) -> None:
    #     super().insert_message(message)

    @transactional
    def delete_message(self, message_id: str) -> None:
        super().delete_message(message_id)

    @load_only
    def extract_recap(self, last_n: int = 0) -> str:
        return super().extract_recap(last_n=last_n)
