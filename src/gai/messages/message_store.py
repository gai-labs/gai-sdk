from threading import Lock
from pydantic import BaseModel, Field
import os
import json
import time

from typing import Generic, Optional, TypeVar, Type, Any
from gai.messages.typing import MessagePydantic
from gai.lib.logging import getLogger

logger = getLogger(__name__)

MessagePydanticT = TypeVar("MessagePydanticT", bound=BaseModel)


class MessageStoreLoadError(Exception):
    def __init__(self, file_path: str, data: str, *, cause: Exception | None = None):
        self.file_path = file_path
        self.data = data
        self.cause = cause
        msg = (
            f"MessageStore: Failed to load internal structure from {file_path} "
            f"with data={data!r}. Creating new one."
        )
        if cause:
            # preserve chaining
            super().__init__(msg + f" Cause: {cause!r}")
        else:
            super().__init__(msg)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(file_path={self.file_path!r}, "
            f"data={self.data!r}, cause={self.cause!r})"
        )


class MessageStore(Generic[MessagePydanticT]):
    class InternalStructure(BaseModel):
        next_message_order: int = 0
        # store raw dicts, not BaseModel instances
        messages: list[dict[str, Any]] = Field(default_factory=list)

    # file lock
    file_lock = Lock()

    def __init__(self, file_path: str, MessagePydantic_cls: Type[MessagePydanticT]):
        self.MessagePydantic_cls = MessagePydantic_cls
        self.file_path = file_path

        # Ensure the directory exists
        file_dir = os.path.dirname(file_path)
        if file_dir:
            os.makedirs(file_dir, exist_ok=True)
            logger.info(f"MessageStore: Created new message directory {file_dir}")

        # Ensure file is created with valid structure
        if not os.path.exists(self.file_path):
            self.reset()
            logger.info(f"MessageStore: Created new message store {file_dir}")
        else:
            logger.info(f"MessageStore: Use existing message store {file_dir}")

    def get_message(self, message_id: str) -> Optional[MessagePydanticT]:
        """Get a message from the dialogue file by its ID."""
        with MessageStore.file_lock:
            if not os.path.exists(self.file_path):
                logger.error(f"MessageStore: file not found. path={self.file_path}")
                raise FileNotFoundError(
                    f"MessageStore: file not found. path={self.file_path}"
                )

            with open(self.file_path, "r") as f:
                try:
                    internal_structure = MessageStore.InternalStructure(**json.load(f))
                except json.JSONDecodeError:
                    logger.warning(
                        f"MessageStore: Failed to load internal structure from {self.file_path}. Creating new one."
                    )
                    internal_structure = MessageStore.InternalStructure()

            for message in internal_structure.messages:
                if message["id"] == message_id:
                    return self.MessagePydantic_cls(**message)
            return None

    def list_messages(self) -> list[MessagePydanticT]:
        """List all messages in the messages store file."""
        try:
            with MessageStore.file_lock:
                if not os.path.exists(self.file_path):
                    logger.warning(
                        f"MessageStore: File not found. path={self.file_path}"
                    )
                    raise FileNotFoundError(
                        f"MessageStore: File not found path={self.file_path}"
                    )

                with open(self.file_path, "r") as f:
                    data = ""
                    try:
                        data = f.read()
                        jsoned = json.loads(data)
                        internal_structure = MessageStore.InternalStructure(**jsoned)
                    except json.JSONDecodeError as e:
                        logger.error(
                            f"MessageStore: Failed to load internal structure from {self.file_path} with data='{data}'. Creating new one."
                        )
                        raise MessageStoreLoadError(
                            self.file_path, data, cause=e
                        ) from e
        except MessageStoreLoadError as e:
            """
            Recreate the file with an empty structure
            """
            logger.error(
                f"MessageStore: Recreating file {self.file_path} due to load error: {e}"
            )
            self.reset()
            internal_structure = MessageStore.InternalStructure()

        return [self.MessagePydantic_cls(**msg) for msg in internal_structure.messages]

    def insert_message(self, message: MessagePydanticT):
        """Insert a message into the messages store file."""
        with MessageStore.file_lock:
            # Read existing data from file
            if os.path.exists(self.file_path):
                with open(self.file_path, "r") as f:
                    try:
                        internal_structure = MessageStore.InternalStructure(
                            **json.load(f)
                        )
                    except json.JSONDecodeError:
                        logger.warning(
                            f"MessageStore: Failed to load internal structure from {self.file_path}. Creating new one."
                        )
                        internal_structure = MessageStore.InternalStructure()
            else:
                internal_structure = MessageStore.InternalStructure()

            # Update internal structure

            message.header.order = internal_structure.next_message_order
            internal_structure.messages.append(message.model_dump())
            internal_structure.next_message_order += 1

            # Save internal structure back to file
            with open(self.file_path, "w") as f:
                jsoned = internal_structure.model_dump()
                f.write(json.dumps(jsoned, indent=4))

            logger.debug(
                f"MessageStore: Message added to file. path={self.file_path}. message={message}"
            )

    def bulk_insert_messages(self, messages: list):
        """Insert multiple messages into the messages store file."""
        with MessageStore.file_lock:
            if not messages:
                logger.warning(
                    f"MessageStore: No messages to insert. path={self.file_path}"
                )
                return

            # Read existing data from file

            with open(self.file_path, "r") as f:
                try:
                    internal_structure = MessageStore.InternalStructure(**json.load(f))
                except json.JSONDecodeError:
                    logger.warning(
                        f"MessageStore: Failed to load internal structure from {self.file_path}. Creating new one."
                    )
                    internal_structure = MessageStore.InternalStructure()

            # Update internal structure

            for message in messages:
                message.header.order = internal_structure.next_message_order
                internal_structure.messages.append(message.model_dump())
                internal_structure.next_message_order += 1

            # Save updated internal structure back to file
            with open(self.file_path, "w") as f:
                jsoned = internal_structure.model_dump()
                f.write(json.dumps(jsoned, indent=4))

            logger.debug(
                f"MessageStore: {len(messages)} messages added to file. path={self.file_path}"
            )

    def delete_message(self, id: str):
        """Delete a message from the messages store file."""
        with MessageStore.file_lock:
            # Read existing data from file

            with open(self.file_path, "r") as f:
                try:
                    internal_structure = MessageStore.InternalStructure(**json.load(f))
                except json.JSONDecodeError:
                    logger.warning(
                        f"MessageStore: Failed to load internal structure from {self.file_path}. Creating new one."
                    )
                    internal_structure = MessageStore.InternalStructure()

            # Filter out the message with the given id
            previous_length = len(internal_structure.messages)
            internal_structure.messages = [
                msg for msg in internal_structure.messages if msg["id"] != id
            ]
            if len(internal_structure.messages) == previous_length:
                logger.warning(
                    f"MessageStore: Message with id={id} not found in file. No deletion performed. path={self.file_path}"
                )
                return

            # Save updated internal structure back to file
            with open(self.file_path, "w") as f:
                jsoned = internal_structure.model_dump()
                f.write(json.dumps(jsoned, indent=4))
            logger.debug(
                f"MessageStore: Message deleted from file. id={id} path={self.file_path}"
            )

    def update_message(self, message: MessagePydanticT):
        """Update a message in the messages store file."""
        with MessageStore.file_lock:
            # Read existing data from file

            with open(self.file_path, "r") as f:
                try:
                    internal_structure = MessageStore.InternalStructure(**json.load(f))
                except json.JSONDecodeError:
                    logger.warning(
                        f"MessageStore: Failed to load internal structure from {self.file_path}. Creating new one."
                    )
                    internal_structure = MessageStore.InternalStructure()

            updated = False
            for i, msg in enumerate(internal_structure.messages):
                if msg["id"] == message.id:
                    internal_structure.messages[i] = (
                        message.model_dump()
                    )  # ✅ REPLACE instead of mutating
                    updated = True
                    break

            if updated:
                # Save updated internal structure back to file
                with open(self.file_path, "w") as f:
                    jsoned = internal_structure.model_dump()
                    f.write(json.dumps(jsoned, indent=4))
                logger.debug(
                    f"MessageStore: Message updated in file. message={message} path={self.file_path}"
                )

    def reset(self):
        """Reset the file by clearing all messages."""
        with MessageStore.file_lock:
            with open(self.file_path, "w") as f:
                initial = MessageStore.InternalStructure().model_dump()
                f.write(json.dumps(initial, indent=4))
            logger.debug(
                f"MessageStore: File reset. all messages cleared. path={self.file_path}"
            )
