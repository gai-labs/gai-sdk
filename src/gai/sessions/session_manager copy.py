import os
import json
import copy
import asyncio
import uuid
import time
from typing import (
    Any,
    Callable,
    overload,
    Optional,
    Union,
    Literal,
    TypeVar,
    Generic,
    Type,
)
from pydantic import BaseModel

from gai.lib.constants import DEFAULT_GUID
from gai.lib.utils import get_app_path
from gai.lib.logging import getLogger
from gai.messages.dialogue import Dialogue, FileDialogue
from gai.sessions.message_bus_base import BaseMessageBus
from gai.messages.typing import MessagePydantic
from gai.sessions.message_bus_local import LocalMessageBus

logger = getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class LocalSessionManager(Generic[T]):
    """
    Higher-level orchestrator built on top of the message bus to manage a conversation or dialogue.
    Unlike message bus, the session manager is associated with the underlying message type and manages the dialogue state for the session.
    """

    def __init__(
        self,
        message_type: Type[T],
        # Core configuration
        logger_name: str = "User",
        dialogue_id: Optional[str] = None,
        max_recap_size: int = 4096,
        queue_maxsize: int = 100,
        reset: bool = False,
        message_bus: Optional[BaseMessageBus] = None,
        file_path: Optional[str] = None,
    ):
        """
        Initialize LocalSessionManager.

        Args:
            message_type: The message class type to use
            logger_name: Agent/perspective name for this dialogue
            dialogue_id: Unique identifier (auto-generated if not provided)
            max_recap_size: Maximum size for conversation recap
            transport: Message bus transport type ('local' or 'nats')
            queue_maxsize: Queue size for local transport
            servers: NATS servers for network transport
            file_path: Optional file path for persistent storage
            reset: Clear existing storage on initialization
            bus_transport: (Legacy) Use 'transport' instead
            nats_servers: (Legacy) Use 'servers' instead
        """

        # Core attributes
        self.message_type = message_type
        self.logger_name = logger_name
        self.dialogue_id = dialogue_id or str(uuid.uuid4())
        self.max_recap_size = max_recap_size
        self.messages: list[T] = []

        # Message bus
        self.bus = message_bus
        if not message_bus:
            self.bus = LocalMessageBus(
                message_type=message_type, queue_maxsize=queue_maxsize
            )

        # Storage setup
        if file_path:
            self.dialogue = FileDialogue(file_path=file_path)
            if reset:
                self.dialogue.reset()
            # Load existing messages
            try:
                self.messages = self.dialogue.list_messages()
            except Exception as e:
                logger.warning(f"Could not load messages from storage: {e}")
                self.messages = []
        else:
            self.dialogue = Dialogue(agent_name=logger_name)

    # === Context Manager Support ===

    async def __aenter__(self):
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        if exc_type:
            logger.warning(
                f"LocalSessionManager: __aexit__ triggered due to exception. exc_type={exc_type}, exc={exc}"
            )
        else:
            logger.debug("LocalSessionManager: __aexit__ triggered normally.")

        try:
            await self.stop()
        except Exception as e:
            logger.error(f"LocalSessionManager: Error in __aexit__: {e}")

    # === Lifecycle Management ===

    async def start(self):
        """Start the dialogue manager."""
        await self.bus.start()
        logger.info("LocalSessionManager: ready.")

    async def stop(self):
        """Stop the dialogue manager."""
        logger.debug("LocalSessionManager: Stopping...")

        try:
            if self.bus:
                await self.bus.stop()
        except Exception as e:
            logger.error(f"LocalSessionManager: Error during stop: {e}")
        finally:
            logger.info("LocalSessionManager: stopped.")

    # === Message Bus Operations (maintaining original interface) ===

    async def subscribe(self, subject: str, callback: dict[str, Callable]):
        """
        Subscribe to a subject with named callbacks.

        Args:
            subject: Subject pattern to subscribe to
            callback: Dictionary of {name: callback_function} pairs
        """
        if not isinstance(callback, dict):
            raise TypeError(
                "LocalSessionManager: subscribe() requires a dictionary of callbacks."
            )

        await self.bus.subscribe(subject, callback)

    async def unsubscribe(self, subject: str, subscriber_name: str):
        """Unsubscribe a subscriber from a subject."""
        await self.bus.unsubscribe(subject, subscriber_name)

    async def unsubscribe_all(self):
        """Remove all subscribers."""
        await self.bus.unsubscribe_all()

    def is_subscribed(self, subject: str, subscriber_name: str) -> bool:
        """Check if a subscriber is subscribed to a subject."""
        return self.bus.is_subscribed(subject, subscriber_name)

    # === Publishing (maintaining original overloads) ===

    @overload
    async def publish(self, pydantic: T) -> T: ...

    @overload
    async def publish(self, type: str, body: str, sender: str, recipient: str) -> T: ...

    async def publish(
        self,
        *,
        pydantic: Optional[T] = None,
        type: Optional[str] = None,
        body: Optional[str] = None,
        sender: Optional[str] = None,
        recipient: Optional[str] = None,
    ) -> T:
        """
        Publish a message to the dialogue bus.

        Can be called either with a message object or with individual parameters.
        """

        # Create or validate message
        message = self._prepare_message(pydantic, type, body, sender, recipient)

        # Handle chunk filtering
        try:
            await self.bus.publish(message)
            return message

        except Exception as e:
            logger.error(f"LocalSessionManager: Error during publish: {e}")
            raise

    def _prepare_message(
        self,
        pydantic: Optional[T],
        type: Optional[str],
        body: Optional[str],
        sender: Optional[str],
        recipient: Optional[str],
    ) -> T:
        """Prepare message for publishing (replaces _validate_message_copy)."""
        if pydantic:
            return copy.deepcopy(pydantic)
        elif all([type, body, sender, recipient]):
            return self.message_type(
                id=str(uuid.uuid4()),
                header={
                    "sender": sender,
                    "recipient": recipient,
                    "timestamp": time.time(),
                },
                body={"type": type, "content": body},
            )
        else:
            raise TypeError(
                "LocalSessionManager: publish() requires either `pydantic` or all of `type`, `body`, `sender`, and `recipient` as keyword arguments."
            )

    def _should_log_message(self, pydantic: T) -> bool:
        """Determine if message should be logged to dialogue history."""
        # Don't log mid-stream chunks
        if pydantic.body.type in ("reply", "chat.reply"):
            return getattr(pydantic.body, "chunk", "<eom>") == "<eom>"
        return True

    # === State and Message Management ===

    def is_started(self) -> bool:
        """Check if the dialogue manager is started."""
        return self.bus.is_started if self.bus else False

    def list_messages(self) -> list[T]:
        """Get all messages in dialogue history."""
        return self.messages

    def log_message(self, pydantic: T):
        """Add a message to dialogue history."""
        logger.debug(f"LocalSessionManager: message={pydantic}")

        # Check if message is already appended before appending to avoid duplicating message
        if any(m for m in self.messages if pydantic.id == m.id):
            logger.debug(
                f"LocalSessionManager: message with id={pydantic.id} already logged, skipping."
            )
            return

        self.messages.append(pydantic)

        # Prepare message for dialogue
        if pydantic.body.type == "chat.send":
            try:
                self.dialogue.add_user_message(
                    sender=pydantic.header.sender,
                    recipient=pydantic.header.recipient,
                    content=pydantic.body.content,
                )
            except Exception as e:
                logger.error(f"Failed to persist chat.send message to storage: {e}")

        if pydantic.body.type == "chat.reply":
            try:
                self.dialogue.add_assistant_message(
                    sender=pydantic.header.sender,
                    recipient=pydantic.header.recipient,
                    chunk=pydantic.body.chunk,
                    content=pydantic.body.content,
                )
            except Exception as e:
                logger.error(f"Failed to persist chat.reply message to storage: {e}")

    def extract_recap(self, last_n: int = 0) -> str:
        """Extract conversation recap for LLM context."""
        from gai.messages.message_helper import extract_recap

        return extract_recap(
            self.messages, last_n=last_n, max_recap_size=self.max_recap_size
        )

    def reset(self):
        """Reset the dialogue by clearing all messages."""
        self.messages.clear()
        logger.debug("LocalSessionManager: Dialogue reset. all messages cleared.")

        if self.dialogue:
            self.dialogue.reset()

    def delete_message(self, message_id: str):
        """Delete a message by its ID."""
        self.messages = [m for m in self.messages if m.id != message_id]
        logger.debug(f"LocalSessionManager: Message deleted. message_id= {message_id}")

        if self.dialogue:
            self.dialogue.delete_message(message_id)


def make_session_manager(message_cls: Type[T]) -> Type[LocalSessionManager[T]]:
    """
    Factory function to create a dialogue manager with a specific message class.

    Args:
        message_cls: The message class type to use

    Returns:
        A SessionManager class bound to the specified message type

    Usage:
        SessionManager = make_session_manager(MessagePydantic)
        dm = SessionManager(logger_name="User", ...)
    """

    class SessionManager(LocalSessionManager[T]):
        def __init__(self, **kwargs):
            if "message_type" in kwargs:
                kwargs.pop("message_type")
            super().__init__(message_type=MessagePydantic, **kwargs)

    return SessionManager


# Maintain backward compatibility

from gai.messages.typing import MessagePydantic

SessionManager = make_session_manager(MessagePydantic)
