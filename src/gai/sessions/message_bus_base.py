import asyncio
import inspect
import json
from collections import defaultdict
from typing import (
    Callable,
    Dict,
    Optional,
    TypeAlias,
    Union,
    Protocol,
    Generic,
    TypeVar,
    Type,
)
from gai.lib.logging import getLogger
from pydantic import ValidationError

logger = getLogger(__name__)

T = TypeVar("T")
MessageInput: TypeAlias = Union[dict, T]


class MessageBusProtocol(Protocol):
    """Protocol defining the interface that all message bus implementations must follow."""

    async def start(self) -> None:
        """Start the message bus."""
        ...

    async def stop(self) -> None:
        """Stop the message bus."""
        ...

    async def subscribe(self, subject: str, callback: Dict[str, Callable]) -> None:
        """Subscribe to a subject with named callbacks."""
        ...

    async def publish(self, message: MessageInput) -> None:
        """Publish a message to the bus."""
        ...

    async def unsubscribe(self, subject: str, subscriber_name: str) -> None:
        """Unsubscribe a named subscriber from a subject."""
        ...

    async def unsubscribe_all(self) -> None:
        """Remove all subscribers."""
        ...


class BaseMessageBus(Generic[T]):
    """
    Base class providing common functionality for all message bus implementations.

    This class contains shared logic for subscriber management, message validation,
    and utility functions that are transport-agnostic.
    """

    def __init__(self, message_type: Type[T]):
        """
        Initialize base message bus.

        Args:
            message_type: The message type class to use for validation
        """
        self.message_type = message_type

        # Common attributes for all implementations
        self.subscribers: Dict[str, Dict[str, Callable]] = defaultdict(dict)
        self.ready_event = asyncio.Event()
        self.is_started = False

    def is_subscribed(self, subject: str, subscriber_name: str) -> bool:
        """
        Check if a subscriber is subscribed to a subject.

        Args:
            subject: The subject pattern
            subscriber_name: Name of the subscriber

        Returns:
            True if subscriber is subscribed to the subject
        """
        return subscriber_name in self.subscribers.get(subject, {})

    def _validate_message(self, message: Union[dict, T]) -> T:
        """
        Validate and convert message to proper type.

        Args:
            message: Message to validate (dict or typed message)

        Returns:
            Validated message of type T

        Raises:
            ValidationError: If message validation fails
            ValueError: If message type is not registered
        """
        if isinstance(message, dict):
            try:
                return self.message_type.model_validate(message)
            except ValidationError as e:
                # Check if this is an unregistered message type error
                for error in e.errors():
                    if error.get("type") == "union_tag_invalid" and "body" in error.get(
                        "loc", []
                    ):
                        input_value = error.get("input", {})
                        if isinstance(input_value, dict) and "type" in input_value:
                            message_type = input_value["type"]
                            raise ValueError(
                                f"Message type '{message_type}' is not registered."
                            )
                raise
        return message

    def _validate_subscription_params(
        self, subject: str, subscriber: Dict[str, Callable]
    ) -> None:
        """
        Validate subscription parameters.

        Args:
            subject: Subject pattern to subscribe to
            subscriber: Dictionary of {name: callback} pairs

        Raises:
            TypeError: If parameters have wrong types
            ValueError: If parameters are empty or invalid
            RuntimeError: If bus is not started
        """
        if not isinstance(subscriber, dict):
            raise TypeError("subscriber must be a dictionary of name to callback.")
        if not isinstance(subject, str):
            raise TypeError("subject must be a string.")
        if not subject:
            raise ValueError("subject cannot be empty.")
        if not subscriber:
            raise ValueError("subscriber cannot be empty.")
        if not self.is_started:
            raise RuntimeError(
                "Bus not started. Call `await bus.start()` before subscribing."
            )

    def _matches(self, pattern: str, subject: str) -> bool:
        """
        Pattern matching for subject routing (supports wildcards).

        Supports NATS-style wildcards:
        - '*' matches exactly one token
        - '>' matches one or more tokens (only at end)

        Args:
            pattern: Pattern with possible wildcards
            subject: Subject to match against

        Returns:
            True if subject matches pattern
        """
        pattern_tokens = pattern.split(".")
        subject_tokens = subject.split(".")

        pi = si = 0
        while pi < len(pattern_tokens):
            pt = pattern_tokens[pi]
            if pt == ">":
                # '>' must be last token and matches remaining subject tokens
                return pi == len(pattern_tokens) - 1 and si < len(subject_tokens)
            if si >= len(subject_tokens):
                return False
            if pt != "*" and pt != subject_tokens[si]:
                return False
            pi += 1
            si += 1
        return si == len(subject_tokens)

    async def _safe_call(self, callback: Callable, message: T) -> asyncio.Task:
        """
        Safely call a callback and return the task.

        Handles both sync and async callbacks, including async generators.
        Logs errors without propagating them to avoid breaking other subscribers.

        Args:
            callback: The callback function to call
            message: Message to pass to callback

        Returns:
            Task representing the callback execution
        """

        async def _wrapped_call():
            try:
                if inspect.isasyncgenfunction(callback):
                    # Handle async generator callbacks
                    agen = callback(message)
                    try:
                        async for chunk in agen:
                            logger.debug(f"Chunk from {callback}: {chunk!r}")
                    finally:
                        await agen.aclose()
                elif inspect.iscoroutinefunction(callback):
                    # Handle async function callbacks
                    await callback(message)
                else:
                    # Handle sync function callbacks
                    await asyncio.to_thread(callback, message)
            except Exception as e:
                logger.exception(f"Subscriber error in {callback}: {e}")

        return asyncio.create_task(_wrapped_call())

    def _cleanup_subscriber_references(
        self, subject: str, subscriber_name: str
    ) -> None:
        """
        Clean up subscriber references from internal data structures.

        Args:
            subject: Subject to clean up
            subscriber_name: Subscriber name to remove
        """
        try:
            if self.subscribers.get(subject, {}).get(subscriber_name):
                del self.subscribers[subject][subscriber_name]
                # Clean up empty subject entries
                if not self.subscribers[subject]:
                    del self.subscribers[subject]
        except KeyError:
            # Already removed, which is fine
            pass

    def _get_subscriber_count(self) -> int:
        """Get total number of subscribers across all subjects."""
        return sum(len(callbacks) for callbacks in self.subscribers.values())

    def _get_subjects(self) -> list[str]:
        """Get list of all subjects with active subscribers."""
        return list(self.subscribers.keys())

    def _log_subscriber_info(
        self, action: str, subject: str, subscriber_name: str
    ) -> None:
        """
        Log subscriber action with consistent format.

        Args:
            action: Action performed (subscribed, unsubscribed, etc.)
            subject: Subject involved
            subscriber_name: Name of subscriber
        """
        logger.debug(
            f"MessageBus: {action} {subscriber_name} to/from subject '{subject}'"
        )


class MessageBusError(Exception):
    """Base exception for message bus errors."""

    pass


class TransportError(MessageBusError):
    """Error related to transport layer."""

    pass


class SubscriptionError(MessageBusError):
    """Error related to subscription management."""

    pass


class PublishError(MessageBusError):
    """Error related to message publishing."""

    pass
