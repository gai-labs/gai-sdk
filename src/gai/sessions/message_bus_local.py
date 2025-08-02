import asyncio
from collections import defaultdict
from typing import (
    Callable,
    Dict,
    Optional,
    Union,
    Generic,
    TypeVar,
    Type,
)
from gai.lib.logging import getLogger
from gai.sessions.message_bus_base import BaseMessageBus, MessageBusProtocol

logger = getLogger(__name__)

T = TypeVar("T")


class LocalMessageBus(BaseMessageBus[T], MessageBusProtocol, Generic[T]):
    """
    Local transport message bus implementation.

    Provides in-process message routing using asyncio queues and tasks.
    Supports pattern matching, wildcards, and reliable message delivery
    within a single process.

    Usage:
        bus = LocalMessageBus(message_type=MessagePydantic, queue_maxsize=100)
        await bus.start()
        await bus.subscribe("user.*", {"handler": my_callback})
        await bus.publish(message)
    """

    def __init__(
        self,
        message_type: Type[T],
        *,
        queue_maxsize: int = 100,
    ):
        """
        Initialize LocalMessageBus.

        Args:
            message_type: The message type class to use for validation
            queue_maxsize: Size limit for the internal dispatch queue
        """
        super().__init__(message_type)

        self.queue_maxsize = queue_maxsize

        # Local transport specific attributes
        self.subscriber_tasks: Dict[str, Dict[str, list[asyncio.Task]]] = defaultdict(
            dict
        )
        self.dispatch_queue = asyncio.Queue(maxsize=queue_maxsize)
        self.lock = asyncio.Lock()
        self.start_stop_lock = asyncio.Lock()
        self.broadcast_allowed = True
        self.amb_task = None

    async def start(self, timeout: Optional[float] = None):
        """
        Start the local message bus.

        Args:
            timeout: Optional timeout for dispatch loop operations
        """
        if self.is_started:
            logger.warning(
                "LocalMessageBus: start() called but bus is already started."
            )
            return

        async with self.start_stop_lock:
            self.amb_task = asyncio.create_task(self._dispatch_loop(timeout=timeout))
            await self.ready_event.wait()

        self.is_started = True
        logger.info("LocalMessageBus: Started with local transport.")

    async def stop(self):
        """Stop the local message bus with proper cleanup order."""
        if not self.is_started:
            logger.debug("LocalMessageBus: Already stopped.")
            return

        logger.debug("LocalMessageBus: Shutting down.")

        try:
            # STEP 1: Unsubscribe all subscribers FIRST
            await self.unsubscribe_all()

            # STEP 2: Stop the dispatch loop
            await self._stop_local()

        except Exception as e:
            logger.error(f"LocalMessageBus: Error during shutdown: {e}")
        finally:
            self.is_started = False
            logger.info("LocalMessageBus: Stopped.")

    async def _stop_local(self):
        """Stop local transport."""
        try:
            loop = asyncio.get_running_loop()
            loop.call_soon_threadsafe(self.dispatch_queue.put_nowait, "__STOP__")
        except RuntimeError:
            self.dispatch_queue.put_nowait("__STOP__")

        if self.amb_task:
            self.amb_task.cancel()
            try:
                await self.amb_task
            except asyncio.CancelledError:
                pass

    async def subscribe(self, subject: str, subscriber: Dict[str, Callable]):
        """
        Subscribe to a subject with named callbacks.

        Args:
            subject: Subject pattern to subscribe to (supports wildcards)
            subscriber: Dictionary of {name: callback} pairs
        """
        # Use base class validation
        self._validate_subscription_params(subject, subscriber)

        try:
            for name, callback in subscriber.items():
                # Unsubscribe existing subscriber with same name
                if name in self.subscribers[subject]:
                    await self.unsubscribe(subject=subject, subscriber_name=name)

                # For local transport, just store the callback
                self.subscribers[subject][name] = callback
                self._log_subscriber_info("Subscribed", subject, name)

        except Exception as e:
            logger.exception(f"LocalMessageBus: Error during subscribe: {e}")
            raise

    async def unsubscribe(self, subject: str, subscriber_name: str):
        """
        Unsubscribe a named subscriber from a subject.

        Args:
            subject: Subject pattern to unsubscribe from
            subscriber_name: Name of the subscriber to remove
        """
        # Cancel any running tasks for this subscriber
        if self.subscriber_tasks.get(subject, {}).get(subscriber_name):
            for task in self.subscriber_tasks[subject].get(subscriber_name, []):
                task.cancel()
            del self.subscriber_tasks[subject][subscriber_name]
            if not self.subscriber_tasks[subject]:
                del self.subscriber_tasks[subject]

        # Use base class cleanup method
        self._cleanup_subscriber_references(subject, subscriber_name)
        self._log_subscriber_info("Unsubscribed", subject, subscriber_name)

    async def unsubscribe_all(self):
        """Remove all subscribers with error handling."""
        if not self.subscribers:
            logger.debug("LocalMessageBus: No subscribers to unsubscribe.")
            return

        subjects = list(self.subscribers.keys())
        for subject in subjects:
            names = list(self.subscribers[subject].keys())
            for name in names:
                try:
                    await self.unsubscribe(subject=subject, subscriber_name=name)
                except Exception as e:
                    # Log but continue with other unsubscriptions
                    logger.warning(
                        f"LocalMessageBus: Failed to unsubscribe {name} from {subject}: {e}"
                    )

        # Clear any remaining subscribers
        self.subscribers.clear()
        logger.debug("LocalMessageBus: All subscribers removed.")

    async def publish(self, message: Union[dict, T]):
        """
        Publish a message to the bus.

        Args:
            message: Message to publish (dict or typed message)
        """
        try:
            # Use base class message validation
            message = self._validate_message(message)

            if not self.is_started:
                raise RuntimeError(
                    "LocalMessageBus: Bus not started. Call `await bus.start()` before publishing."
                )

            # Check broadcast permission
            if not self.broadcast_allowed and message.header.recipient == "*":
                raise ValueError("LocalMessageBus: Broadcast is disallowed.")

            # Enqueue for dispatch
            await self.dispatch_queue.put(message)

        except Exception as e:
            logger.exception(f"LocalMessageBus: Error during publish: {e}")
            raise

    async def _deliver(self, message: Union[dict, T]):
        """Deliver message to subscribers."""
        # Use base class message validation
        message = self._validate_message(message)

        logger.debug(
            f"LocalMessageBus._deliver: message.body.type='{message.body.type}', subscribers={list(self.subscribers.keys())}"
        )

        if message.body.type is None:
            raise ValueError("LocalMessageBus: Message missing 'type'.")

        async with self.lock:
            for pattern, callbacks in self.subscribers.items():
                # Use base class pattern matching
                if self._matches(pattern=pattern, subject=message.body.type):
                    for name, cb in callbacks.items():
                        logger.debug(
                            f"LocalMessageBus._deliver: dispatching to name={name} with subject='{pattern}' for message.body.type='{message.body.type}'"
                        )

                        if self.subscriber_tasks[message.body.type].get(name) is None:
                            self.subscriber_tasks[message.body.type][name] = []

                        # Use base class safe call
                        self.subscriber_tasks[message.body.type][name].append(
                            await self._safe_call(cb, message)
                        )

    async def _dispatch_loop(self, timeout: Optional[float] = None):
        """Main dispatch loop for local transport."""
        try:
            self.ready_event.set()
            while True:
                try:
                    if timeout is not None:
                        message = await asyncio.wait_for(
                            self.dispatch_queue.get(), timeout=timeout
                        )
                    else:
                        message = await self.dispatch_queue.get()
                except asyncio.TimeoutError:
                    logger.info("LocalMessageBus: Dispatch loop timeout reached.")
                    self.is_started = False
                    break

                if message == "__STOP__":
                    logger.info(
                        "LocalMessageBus: Dispatch loop received __STOP__ message."
                    )
                    self.is_started = False
                    break

                await self._deliver(message)
                logger.debug("LocalMessageBus: Dispatch loop: message dispatched.")

        except Exception as e:
            import traceback

            logger.error(f"LocalMessageBus: Dispatch loop crashed. error={e}")
            traceback.print_exc()

    def set_broadcast_allowed(self, allowed: bool):
        """
        Enable or disable broadcast messages.

        Args:
            allowed: Whether to allow broadcast messages (recipient="*")
        """
        self.broadcast_allowed = allowed
