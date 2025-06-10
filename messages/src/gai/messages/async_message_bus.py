import inspect
import asyncio
from collections import defaultdict
from typing import Callable, Dict, Optional, TypeAlias, Union, Protocol
from .typing import MessagePydantic
from gai.lib.logging import getLogger

logger = getLogger(__name__)
MessageInput: TypeAlias = Union[dict, MessagePydantic]

class MessageBusProtocol(Protocol):
    async def start(self) -> None: ...
    async def stop(self) -> None: ...
    async def subscribe(self, subject: str, callback: Dict[str, Callable]) -> None: ...
    async def publish(self, message: MessageInput) -> None: ...
    async def unsubscribe(self, subject: str, subscriber_name: str) -> None: ...
    async def unsubscribe_all(self) -> None: ...

class AsyncMessageBus(MessageBusProtocol):

    def __init__(self, *, queue_maxsize: int = 100):

        """
        self.subscribers is a dictionary of dictionaries denoting a subject-to-subscriber mapping to a callback.
        Each **subject-subscriber pair** points to exactly one callback.
        Usage:

        await bus.subscribe(subject="send", subscriber={"Sara": handle_send})
        await bus.subscribe(subject="reply", subscriber={"Sara": handle_reply})
        await bus.subscribe(subject="send", subscriber={"Diana": handle_send})
        await bus.subscribe(subject="reply", subscriber={"Diana": handle_reply})
        
        A callback such as self.subscribers["send"]["Sara"] will be called by a subscriber called "Sara" when a message with subject "send" is published to the bus.

        Examples:

        1. Valid:
            - subject = "send" and name = "Sara" maps to handle_send()
            - subject = "reply" and name = "Sara" maps to handle_reply()

        2. Valid:
            - subject = "send" and name = "Sara" maps to handle_send()
            - subject = "send" and name = "Diana" maps to another instance of handle_send()

        3. Invalid:
            - subject = "send" and name = "Sara" maps to handle_send()
            - subject = "send" and name = "Sara" again will result in the earlier handle_send() being unsubscribed and replaced by the new one.
        """
        
        self.subscribers: Dict[str, Dict[str, Callable]] = defaultdict(dict)
        
        """
        self.subscriber_tasks stores the corresponding asyncio callback tasks for each subscriber.
        These tasks are created by _deliver() when dispatching messages to subscribers and used for housekeeping.

        Note:
        - The tasks in the list may be pending, running, completed, or cancelled.
        - This structure only tracks that the tasks were started; it does not guarantee their current state.
        """
        
        self.subscriber_tasks: Dict[str, Dict[str,list[asyncio.Task]]] = defaultdict(dict)
        
        # This queue contains callbacks that are waiting to be dispatched.

        self.dispatch_queue  = asyncio.Queue(maxsize=queue_maxsize)
        self.lock = asyncio.Lock()

        # This lock is only used to prevent race conditions when starting and stopping the bus.
        
        self.start_stop_lock = asyncio.Lock()

        # This flag is used to prevent the dispatch loop from running multiple times.        
        
        self.is_started = False
        
        # This flag is used to determine if "*" wildcard broadcast is allowed.
        
        self.broadcast_allowed = True
        
        # This event is used to signal setting is_started once the startup is completed.
        
        self.ready_event = asyncio.Event()
        
    async def start(self,timeout: Optional[float] = None):
        
        async with self.start_stop_lock:
        
            if self.is_started:
                logger.warning("AsyncMessageBus: start() called but bus is already started.")
                return        
            self.amb_task=asyncio.create_task(self._dispatch_loop(timeout=timeout))
            await self.ready_event.wait()
            self.is_started = True            

    async def stop(self):
        """Signal the remote (worker) bus to shut down."""
        logger.debug("AsyncMessageBus: Message bus shutting down.")

        # Stop the dispatch loop

        try:
            loop = asyncio.get_running_loop()
            loop.call_soon_threadsafe(self.dispatch_queue.put_nowait, "__STOP__")
        except RuntimeError:
            self.dispatch_queue.put_nowait("__STOP__")

        # Unsubscribe and cancel all callbacks

        await self.unsubscribe_all()

        # Cancel the dispatch loop task
        
        if self.amb_task:
            self.amb_task.cancel()
            try:
                await self.amb_task
            except asyncio.CancelledError:
                pass        
        
    async def subscribe(self, subject: str, subscriber: Dict[str, Callable]):

        """
        AsyncMessageBus keeps track of subscribed `task`.
        If the subscriber is already subscribed to the subject, it will be unsubscribed first before subscribing again.
        This is to prevent multiple subscriptions to the same subject with the same name.
        If the previous subscriber is unresponsive, eg. variable is reassigned, this will allow the new subscriber to be registered.
        """

        if not isinstance(subscriber, dict):
            raise TypeError("AsyncMessageBus: subscriber must be a dictionary of name to callback.")
        if not isinstance(subject, str):
            raise TypeError("AsyncMessageBus: subject must be a string.")
        if not subject:
            raise ValueError("AsyncMessageBus: subject cannot be empty.")
        if not subscriber:
            raise ValueError("AsyncMessageBus: subscriber cannot be empty.")
        if not self.is_started:
            raise RuntimeError("AsyncMessageBus: Bus not started. Call `await bus.run()` or `asyncio.create_task(bus.run())` before subscribing.")

        try:
            for name, callback in subscriber.items():
                
                # If a subscriber (name) is already subscribed to the subject, unsubscribe it first then subscribe it again.
                
                if name in self.subscribers[subject]:
                    await self.unsubscribe(subject=subject, subscriber_name=name)

                self.subscribers[subject][name] = callback
        except Exception as e:
            logger.exception(f"AsyncMessageBus: Error during subscribe: {e}")
            raise
        
    async def unsubscribe(self, subject: str, subscriber_name: str):
        
        if self.subscriber_tasks.get(subject, {}).get(subscriber_name):
            
            # Cancel all tasks for this subscriber

            for task in self.subscriber_tasks[subject].get(subscriber_name, []):
                task.cancel()
                
            # After cancelling, remove task list
            del self.subscriber_tasks[subject][subscriber_name]
            
            # Clean up empty dict if no more subscribers under subject
            if not self.subscriber_tasks[subject]:
                del self.subscriber_tasks[subject]                                

        if self.subscribers.get(subject, {}).get(subscriber_name):                
                            
            # Remove the subscriber from the list
            del self.subscribers[subject][subscriber_name]
            
            # clean up empty dict if no more subscribers under subject
            if not self.subscribers[subject]:
                del self.subscribers[subject]

        logger.debug(f"AsyncMessageBus: Unsubscribed {subscriber_name} from subject '{subject}'.")        
        
    async def unsubscribe_all(self):
        """Remove all subscriber callbacks."""
        subjects = [subject for subject in self.subscribers.keys()]
        for subject in subjects:
            names = [name for name in self.subscribers[subject].keys()]
            for name in names:
                await self.unsubscribe(subject=subject, subscriber_name=name)
            
        self.subscribers.clear()
        logger.debug("AsyncMessageBus: All subscribers removed.")        

    async def publish(self, message: MessageInput):
        
        try:
            if isinstance(message, dict):
                # Convert dict to MessagePydantic object
                message = MessagePydantic.model_validate(message)

            if self.broadcast_allowed==False and message.header.recipient == "*":
                raise ValueError("AsyncMessageBus: Broadcast is disallowed.")

            if not self.is_started:
                raise RuntimeError("AsyncMessageBus: Bus not started. Call `await bus.run()` or `asyncio.create_task(bus.run())` before publishing messages.")

            # Worker just enqueues into its dispatch loop
            await self.dispatch_queue.put(message)

        except Exception as e:
            logger.exception(f"AsyncMessageBus: Error during publish: {e}")
            raise
        
    def _matches(self, pattern: str, subject: str) -> bool:
        pattern_tokens = pattern.split('.')
        subject_tokens = subject.split('.')

        pi = si = 0
        while pi < len(pattern_tokens):
            pt = pattern_tokens[pi]
            if pt == '>':
                # '>' matches one or more tokens and must be at the end
                return pi == len(pattern_tokens) - 1 and si < len(subject_tokens)
            if si >= len(subject_tokens):
                return False
            if pt != '*' and pt != subject_tokens[si]:
                return False
            pi += 1
            si += 1
        return si == len(subject_tokens)

    async def _safe_call(self, cb:Callable, message: MessagePydantic) -> asyncio.Task:
        """Create and return a task that safely calls the callback"""
        async def _wrapped_call():
            try:
                if inspect.iscoroutinefunction(cb):
                    await cb(message)
                else:
                    await asyncio.to_thread(cb, message)
            except Exception as e:
                logger.exception(f"AsyncMessageBus: Subscriber error= {e}")

        return asyncio.create_task(_wrapped_call())

    async def _deliver(self, message_or_dict: MessageInput):
        """
        At this point, we need to decide who to route the message to for handling.
        """
        
        # We can take in either a MessagePydantic object or a dict but it must be converted to a MessagePydantic object before dispatching.
        if isinstance(message_or_dict,dict):
            message:MessagePydantic = MessagePydantic.model_validate(message_or_dict)
        else:
            message:MessagePydantic = message_or_dict

        # loops through subscriber to find subjects that match the message type
        
        logger.debug(f"_deliver: message.body.type='{message.body.type}', subscribers={list(self.subscribers.keys())}")
        if message.body.type is None:
            raise ValueError("AsyncMessageBus: Message missing 'type'.")
        
        async with self.lock:
            for pattern, callbacks in self.subscribers.items():
                if self._matches(pattern=pattern, subject=message.body.type):
                    for name, cb in callbacks.items():
                        
                        logger.debug(f"_deliver: dispatching to name={name} with subject='{pattern}' for message.body.type='{message.body.type}'")
                        
                        if self.subscriber_tasks[message.body.type].get(name) is None:
                            self.subscriber_tasks[message.body.type][name] = []
                            
                        # Saves the task in the subscriber_tasks dictionary for housekeeping
                        
                        self.subscriber_tasks[message.body.type][name].append(
                            await self._safe_call(cb, message)
                            )

    def is_subscribed(self, subject: str, subscriber_name: str) -> bool:
        """Check if a subscriber is subscribed to a subject."""
        return subscriber_name in self.subscribers.get(subject, {})

    async def _dispatch_loop(self, timeout: Optional[float] = None):
        # This is the main loop that dispatches messages to subscribers via _deliver.
        
        try:
            
            # ✅ At this point, the dispatch loop is live and ready to consume
            self.ready_event.set()
            while True:

                try:
                    
                    # Run the dispatch loop infinitely or until a timeout is reached
                    
                    if timeout is not None:
                        message = await asyncio.wait_for(self.dispatch_queue.get(), timeout=timeout)
                    else:
                        message = await self.dispatch_queue.get()
                        
                except asyncio.TimeoutError:
                    logger.info("AsyncMessageBus: Dispatch loop timeout reached.")
                    self.is_started = False
                    break
                
                if message == "__STOP__":
                    logger.info("AsyncMessageBus: Dispatch loop received __STOP__ message.")
                    self.is_started = False
                    break
                
                await self._deliver(message)
                logger.debug("AsyncMessageBus: Dispatch loop: message dispatched.")

        except Exception as e:
            import traceback
            logger.error(f"AsyncMessageBus: Dispatch loop crashed. error= {e}")
            traceback.print_exc()

