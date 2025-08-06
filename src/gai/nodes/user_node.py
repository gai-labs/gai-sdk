import asyncio
from typing import Callable, Optional
from gai.sessions import SessionManager
from gai.sessions.operations.chat import ChatSender
from gai.sessions.operations.handshake import HandshakeSender
from gai.messages.typing import MessagePydantic, OrchPlanPydantic


class UserNode:
    """
    A simplified wrapper around ChatSender that handles user interactions
    in multi-agent sessions with built-in response streaming.
    """

    def __init__(
        self,
        node_name: str = "User",
        session_mgr: SessionManager = None,
        timeout: int = 2,
    ):
        self.node_name = node_name
        self.session_mgr = session_mgr
        self.timeout = timeout
        self.sender = ChatSender(
            node_name=node_name, session_mgr=session_mgr, timeout=timeout
        )
        self.response_queue = asyncio.Queue()
        self.plan = None

    async def subscribe(self):
        """
        Subscribe to chat responses from agents.
        """
        await self.sender.subscribe(output_chunks_callback=self._response_handler)

    async def _response_handler(self, pydantic: MessagePydantic):
        """
        Handle incoming responses from agents.
        """
        self.response_queue.put_nowait(pydantic)

    async def start_conversation(self, user_message: str, flow_plan: str):
        """
        Start a conversation with the given message and plan.

        Args:
            user_message: The initial message to send
            flow_plan: Plan description in format "User ->> Agent1\\nAgent1 ->> Agent2"
        """
        # Create the plan
        self.plan = HandshakeSender.create_plan(flow_plan)

        # Send the initial message
        step = await self.sender.chat_send(user_message=user_message, plan=self.plan)
        return step

    async def send_message(self, user_message: str, plan: OrchPlanPydantic):
        """
        Send a message with an existing plan.
        """
        self.plan = plan
        step = await self.sender.chat_send(user_message=user_message, plan=plan)
        return step

    async def continue_conversation(self):
        """
        Continue the conversation to the next step in the plan.
        """
        if not self.plan:
            raise ValueError(
                "No active plan. Call start_conversation or send_message first."
            )

        return await self.sender.next()

    async def stream_response(
        self, display_callback: Optional[Callable[[str, str], None]] = None
    ):
        """
        Stream and return the next agent response.

        Args:
            display_callback: Optional callback function(sender_name, chunk_text) for real-time display

        Returns:
            bool: True if more responses are expected, False if conversation is complete
        """
        try:
            chunk = await asyncio.wait_for(self.response_queue.get(), timeout=10.0)
        except asyncio.TimeoutError:
            raise RuntimeError(
                f"Timeout waiting for initial response from agent (10s timeout)"
            )
        sender_name = chunk.header.sender
        full_content = ""

        # Process streaming chunks with timeout protection
        max_chunks = 500  # Reduced limit to prevent infinite loops
        chunk_count = 0
        name_prefix_added = False
        total_timeout = 30.0  # Maximum total time for response
        start_time = asyncio.get_event_loop().time()

        while chunk.body.chunk != "<eom>" and chunk_count < max_chunks:
            # Check total elapsed time
            elapsed = asyncio.get_event_loop().time() - start_time
            if elapsed > total_timeout:
                print(f"Warning: Total response timeout ({total_timeout}s) exceeded for {sender_name}")
                break

            if chunk.body.chunk_no == 0 and display_callback and not name_prefix_added:
                display_callback(sender_name, f"\n{sender_name}: ")
                name_prefix_added = True

            if isinstance(chunk.body.chunk, str):
                full_content += chunk.body.chunk
                if display_callback:
                    display_callback("", chunk.body.chunk)  # Don't pass sender_name for chunks

            try:
                chunk = await asyncio.wait_for(self.response_queue.get(), timeout=3.0)  # Reduced timeout
                chunk_count += 1
            except asyncio.TimeoutError:
                print(
                    f"Warning: Timeout waiting for next chunk from {sender_name}, assuming end of message"
                )
                break

        if chunk_count >= max_chunks:
            print(
                f"Warning: Maximum chunk count ({max_chunks}) reached for {sender_name}, assuming end of message"
            )

        if display_callback:
            display_callback(sender_name, "\n")  # New line after response

        # Check if more steps remain
        has_more = len(self.sender.plan.steps) - 1 > self.sender.plan.curr_step_no
        return has_more, sender_name, full_content

    async def run_full_conversation(
        self,
        user_message: str,
        flow_plan: str,
        display_callback: Optional[Callable[[str, str], None]] = None,
    ):
        """
        Run a complete multi-agent conversation from start to finish.

        Args:
            user_message: The initial message to send
            flow_plan: Plan description in format "User ->> Agent1\\nAgent1 ->> Agent2"
            display_callback: Optional callback function(sender_name, chunk_text) for real-time display

        Returns:
            list: List of (sender_name, content) tuples for each response
        """
        # Start the conversation
        await self.start_conversation(user_message, flow_plan)

        if display_callback:
            display_callback("User", f"User: {user_message}\n")

        responses = []

        # Stream all responses
        has_more, sender, content = await self.stream_response(display_callback)
        responses.append((sender, content))

        while has_more:
            await self.continue_conversation()
            has_more, sender, content = await self.stream_response(display_callback)
            responses.append((sender, content))

        return responses

    async def unsubscribe(self):
        """
        Unsubscribe from the session.
        """
        await self.sender.unsubscribe()
