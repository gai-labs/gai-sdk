import asyncio
from typing import Callable, Optional
from gai.sessions import SessionManager
from gai.sessions.operations.chat import ChatSender
from gai.sessions.operations.handshake import HandshakeSender
from gai.messages.typing import MessagePydantic, OrchPlanPydantic
from gai.sessions.operations.handshake import HandshakeSender


class UserNode:
    def __init__(self, session_mgr: SessionManager):
        self.session_mgr = session_mgr
        self.a_queue = asyncio.Queue()
        self.chat_sender = ChatSender(node_name="User", session_mgr=session_mgr)

    async def output_chunks_handler(self, pydantic: MessagePydantic):
        """Callback to handle output chunks from the agent."""
        self.a_queue.put_nowait(pydantic)

    async def start(self, user_message, flow_plan):
        await self.chat_sender.subscribe(
            output_chunks_callback=self.output_chunks_handler
        )
        self.flow_plan = HandshakeSender.create_plan(flow_plan)

        """Send a message to the agent."""
        await self.chat_sender.chat_send(user_message=user_message, plan=self.flow_plan)

        async def streamer():
            """Stream the response from the agent."""
            chunk = await self.a_queue.get()
            sender = chunk.header.sender
            while chunk.body.chunk != "<eom>":
                if sender:
                    yield sender
                    sender = None
                if isinstance(chunk.body.chunk, str):
                    yield chunk.body.chunk
                chunk = await self.a_queue.get()

        return streamer()

    async def resume(self):
        """Resume the UserNode."""
        await self.chat_sender.next()

        async def streamer():
            """Stream the response from the agent."""
            chunk = await self.a_queue.get()
            sender = chunk.header.sender
            while chunk.body.chunk != "<eom>":
                if sender:
                    yield sender
                    sender = None
                if isinstance(chunk.body.chunk, str):
                    yield chunk.body.chunk
                chunk = await self.a_queue.get()

        return streamer()
