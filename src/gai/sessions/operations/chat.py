import asyncio
import time
from typing import Any, Awaitable, Callable
from gai.sessions.session_manager import SessionManager
from gai.messages.typing import MessageHeaderPydantic
from gai.messages.typing import (
    ChatSendBodyPydantic,
    ChatReplyBodyPydantic,
    MessagePydantic,
)
from gai.messages.typing import OrchPlanPydantic
from gai.lib.logging import getLogger

logger = getLogger(__name__)

# --- Operations ---


def create_message_id(
    type: str,
    sender: str,
    recipient: str,
    dialogue_id: str,
    round_no: int,
    step_no: int,
) -> str:
    return f"{type}.{sender}.{recipient}.{dialogue_id}.{round_no}.{step_no}"


class ChatSender:
    def __init__(
        self,
        node_name: str,
        session_mgr: SessionManager,
        timeout=2,
    ):
        self.node_name = node_name
        self.session_mgr = session_mgr
        self.timeout = timeout
        self.plan = None
        self.user_message = None
        self.content = None

    def inc_step(self):
        self.plan.curr_step_no += 1

    async def chat_send(self, user_message: str, plan: OrchPlanPydantic):
        self.user_message = user_message
        self.plan = plan

        if self.plan.curr_step_no >= len(self.plan.steps):
            return None

        step = self.plan.steps[self.plan.curr_step_no]

        pydantic = MessagePydantic(
            header=MessageHeaderPydantic(
                sender=step.sender, recipient=step.recipient, timestamp=time.time()
            ),
            body=ChatSendBodyPydantic(
                dialogue_id=self.plan.dialogue_id,
                round_no=self.plan.round_no,
                step_no=step.step_no,
                content=user_message,
            ),
        )

        if self.plan.flow_type not in ["poll", "chain"]:
            raise ValueError(
                f"ChatSender.chat_send_handler: Invalid flow type: {self.plan.flow_type}"
            )

        # Log user's outgoing message
        self.session_mgr.log_message(pydantic=pydantic)
        await self.session_mgr.publish(pydantic=pydantic)

        self.inc_step()

        logger.info(
            f"ChatSender.chat_send_handler: step: #{self.plan.curr_step_no} - User message sent."
        )

        # Introduce a 2 second delay to avoid overloading the network

        await asyncio.sleep(self.timeout)

        return step

    async def next(self):
        self.inc_step()
        user_message = self.user_message
        if self.plan.flow_type == "chain":
            user_message = "it is your turn to continue."
        return await self.chat_send(user_message=user_message, plan=self.plan)

    async def subscribe(
        self, output_chunks_callback: Callable[[MessagePydantic], Awaitable[Any]]
    ):
        async def _output_chunks_handler(message: MessagePydantic):
            if message.body.chunk == "<eom>":
                # Log incoming streamed message from agent
                self.session_mgr.log_message(message)

            await output_chunks_callback(message)

        await self.session_mgr.subscribe(
            subject="chat.reply", callback={self.node_name: _output_chunks_handler}
        )

    async def unsubscribe(self):
        """Unsubscribe from the chat.reply messages."""
        await self.session_mgr.unsubscribe(
            subject="chat.reply", subscriber_name=self.node_name
        )
        logger.info(
            f"ChatSender({self.node_name}).unsubscribe: Unsubscribed from chat.reply messages."
        )


###---------------------------------------------------------------------------------


class ChatResponder:
    def __init__(self, node_name: str, session_mgr: SessionManager):
        self.node_name = node_name
        self.session_mgr = session_mgr
        self.plans = {}

    def inc_step(self, dialogue_id):
        self.plans[dialogue_id].curr_step_no += 1

    async def subscribe(
        self,
        input_chunks_callback: Callable[[MessagePydantic], Awaitable[Any]],
        completed_content_callback: Callable[[MessagePydantic], Awaitable[Any]],
    ):
        # ChatResponder is used by Agents so it only listen to send messages from user and reply from other agents as well.

        async def _chatsend_handler(message: MessagePydantic):
            if not self.plans:
                # If no plans are available that means this agent is not part of any chat session, ignore and return early.

                logger.warning(
                    f"ChatResponder({self.node_name})._chatsend_handler: No plans available; ignoring message."
                )
                return None

            try:
                pydantic = message.copy()
                plan = self.plans[pydantic.body.dialogue_id]

                if pydantic.header.sender == self.node_name:
                    # Ignore messages sent by self
                    return None

                logger.debug(
                    f"ChatResponder({self.node_name})._chatsend_handler: chat.send received. curr_step_no={plan.curr_step_no}"
                )

                if pydantic.body.step_no != plan.curr_step_no:
                    raise ValueError(
                        f"ChatResponder({self.node_name})._chatsend_handler: Step number mismatch: message_step_no={pydantic.body.step_no} curr_step_no={plan.curr_step_no}"
                    )

                self.inc_step(pydantic.body.dialogue_id)

                if pydantic.header.recipient != self.node_name:
                    # Reject all messages not meant for this agent.

                    return None

                # NOTE: Get streamer from LLM
                # We need to be careful here because the input_chunks_callback may return a coroutine or an async generator.
                # If it returns a coroutine, we need to await it to get the async generator.
                # If it returns an async generator, we can use it directly.

                # Log incoming message from user
                self.session_mgr.log_message(pydantic)

                res = input_chunks_callback(pydantic=pydantic)
                import inspect

                if inspect.iscoroutine(res):
                    streamer = await res
                elif inspect.isasyncgen(res):
                    streamer = res
                else:
                    raise TypeError(
                        f"ChatResponder({self.node_name})._chatsend_handler: Expected async generator or coroutine"
                    )

                # This will be blocked until the streamer is exhausted.
                last_chunk = await self._send_chunks(streamer, pydantic)

                # Log streamed outgoing message from assistant
                await completed_content_callback(last_chunk)
                self.session_mgr.log_message(last_chunk)

                # ✅ Step increment only once after sending full response

                self.inc_step(pydantic.body.dialogue_id)

                logger.info(
                    f"\nChatResponder({self.node_name})._chatsend_handler: step= #{plan.curr_step_no} - Text generation completed."
                )

            except Exception as e:
                logger.exception(
                    "ChatResponder(%s)._chatsend_handler: ", self.node_name
                )
                raise

        await self.session_mgr.subscribe(
            subject="chat.send", callback={self.node_name: _chatsend_handler}
        )

        async def _chatreply_handler(message: MessagePydantic):
            try:
                logger.debug(
                    f"ChatResponder({self.node_name})._chatreply_handler: chat.reply received"
                )

                pydantic = message.model_copy()

                # If no plans are available that means this agent is not part of any chat session, ignore and return early.

                plan = self.plans.get(pydantic.body.dialogue_id)
                if plan is None:
                    logger.warning(
                        "ChatResponder({self.node_name})._chatreply_handler: Received chat reply but self.plan is None; ignoring message."
                    )
                    return

                # Ignore reply that comes from self

                if pydantic.header.sender == self.node_name:
                    return

                # if self.input_chunks_callback:
                # Disabled: We don't need to process reply from other agents at chunk level.
                #
                #     pydantic = await self.input_chunks_callback(pydantic=pydantic)

                if pydantic.body.chunk == "<eom>":
                    # This is to keep the steps in sync with the plan even if the message is not meant for this agent.

                    if (
                        pydantic.body.step_no
                        != self.plans[pydantic.body.dialogue_id].curr_step_no
                    ):
                        raise ValueError(
                            f"ChatResponder({self.node_name})._chatreply_handler: Step number mismatch. message_step_no={pydantic.body.step_no} curr_step_no ={self.plans[pydantic.body.dialogue_id].curr_step_no}"
                        )

                    if not plan.steps[plan.curr_step_no].is_pm:
                        # Note: Log streamed outgoing message from other agent

                        await completed_content_callback(pydantic)
                        self.session_mgr.log_message(pydantic)

                    # ✅ Only increment here
                    self.inc_step(pydantic.body.dialogue_id)

                    logger.debug(
                        f"ChatResponder({self.node_name})._chatreply_handler: step: #{plan.curr_step_no} - text stream received."
                    )

                return pydantic
            except Exception as e:
                logger.error(
                    f"ChatResponder({self.node_name})._chatreply_handler: error={e}"
                )
                raise

        await self.session_mgr.subscribe(
            subject="chat.reply", callback={self.node_name: _chatreply_handler}
        )

    async def _send_chunks(
        self, streamer, pydantic: MessagePydantic
    ) -> MessagePydantic:
        """
        called by _chatsend_handler read from streamer and publish chunks to session_mgr.
        """

        chunk_no = 0
        combined_chunks = ""
        last_chunk = ""
        curr_step_no = self.plans[pydantic.body.dialogue_id].curr_step_no
        template = MessagePydantic(
            header=MessageHeaderPydantic(
                sender=self.node_name,
                recipient=pydantic.header.sender,
                timestamp=time.time(),
            ),
            body=ChatReplyBodyPydantic(
                dialogue_id=pydantic.body.dialogue_id,
                round_no=pydantic.body.round_no,
                step_no=curr_step_no,
                chunk_no=0,
                chunk="",
                content="",
            ),
        )

        plan = self.plans[pydantic.body.dialogue_id]
        if plan.flow_type not in ["poll", "chain"]:
            raise ValueError(
                f"ChatResponder._send_chunks: Invalid flow type: {plan.flow_type}"
            )

        async for chunk in streamer:
            if isinstance(chunk, str):
                combined_chunks += chunk
                last_chunk = chunk
                reply = template.model_copy()
                reply.body.chunk_no = chunk_no
                reply.body.chunk = chunk
                reply.body.content = combined_chunks
                # print(reply.body.chunk, end="", flush=True)
                await self.session_mgr.publish(pydantic=reply)
                chunk_no += 1

        # Only send final <eom> if the last chunk wasn't already <eom>
        if last_chunk != "<eom>":
            final_reply = template.model_copy()
            final_reply.body.chunk_no = chunk_no
            final_reply.body.chunk = "<eom>"
            final_reply.body.content = combined_chunks
            await self.session_mgr.publish(pydantic=final_reply)
            return final_reply
        else:
            # Return the last reply that was already <eom>
            final_reply = template.model_copy()
            final_reply.body.chunk_no = chunk_no - 1  # Last chunk was already sent
            final_reply.body.chunk = "<eom>"
            final_reply.body.content = combined_chunks
            return final_reply

    async def unsubscribe(self):
        """Unsubscribe from the chat.send and chat.reply messages."""
        await self.session_mgr.unsubscribe(
            subject="chat.send", subscriber_name=self.node_name
        )
        await self.session_mgr.unsubscribe(
            subject="chat.reply", subscriber_name=self.node_name
        )
        logger.info(
            f"ChatResponder({self.node_name}).unsubscribe: Unsubscribed from chat.send and chat.reply messages."
        )
