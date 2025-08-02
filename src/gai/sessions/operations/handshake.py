import asyncio
import time
from gai.lib.constants import DEFAULT_GUID
from gai.lib.logging import getLogger
from typing import Any, Awaitable, Callable
from gai.sessions import SessionManager
from gai.sessions.operations.orchestrator import parse_flow
from gai.messages.typing import MessageHeaderPydantic, MessagePydantic
from gai.messages.typing import (
    OrchPlanPydantic,
    HandshakeBodyPydantic,
    HandshakeAckBodyPydantic,
)

logger = getLogger(__name__)

# --- Operations ---


class HandshakeSender:
    def __init__(self, node_name: str, session_mgr: SessionManager, timeout=2):
        self.node_name = node_name
        self.session_mgr = session_mgr
        self.timeout = timeout
        self.participants = {}

    @staticmethod
    def create_plan(
        flow: str,
        dialogue_id: str = DEFAULT_GUID,
        round_no: int = 0,
    ) -> OrchPlanPydantic:
        logger.info(f"user_node.handshake: start handshake")
        plan = parse_flow(dialogue_id, round_no, flow)
        return plan

    async def handshake(self, plan: OrchPlanPydantic):
        logger.info("user_node.handshake: sending handshake messages")
        for participant in plan.participants:
            self.participants[participant] = False

            pydantic = MessagePydantic(
                header=MessageHeaderPydantic(
                    sender=self.node_name, recipient=participant, timestamp=time.time()
                ),
                body=HandshakeBodyPydantic(body=plan.model_dump()),
            )

            # Send orchestration plan to each participant

            await self.session_mgr.publish(pydantic=pydantic)

        # Wait for acks from all participants

        await asyncio.sleep(self.timeout)

    async def subscribe(
        self,
        received_handshake_ack_callback: Callable[[MessagePydantic], Awaitable[Any]],
    ):
        async def _received_ack_handler(pydantic: MessagePydantic):
            try:
                # Received ack from a participant

                logger.info(
                    f"HandshakeSender({self.node_name}).profile_handler: received {pydantic.body.type} from {pydantic.header.sender}."
                )
                self.participants[pydantic.header.sender] = True
                await received_handshake_ack_callback(pydantic)
            except Exception as e:
                logger.error(
                    f"HandshakeSender({self.node_name}).received_ack_handler: Error in received_ack_handler: {e}"
                )
                raise

        await self.session_mgr.subscribe(
            subject="system.handshake_ack",
            callback={self.node_name: _received_ack_handler},
        )

    async def unsubscribe(self):
        """Unsubscribe from the handshake ack messages."""
        await self.session_mgr.unsubscribe(
            subject="system.handshake_ack", subscriber_name=self.node_name
        )
        logger.info(
            f"HandshakeSender({self.node_name}).unsubscribe: Unsubscribed from handshake ack messages."
        )


###---------------------------------------------------------------------------------


class HandshakeResponder:
    def __init__(self, node_name: str, session_mgr: SessionManager):
        self.node_name = node_name
        self.session_mgr = session_mgr
        self.orchestration_plan = None

    async def subscribe(
        self, get_plan_callback: Callable[[MessagePydantic], Awaitable[MessagePydantic]]
    ):
        async def _handshake_handler(pydantic: MessagePydantic):
            try:
                if pydantic.header.recipient != self.node_name:
                    return  # Not intended for me

                logger.info(
                    f"HandshakeResponder({self.node_name}).handshake_handler: received {pydantic.body.type} from {pydantic.header.sender}."
                )
                await get_plan_callback(pydantic=pydantic)

                # Create Ack
                if not pydantic:
                    raise Exception(
                        "HandshakeResponder.handshake_handler: message cannot be None."
                    )

                self.orchestration_plan = pydantic.body

                pydantic = MessagePydantic(
                    header=MessageHeaderPydantic(
                        sender=self.node_name,
                        recipient=pydantic.header.sender,
                        timestamp=time.time(),
                    ),
                    body=HandshakeAckBodyPydantic(body=self.orchestration_plan.body),
                )

                # Send Ack

                await self.session_mgr.publish(pydantic=pydantic)
                logger.info(
                    f"HandshakeResponder({self.node_name}).handshake_handler: sent {pydantic.body.type} to {pydantic.header.recipient}."
                )
                return pydantic

            except Exception as e:
                logger.error(e)
                print(e)
                raise

        await self.session_mgr.subscribe(
            subject="system.handshake", callback={self.node_name: _handshake_handler}
        )

    async def unsubscribe(self):
        """Unsubscribe from the handshake messages."""
        await self.session_mgr.unsubscribe(
            subject="system.handshake", subscriber_name=self.node_name
        )
        logger.info(
            f"HandshakeResponder({self.node_name}).unsubscribe: Unsubscribed from handshake messages."
        )
