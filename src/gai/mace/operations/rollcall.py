import uuid
import time
import asyncio

from gai.lib.logging import getLogger
logger = getLogger(__name__)

from typing import Any, Awaitable, Callable

# --- Rollcall Messages ---
from gai.messages.typing import MessagePydantic, MessageHeaderPydantic, RollcallBodyPydantic


# ###---------------------------------------------------------------------------------
        
class RollcallSender:

    def __init__(self, node_name, dialogue, timeout=2):
        self.node_name = node_name
        self.dialogue = dialogue
        self.timeout = timeout

    async def subscribe(self, received_profile_callback:Callable[[MessagePydantic],Awaitable[Any]]):

        async def _profile_handler(pydantic: MessagePydantic):
            try:
                
                # Received profile from a participant
                
                logger.info(f"RollcallSender({self.node_name}).profile_handler: received {pydantic.body.type} from {pydantic.header.sender}.")
                await received_profile_callback(pydantic)
            except Exception as e:
                logger.error(f"RollcallSender({self.node_name}).profile_handler: Error in profile_handler: {e}")
                raise
            
        await self.dialogue.subscribe(subject="system.profile",callback={self.node_name:_profile_handler})
        
    async def rollcall(self):
        logger.info(f"RollcallSender({self.node_name}).rollcall: start rollcall")
        pydantic = MessagePydantic(
            header=MessageHeaderPydantic(
                sender=self.node_name,
                recipient="*",
                timestamp=time.time()
            ),
            body = RollcallBodyPydantic()
        )
        
        # Broadcast rollcall message to all participants
        
        await self.dialogue.publish(pydantic = pydantic)
        
        # Wait for a response from each participant
        
        await asyncio.sleep(self.timeout)


###---------------------------------------------------------------------------------

class RollcallResponder:

    def __init__(self, 
                 node_name: str, 
                 dialogue):
        self.node_name = node_name
        self.dialogue = dialogue
    
    async def subscribe(self, create_profile_callback:Callable[[MessagePydantic],Awaitable[MessagePydantic]]):

        async def _rollcall_handler(pydantic: MessagePydantic):
            try:
                if pydantic.header.recipient not in ["*", self.node_name]:
                    return  # Not for me

                logger.info(f"RollcallResponder({self.node_name}).rollcall_handler: received {pydantic.body.type} from {pydantic.header.sender}.")
                profile_msg = await create_profile_callback(pydantic=pydantic)
                await self.dialogue.publish(pydantic=profile_msg)
                logger.info(f"RollcallResponder({self.node_name}).rollcall_handler: sent {profile_msg.body.type} to {profile_msg.header.recipient}.")
            except Exception as e:
                logger.error(f"agent_node.rollcall_handler: Error in rollcall_handler: {e}")
                raise
            
        await self.dialogue.subscribe(subject="system.rollcall", callback={self.node_name:_rollcall_handler})
        
        
