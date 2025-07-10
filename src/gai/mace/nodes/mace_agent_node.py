import time
from typing import Union
from gai.lib.image_utils import bytes_to_imageurl
from gai.dialogue import DialogueBus
from gai.mace.operations.chat import ChatResponder
from gai.mace.operations.handshake import HandshakeResponder
from gai.mace.operations.rollcall import RollcallResponder
# from gai.mace.operations.handshake import HandshakeMessagePydantic, HandshakeResponder
# from gai.mace.operations.chat import ChatReplyMessagePydantic, ChatResponder, ChatSendMessagePydantic
# from gai.mace.operations.system import SystemResponder
from gai.lib.logging import getLogger
from gai.messages.typing import MessageHeaderPydantic, MessagePydantic, ProfileBodyPydantic
logger = getLogger(__name__)

class MaceAgentNode:

    def __init__(self, 
                 dialogue: DialogueBus, 
                 persona):
        self.dialogue = dialogue
        self.persona = persona
        self.node_name = persona.name

    async def register(self):
        
        # Register rollcall
        self.rollcall = RollcallResponder(
            node_name=self.node_name,
            dialogue=self.dialogue, 
            )
        await self.rollcall.subscribe(create_profile_callback=self.create_profile)
        
        # Register handshake
        self.handshake = HandshakeResponder(
            dialogue = self.dialogue,
            node_name = self.node_name,
            )
        await self.handshake.subscribe(get_plan_callback=self.get_plan_handler)
        
        # Register chat
        self.chat = ChatResponder(
            node_name=self.node_name,
            dialogue=self.dialogue,
            )
        await self.chat.subscribe(
            input_chunks_callback=self.input_chunks_handler, 
            completed_content_callback=self.completed_content_handler
            )

    # Registered to "system.rollcall"
    async def create_profile(self, pydantic: MessagePydantic):
        logger.info(f"MaceAgentNode({self.node_name}).create_profile: header={pydantic.header}")
        image_64x64=None
        image_128x128=None
        if self.persona.portraits:
            image_64x64 = bytes_to_imageurl(self.persona.portraits.image64)["url"]
            image_128x128 = bytes_to_imageurl(self.persona.portraits.image128)["url"]
        output = MessagePydantic(
            header=MessageHeaderPydantic(
                sender=self.persona.name,
                recipient=pydantic.header.sender,
                timestamp=time.time()
            ),
            body=ProfileBodyPydantic(
                name=self.persona.name,
                desc=self.persona.self_introduction,
                skills=self.persona.skills,
                agent_class=self.persona.agent_class,
                image_64x64= image_64x64,
                image_128x128=image_128x128,
            )
        )
        
        return output

    # Registered to "system.handshake"
    async def get_plan_handler(self, pydantic: MessagePydantic):
        logger.info(f"MaceAgentNode({self.node_name}).get_plan_handler: header={pydantic.header}")
        try:
            from gai.mace.orchestrator import OrchPlanPydantic
            plan = OrchPlanPydantic(**pydantic.body.body)
            self.chat.plans[plan.dialogue_id] = plan
            return pydantic
        except Exception as e:
            logger.error(f"MaceAgentNode({self.node_name}).get_plan_handler: Error in get_plan_handler: {e}")
            raise

    # Registered to "chat.send" from user    
    async def input_chunks_handler(self, pydantic: MessagePydantic):
        """
        This function is meant for handling streamed chunks
        """
        
        logger.info(f"MaceAgentNode({self.node_name}).input_chunks_handler: header={pydantic.header}")
        if pydantic.body.type == "chat.reply":
            
            # Should never handle reply messages.
            
            raise ValueError("MaceAgentNode({self.node_name}).input_chunks_handler: input_chunks_callback should not process reply messages.")

        # Time to call chat completion
        async def get_streamer():
            streamer = await self.persona.run_async(pydantic.body.content)
            async for chunk in streamer():
                yield chunk
                
        return get_streamer()
    
    async def completed_content_handler(self, pydantic:MessagePydantic):
        """
        This function is meant for handling completed content.
        From an agent's view, there really isn't much to do here apart from logging the output.
        """
        logger.info(f"MaceAgentNode({self.node_name}).completed_content_handler: header={pydantic.header}")
        logger.debug(f"MaceAgentNode({self.node_name}).completed_content_handler: message={pydantic}")