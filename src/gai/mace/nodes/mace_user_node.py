import asyncio
from gai.lib.utils import get_app_path
from gai.lib.constants import DEFAULT_GUID

from gai.dialogue import DialogueBus
from gai.lib.logging import getLogger
from gai.messages.typing import MessagePydantic, ProfileBodyPydantic
from gai.mace.operations.rollcall import RollcallSender
from gai.mace.operations.handshake import HandshakeSender
from gai.mace.operations.chat import ChatSender
logger = getLogger(__name__)

class MaceUserNode:

    def __init__(self, dialogue: DialogueBus):
        self.node_name = "User"
        self.rsvp: list[str] = []
        self.dialogue = dialogue
        self.flow = ""
        self.a_queue = asyncio.Queue()
        self.participants = []

    async def register(self):
        # Register rollcall
        self.rollcall_sender = RollcallSender(
            dialogue=self.dialogue,
            node_name=self.node_name
        )
        await self.rollcall_sender.subscribe(received_profile_callback=self.received_profile_handler)


        # Register handshake
        self.handshake_sender = HandshakeSender(
            dialogue=self.dialogue,
            node_name=self.node_name
        )
        await self.handshake_sender.subscribe(received_handshake_ack_callback=self.received_handshake_ack_handler)
        
        # Register chat
        self.chat_sender = ChatSender(
            dialogue=self.dialogue,
            node_name=self.node_name,
        )
        await self.chat_sender.subscribe(output_chunks_callback=self.output_chunks_handler)
        
        # # Register system
        # self.system_sender = SystemSender(
        #     dialogue=self.dialogue,
        #     node_name=self.node_name
        # )

    async def received_profile_handler(self, pydantic:MessagePydantic):
        try:
            logger.info(f"MaceUserNode.received_profile_handler: header={pydantic.header}")
            logger.debug(f"MaceUserNode.received_profile_handler: pydantic={pydantic}")
            
            # Add participant profile for this round of conversation
            
            profile = ProfileBodyPydantic(**pydantic.body.model_dump())
            self.participants.append(profile)
            
            # Add sender to RSVP list for the handshake
            
            self.rsvp.append(pydantic.header.sender)
        except Exception as e:
            logger.error(f"MaceUserNode.received_profile_handler: Error in received_profile_handler: {e}")

    async def received_handshake_ack_handler(self, pydantic:MessagePydantic):
        logger.info(f"MaceUserNode.received_handshake_ack_handler: header={pydantic.header}")

    async def output_chunks_handler(self, pydantic: MessagePydantic):
        logger.info(f"MaceUserNode.output_chunks_handler: header={pydantic.header}")
        try:
            
            # Put the chunk in the queue for streaming
            self.a_queue.put_nowait(pydantic)

        except Exception as e:
            logger.error(f"MaceUserNode.output_chunks_handler: Error in output_chunks_handler: {e}")
        
    async def rollcall(self):
        self.participants = []
        self.rsvp = []
        await self.rollcall_sender.rollcall()
        return self.participants

    # Returns a stream of ChatReplyMessagePydantic

    async def chat(self,
                   flow: str,
                   round_no: int,
                   user_message: str,
                   dialogue_id:str = DEFAULT_GUID,
                   ):

        # The flow determines the dialgoue pattern

        self.flow = flow

        # Run rollcall to gather participants for this round.
        
        await self.rollcall_sender.rollcall()
        
        # Create the orchestration plan for participants        
        
        plan=HandshakeSender.create_plan(
            dialogue_id=dialogue_id,
            round_no=round_no,
            flow=self.flow
        )
        
        # Handshake with participants
        
        await self.handshake_sender.handshake(
            plan=plan
        )
        
        # Begin chatting
        await self.chat_sender.chat_send(
            user_message=user_message,
            plan=plan
        )
       
        # Stream chunks from the message queue
        
        chunk = await self.a_queue.get()
        while chunk.body.chunk != "<eom>":
            yield chunk
            chunk = await self.a_queue.get()
            
    async def next(self):
        await self.chat_sender.next()
        
        # Stream chunks from the message queue
        
        chunk = await self.a_queue.get()
        while chunk.body.chunk != "<eom>":
            yield chunk
            chunk = await self.a_queue.get()

