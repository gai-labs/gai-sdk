import asyncio
import time
import uuid
from typing import Any, Awaitable, Callable, Literal, Optional, ClassVar
from gai.messages.typing import MessageHeaderPydantic
from pydantic import BaseModel, model_validator
from gai.mace.orchestrator import OrchPlanPydantic
from gai.lib.logging import getLogger
from gai.messages.typing import ChatReplyBodyPydantic, ChatSendBodyPydantic, MessagePydantic
logger = getLogger(__name__)


# --- Chat Messages ---

def create_message_id(type:str, sender:str, recipient:str, dialogue_id:str, round_no:int, step_no:int) -> str:
    return f"{type}.{sender}.{recipient}.{dialogue_id}.{round_no}.{step_no}"
       
class ChatSender:

    def __init__(self, 
                 node_name, 
                 dialogue, 
                 timeout=2, 
                 ):
        self.node_name = node_name
        self.dialogue = dialogue
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

        recap = self.dialogue.extract_recap()
        if recap:
            user_message = f"""
            {self.node_name}, this is a short recap of your conversation so far <recap>{recap}</recap>.
            Refer to this recap to understand the background of the conversation. You will continue from where you left off as {self.node_name}.
            {user_message}
        """
        pydantic = MessagePydantic(
            header=MessageHeaderPydantic(
                sender=step.sender,
                recipient=step.recipient,
                timestamp=time.time()
            ),
            body=ChatSendBodyPydantic(
                dialogue_id= self.plan.dialogue_id,
                round_no= self.plan.round_no,
                turn_no= step.turn_no,
                step_no= step.step_no,
                content= user_message,
                message_id=create_message_id(
                    type="send",
                    sender=step.sender,
                    recipient=step.recipient,
                    dialogue_id=self.plan.dialogue_id,
                    round_no=self.plan.round_no,
                    step_no=step.step_no
                )
            )            
        )

        if self.plan.flow_type not in ["poll", "chain"]:
            raise ValueError(f"ChatSender.chat_send_handler: Invalid flow type: {self.plan.flow_type}")

        await self.dialogue.publish(pydantic=pydantic)

        self.inc_step()

        logger.info(f"ChatSender.chat_send_handler: step: #{self.plan.curr_step_no} - User message sent.")
        
        # Introduce a 2 second delay to avoid overloading the network
        
        await asyncio.sleep(self.timeout)

        return step


    async def next(self):
        self.inc_step()
        user_message = self.user_message
        if self.plan.flow_type == "chain":
            user_message = "It is your turn to continue."
        return await self.chat_send(user_message=user_message, plan=self.plan)

    async def subscribe(self, output_chunks_callback: Callable[[MessagePydantic],Awaitable[Any]]):
        
        async def _output_chunks_handler(message: MessagePydantic):
            await output_chunks_callback(message)
        
        await self.dialogue.subscribe(subject="chat.reply", callback={self.node_name:_output_chunks_handler})
        

###---------------------------------------------------------------------------------

class ChatResponder:

    def __init__(self, node_name, dialogue):
        self.node_name = node_name
        self.dialogue = dialogue
        self.plans = {}

    def inc_step(self, dialogue_id):
        self.plans[dialogue_id].curr_step_no += 1

    async def _send_chunks(self, streamer, pydantic: MessagePydantic) -> MessagePydantic:
        chunk_no = 0
        combined_chunks = ""
        curr_step_no = self.plans[pydantic.body.dialogue_id].curr_step_no
        template = MessagePydantic(
            header=MessageHeaderPydantic(
                sender= self.node_name,
                recipient= pydantic.header.sender,
                timestamp= time.time()
            ),
            body=ChatReplyBodyPydantic(
                dialogue_id= pydantic.body.dialogue_id,
                round_no= pydantic.body.round_no,
                turn_no= pydantic.body.turn_no,
                step_no= curr_step_no,
                chunk_no= 0,
                chunk= "",
                content= "",
                message_id=create_message_id(
                    type="reply",
                    sender=pydantic.header.sender,
                    recipient=pydantic.header.recipient,
                    dialogue_id=pydantic.body.dialogue_id,
                    round_no=pydantic.body.round_no,
                    step_no=curr_step_no
                )
            )
        )

        plan = self.plans[pydantic.body.dialogue_id]
        if plan.flow_type not in ["poll", "chain"]:
            raise ValueError(f"ChatResponder._send_chunks: Invalid flow type: {plan.flow_type}")

        async for chunk in streamer:
            combined_chunks += chunk
            reply = template.model_copy()
            reply.body.chunk_no = chunk_no
            reply.body.chunk = chunk
            reply.body.content = combined_chunks
            #print(reply.body.chunk, end="", flush=True)
            await self.dialogue.publish(pydantic=reply)
            chunk_no += 1

        final_reply = template.model_copy()
        final_reply.body.chunk_no = chunk_no
        final_reply.body.chunk = "<eom>"
        final_reply.body.content = combined_chunks
        await self.dialogue.publish(pydantic=final_reply)

        return final_reply


    async def subscribe(self, 
                        input_chunks_callback: Callable[[MessagePydantic],Awaitable[Any]],
                        completed_content_callback: Callable[[MessagePydantic],Awaitable[Any]]
                        ):
        
        # ChatResponder is used by Agents so it only listen to send messages from user and reply from other agents as well.
        
        async def _chatsend_handler(message: MessagePydantic):
            try:
                
                pydantic = message.copy()
                plan = self.plans[pydantic.body.dialogue_id]

                if pydantic.header.sender == self.node_name:
                    # Ignore messages sent by self
                    return None

                logger.debug(f'ChatResponder({self.node_name})._chatsend_handler: chat.send received. curr_step_no={plan.curr_step_no}')

                if pydantic.body.step_no != plan.curr_step_no:
                    raise ValueError(f"ChatResponder({self.node_name})._chatsend_handler: Step number mismatch: message_step_no={pydantic.body.step_no} curr_step_no={plan.curr_step_no}")
                self.inc_step(pydantic.body.dialogue_id)


                # ⚠️ TODO: This may not work because we are rejecting ALL messages from others.

                if pydantic.header.recipient != self.node_name:

                    # if plan.steps[plan.curr_step_no].is_pm and self.input_chunks_callback:
                        
                    #     # NOTE: This is where to load chunks from external source

                    #     pydantic = await self.input_chunks_callback(pydantic=pydantic)
                    
                    # logger.debug(f'ChatResponder({self.node_name}).chatsend_handler: someone received chat.send. curr_step_no={plan.curr_step_no}')
                    return None

                    
                # NOTE: This is where to load chunks from external source
                
                streamer = await input_chunks_callback(pydantic=pydantic)
                
                # if not hasattr(streamer, "__iter__"):
                #     raise ValueError("processor_cb must return a generator")

                last_chunk = await self._send_chunks(streamer, pydantic)

                await completed_content_callback(last_chunk)

                # ✅ Step increment only once after sending full response
                
                self.inc_step(pydantic.body.dialogue_id)
                
                logger.info(f'\nChatResponder._chatsend_handler: step= #{plan.curr_step_no} - Text generation completed.')            
                    
            except Exception as e:
                logger.error(f"ChatResponder({self.node_name})._chatsend_handler: error={e}")
                raise
                    
        await self.dialogue.subscribe(subject="chat.send", callback={self.node_name:_chatsend_handler})
        
        async def _chatreply_handler(message: MessagePydantic):
            try:
                logger.debug(f'ChatResponder({self.node_name})._chatreply_handler: chat.reply received')
                
                pydantic = message.model_copy()

                # Guard: if self.plan is not set, log an error and return early.

                plan = self.plans.get(pydantic.body.dialogue_id)
                if plan is None:
                    logger.error("ChatResponder({self.node_name})._chatreply_handler: Received chat reply but self.plan is None; ignoring message.")
                    return

                # Ignore reply that comes from self

                if pydantic.header.sender == self.node_name:
                    return


                # ⚠️ TODO: Verify if self.input_chunks_callback is needed here
                
                # if self.input_chunks_callback:
                    
                #     # Note: This is where we load the chunks from external source
                    
                #     pydantic = await self.input_chunks_callback(pydantic=pydantic)

                is_private_message = plan.steps[plan.curr_step_no].is_pm

                if pydantic.body.chunk == "<eom>":
                    
                    if pydantic.body.step_no != self.plans[pydantic.body.dialogue_id].curr_step_no:
                        raise ValueError(f'ChatResponder({self.node_name})._chatreply_handler: Step number mismatch. message_step_no={pydantic.body.step_no} curr_step_no ={self.plans[pydantic.body.dialogue_id].curr_step_no}')

                    # ✅ Only increment here
                    self.inc_step(pydantic.body.dialogue_id)

                    logger.debug(f"ChatResponder({self.node_name})._chatreply_handler: step: #{plan.curr_step_no} - text stream received.")

                if is_private_message and pydantic.body.chunk == "<eom>":
                    
                    # Note: This is where we output the completed content to external sink
                        
                    await completed_content_callback(pydantic)

                return pydantic
            except Exception as e:
                logger.error(f"ChatResponder({self.node_name})._chatreply_handler: error={e}")
                raise        
        
        await self.dialogue.subscribe(subject="chat.reply", callback={self.node_name:_chatreply_handler})
