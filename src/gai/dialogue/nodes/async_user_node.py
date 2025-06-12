from gai.dialogue import DialogueBus
from gai.messages import MessagePydantic,message_helper
from gai.lib.logging import getLogger
logger = getLogger(__name__)


class AsyncUserNode:
    
    # node that will send user message and handle reply messages from agents

    def __init__(self, handle_reply_cb=None):
        self.name = "User"
        self.broadcast_allowed = True
        self.handle_reply_cb = handle_reply_cb
    
    async def subscribe(self, dialogue:DialogueBus):
        self.dialogue = dialogue
        await self.dialogue.subscribe(subject="reply",callback={self.name:self.handle_reply})
        
    def validate_action(self, message:MessagePydantic)->bool:        
        
        # Check if the message is a reply message
        if message.body.type != "reply":
            return False
        
        # Check if the recipient is this responder or a wildcard
        if message.header.recipient != self.name and message.header.recipient != "*":
            logger.debug(f"[{self.name}] Message not for me, ignore. message={message}")
            return False
        elif message.header.recipient == "*":
            if self.broadcast_allowed:
                logger.debug(f"[{self.name}] Handle broadcast message. message={message}")
                return True
            else:
                logger.debug(f"[{self.name}] Ignore broadcast message. message={message}")
                return False
        
        logger.debug(f"[{self.name}] Handle reply message. message={message}")        
        return True    
   
    async def handle_reply(self, message: MessagePydantic):
        if not self.validate_action(message):
            return
        if self.handle_reply_cb:
            await self.handle_reply_cb(message)
            return
        
    async def chat(self, message:str, recipient:str="*"):
        await self.dialogue.publish(pydantic=message_helper.create_user_send_message(
            content=message,
            recipient=recipient,
            ))
        