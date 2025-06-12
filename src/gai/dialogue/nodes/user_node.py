from gai.dialogue import DialogueBus
from gai.messages import MessagePydantic,message_helper
from gai.lib.utils import run_async_function
from rich import print

class UserNode:
    
    # node that will send user message and handle reply messages from agents

    def __init__(self, dialogue: DialogueBus, handle_reply_cb=None):
        self.name = "User"
        self.dialogue = dialogue        
        self.broadcast_allowed = True
        self.handle_reply_cb = handle_reply_cb
        run_async_function(
            self.dialogue.subscribe,
            subject="reply",
            callback={self.name:self.handle_reply}
            )
        
    def validate_action(self, message:MessagePydantic)->bool:        
        
        # Check if the message is a reply message
        if message.body.type != "reply":
            return False
        
        # Check if the recipient is this responder or a wildcard
        if message.header.recipient != self.name and message.header.recipient != "*":
            print(f"[{self.name}] [gray0]Message not for me, ignore. message={message}[/gray0]")
            return False
        elif message.header.recipient == "*":
            if self.broadcast_allowed:
                print(f"[{self.name}] [bright_yellow]Handle broadcast message. message={message}[/bright_yellow]")
                return True
            else:
                print(f"[{self.name}] [gray0]Ignore broadcast message. message={message}[/gray0]")
                return False
        
        #print(f"[{self.name}] [bright_yellow]Handle reply message. message={message}[/bright_yellow]")        
        return True    
   
    def handle_reply(self, message: MessagePydantic):
        if not self.validate_action(message):
            return
        if self.handle_reply_cb:
            self.handle_reply_cb(message)
            return
        
    async def chat(self, message:str, recipient:str="*"):
        await self.dialogue.publish(pydantic=message_helper.create_user_send_message(
            content=message,
            recipient=recipient,
            ))
        