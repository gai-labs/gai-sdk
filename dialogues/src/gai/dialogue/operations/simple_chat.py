import asyncio
from typing import Union, Callable
from gai.lib.utils import get_app_path
from gai.lib.config import GaiClientConfig
from gai.dialogue import DialogueBus
from gai.messages import MessagePydantic, message_helper
from gai.messages.typing import ReplyBodyPydantic, SendBodyPydantic
from rich import print

class SimpleChatSender:

    def __init__(self,name:str, dialogue:DialogueBus, handle_reply_cb:Callable=None):
        self.name = name
        self.dialogue = dialogue
        self.handle_reply_cb = handle_reply_cb
        self.broadcast_allowed = True
        loop = asyncio.get_event_loop()
        if loop.is_running:
            asyncio.create_task(self.dialogue.subscribe("reply", {self.name:self.handle_reply}))
        else:
            asyncio.run(self.dialogue.subscribe("reply", {self.name:self.handle_reply}))
        
    def action_required(self, message:MessagePydantic)->bool:        
        
        # Check if the message is a reply message
        if not isinstance(message.body, ReplyBodyPydantic):
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
        
        print(f"[{self.name}] [bright_yellow]Handle reply message. message={message}[/bright_yellow]")        
        return True
        
    async def handle_reply(self,message:MessagePydantic):
        
        # Check if action required
        
        if not self.action_required(message):
            return
        
        if self.handle_reply_cb:
            # Do nothing but pass the message to the callback
            await self.handle_reply_cb(message)
            return
        
    async def chat(self, content:str, recipient:str="*"):
        message_helper.create_user_send_message(content=content,recipient=recipient)

class SimpleChatResponder:

    def __init__(self,name:str, dialogue:DialogueBus, handle_send_cb:Callable=None):
        self.name = name
        self.dialogue = dialogue
        self.broadcast_allowed = True
        self.handle_send_cb = handle_send_cb
        loop = asyncio.get_event_loop()
        if loop.is_running:
            asyncio.create_task(self.dialogue.subscribe("send", {self.name:self.handle_send}))
        else:
            asyncio.run(self.dialogue.subscribe("send", {self.name:self.handle_send}))
    
    def action_required(self, message:MessagePydantic)->bool:
        """
        Check if the message should be handled by this responder.
        """
        # Check if the message is a send message
        if not isinstance(message.body, SendBodyPydantic):
            return False
        
        # Check if the recipient is this responder or a wildcard
        if message.header.recipient != self.name and message.header.recipient != "*":
            print(f"[{self.name}] [gray0]Message not for me, ignore. message={message}[/gray0]")
            return False
        elif message.header.recipient == "*":
            if self.broadcast_allowed:
                print(f"[{self.name}] [green]Handle broadcast message. message={message}[/green]")
                return True
            else:
                print(f"[{self.name}] [gray0]Ignore broadcast message. message={message}[/gray0]")
                return False

        print(f"[{self.name}] [green]Handle send message. message={message}[/green]")        
        return True
    
    async def handle_send(self,message:MessagePydantic):
        
        # Check if action required

        if not self.action_required(message):
            return            
            
        # Do nothing but pass the message

        if self.handle_send_cb:
            await self.handle_send_cb(message)
