from gai.dialogue import DialogueBus
from gai.messages import MessagePydantic, message_helper
from gai.lib.utils import run_async_function
from rich import print

class AsyncAgentNode:
    
    # dummy agent that will return simulated messages

    def __init__(self, name: str, handle_send_cb=None):
        self.name = name
        self.broadcast_allowed = True
        self.handle_send_cb = handle_send_cb
        
    async def subscribe(self, dialogue:DialogueBus):
        self.dialogue = dialogue
        await self.dialogue.subscribe(subject="send",callback={self.name:self.handle_send})

    def validate_action(self, message:MessagePydantic)->bool:
        """
        Check if the message should be handled by this responder.
        """
        # Check if the message is a send message
        if message.body.type != "send":
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
        
    async def handle_send(self, message: MessagePydantic):

        # Check if action required

        if not self.validate_action(message):
            return

        # Call LLM and get a chunked response
        
        response=None
        if self.handle_send_cb:
            response = await self.handle_send_cb(message)
        content = ""
        chunk_no = 0
        if not response:
            return
        async for chunk in response:
            
            # Send chunk to User
            if hasattr(chunk, "extract"):
                chunk = chunk.extract()
            
            if isinstance(chunk, str):
                content += chunk

                try:
                    pydantic = message_helper.create_assistant_reply_chunk(
                        sender=self.name,
                        recipient=message.header.sender,
                        chunk_no=chunk_no,
                        chunk=chunk,
                        content=content
                    )
                    await self.dialogue.publish(pydantic=pydantic)
                except Exception as e:
                    print(f"Error publishing message: {e}")
            chunk_no += 1                

        # Send the final chunk
        pydantic = message_helper.create_assistant_reply_chunk(
            sender=self.name,
            recipient=message.header.sender,
            chunk_no=chunk_no,
            chunk="<eom>",
            content=content
        )
        await self.dialogue.publish(pydantic=pydantic)
