import re
import uuid
import time
from typing import Optional
from pydantic import BaseModel,ConfigDict
from gai.lib.errors import InternalException, MessageNotFoundException
from gai.lib.logging import getLogger
logger = getLogger(__file__)

class MonologueMessage(BaseModel):
    model_config = ConfigDict(from_attributes=True)  # Allows the model to work with ORM objects
    Order: int
    Name: str
    Role: str
    Title: str
    Content: str
    ChildMessages: Optional[list['MonologueMessage']] = []
    Timestamp: Optional[int] = None

class MonologueMessageBuilder:

    def __init__(self, messages:list[MonologueMessage]=None):
        self.messages = messages or []

    def AddSystemMessage(self, content, title="Init", timestamp:int=None):
        self.messages.append(MonologueMessage(
            Order=len(self.messages),
            Name="System",
            Role="system",
            Title=title,
            Content=content,
            Timestamp= timestamp or int(time.time())
        ))
        return self

    def AddUserMessage(self, content, title="Init", timestamp:int=None):
        self.messages.append(MonologueMessage(
            Order=len(self.messages),
            Name="User",
            Role="user",
            Title=title,
            Content=content,
            Timestamp= timestamp or int(time.time())
        ))
        return self

    def AddAssistantMessage(self, name="Assistant", content=None,title="Init",timestamp:int=None):
        if content is None:
            content = ""
        self.messages.append(MonologueMessage(
            Order=len(self.messages),
            Name=name,
            Role="assistant",
            Title=title,
            Content=content,
            Timestamp= timestamp or int(time.time())
        ))
        return self

    def Build(self):
        return self.messages
    
    def BuildRoleMessages(self):
        return [{
            "role": x.Role, 
            "content": x.Content
            } for x in self.messages]
    
    def ToChatMessages(self):
        chat_messages = []
        if isinstance(self.messages, list) and all(isinstance(m, MonologueMessage) for m in self.messages):
            chat_messages = [{"role":message.Role, "content":message.Content} for message in self.messages]
        elif isinstance(self.messages,list) and not all("role" in m and "content" in m for m in self.messages):
            raise Exception(f"Invalid message format: {self.messages}")

        # clean up whitespace from system messages
        for message in chat_messages:
            if message["role"] == "system":
                message["content"] = re.sub(r'\s+',' ',message["content"])
        
        return chat_messages