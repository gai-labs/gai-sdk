import os
import re
import json
import time
import shutil
from typing import Any, Optional,Union
from gai.lib.constants import DEFAULT_GUID
from gai.lib.logging import getLogger
from gai.messages.typing import MessagePydantic,StateBodyPydantic,MessageHeaderPydantic

logger = getLogger(__file__)

class Monologue:
    
    def __init__(self, 
                 agent_name:str="Assistant", 
                 messages:Optional[Union["Monologue",list[MessagePydantic]]]=None,
                 dialogue_id:str=DEFAULT_GUID
                 ):
        self.dialogue_id = dialogue_id
        self.agent_name = agent_name
        
        self._messages:list[MessagePydantic]=[]
        if isinstance(messages,Monologue):
            self._messages = messages.list_messages()
        else:
            self._messages = messages or []
        
        self.created_at = int(time.time())
        self.updated_at = int(time.time())
        
    def add_user_message(self, state, content:str):
        message = MessagePydantic(
            header=MessageHeaderPydantic(
                sender="User",
                recipient=self.agent_name
            ),
            body=StateBodyPydantic(
                state_name=state.title,
                step_no=state.input["step"],
                role="user",
                content=content
            )
        )
        self._messages.append(message)
        return self    
    
    def add_assistant_message(self, state, content: str):
        message = MessagePydantic(
            header=MessageHeaderPydantic(
                sender=self.agent_name,
                recipient="User"
            ),
            body=StateBodyPydantic(
                state_name=state.title,
                step_no=state.input["step"],
                role="assistant",
                content=content
            )
        )
        self._messages.append(message)
        return self
    
    def copy(self):
        """Returns a copy of the monologue."""
        return Monologue(
            agent_name=self.agent_name,
            messages=self._messages.copy(),
            dialogue_id=self.dialogue_id
        )

    def list_messages(self)->list[MessagePydantic]:
        return self._messages
    
    def list_chat_messages(self) -> list[dict[str,Any]]:
        chat_messages= [{
            "role":m.body.role,
            "content":m.body.content
        } for m in self._messages]

        # clean up whitespace from system messages
        for message in chat_messages:
            if message["role"] == "system":
                message["content"] = re.sub(r'\s+',' ',message["content"])

        return chat_messages        
        



#-----

class FileMonologue(Monologue):
    
    def __init__(self, 
                 agent_name:str="Assistant", 
                 messages:Optional[Union["Monologue",list[MessagePydantic]]]=None,
                 dialogue_id:str=DEFAULT_GUID
                 ):
        super().__init__(agent_name, messages, dialogue_id)
        self.path = f"/tmp/{self.agent_name}.json"
        self._load(self.path)
        
    def _save(self,path:Optional[str]=None):
        if not path:
            path=f"/tmp/{self.agent_name}.json"
        with open(path,"w") as f:
            jsoned = json.dumps([m.model_dump() for m in self._messages],indent=4)
            f.write(jsoned)
            
    def _load(self,path:Optional[str]=None):
        if not path:
            path=f"/tmp/{self.agent_name}.json"
        if not os.path.exists(path):
            # Create empty monologue file if its not available.
            self.reset(path)
        with open(path,"r") as f:
            result=json.load(f)
            self._messages = [ MessagePydantic(**m) for m in result]
            
    def reset(self,path:Optional[str]=None):
        if not path:
            path=f"/tmp/{self.agent_name}.json"
        try:
            os.remove(path)
        except FileNotFoundError:
            pass
        with open(path,"w") as f:
            f.write(json.dumps([]))        

    def add_user_message(self, state, content:str):
        super().add_user_message(state, content)
        self._save(self.path)
        return self

    def add_assistant_message(self, state, content: str):
        super().add_assistant_message(state, content)
        self._save(self.path)
        return self
    
    