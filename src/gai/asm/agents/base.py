from abc import ABC, abstractmethod
from typing import AsyncGenerator, Optional
from gai.messages import Monologue

class AgentBase(ABC):
    
    def __init__(self, agent_name:str, monologue=None,llm_config=None):

        # Initialize monologue
        self.monologue = monologue
        if not self.monologue:
            self.monologue = Monologue(agent_name=agent_name)
        
        if not llm_config:
            raise ValueError("ChatAgent: llm_config is required.")

    
    @abstractmethod
    def run(self, user_message:Optional[str]=None) -> AsyncGenerator[str, None]:
        """Execute the agent’s main behavior."""
        ...

    async def run_async(self, user_message: Optional[str] = None) -> AsyncGenerator[str, None]:
        # maintain for backward compatibility
        return self.run(user_message)