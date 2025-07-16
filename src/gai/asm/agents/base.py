from abc import ABC, abstractmethod
from typing import AsyncGenerator, Optional

class AgentBase(ABC):
    
    @abstractmethod
    def run(self, user_message:Optional[str]=None) -> AsyncGenerator[str, None]:
        """Execute the agent’s main behavior."""
        ...

    async def run_async(self, user_message: Optional[str] = None) -> AsyncGenerator[str, None]:
        # maintain for backward compatibility
        return self.run(user_message)