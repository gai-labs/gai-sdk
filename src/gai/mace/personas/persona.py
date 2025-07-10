from gai.asm.agents import ChatAgent as AsmChatAgent
from typing import Optional
from gai.lib.config import GaiClientConfig
from gai.dialogue import DialogueBus
from gai.lib.constants import DEFAULT_GUID
from gai.asm import Monologue
from .dtos import PersonaImagePydantic
import importlib
from gai.mcp.client import McpAggregatedClient

class Persona:
    
    def __init__(
        self,
        name:str,
        sex:str,
        job_description:str,
        self_introduction:str,
        skills:str,
        traits:str,
        agent_class: str,
        llm_config:GaiClientConfig,
        mcp_client: Optional[McpAggregatedClient]=None,
        monologue: Optional[Monologue]=None,
        dialogue: Optional[DialogueBus]=None,
        portraits: Optional[PersonaImagePydantic]=None
    ):
        self.id = DEFAULT_GUID
        self.owner_id = DEFAULT_GUID
        self.name = name
        self.job_description = job_description
        self.self_introduction = self_introduction
        self.sex = sex
        self.skills = skills
        self.traits = traits
        self.agent_class = agent_class
        self.llm_config = llm_config
        self.monologue = monologue
        self.dialogue=dialogue
        self.portraits=portraits
        
        # load agent via reflection based on agent_class.
        # The package should be at gai.asm.agents
        try:
            module = importlib.import_module(f"gai.asm.agents")
            AgentCls = getattr(module,agent_class)
        except (ModuleNotFoundError, AttributeError) as e:
            raise ImportError(f"Could not load agent_class '{agent_class}'") from e
        self.agent = AgentCls(
            project_name=self.dialogue.dialogue_id,
            agent_name=self.name,
            llm_config=self.llm_config,
            mcp_client=mcp_client
        )        
                
    async def run_async(self,user_message):
        return await self.agent.run_async(user_message)