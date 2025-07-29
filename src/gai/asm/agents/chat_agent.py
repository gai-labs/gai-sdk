from .tool_use_agent import ToolUseAgent
from typing import Optional
from gai.lib.logging import getLogger
from gai.lib.config import GaiClientConfig
from gai.messages import Monologue
from gai.mcp.client import McpAggregatedClient

logger = getLogger(__name__)


class ChatAgent(ToolUseAgent):
    def __init__(
        self,
        agent_name: str,
        llm_config: GaiClientConfig,
        monologue: Optional[Monologue] = None,
    ):
        super().__init__(
            agent_name=agent_name,
            monologue=monologue,
            llm_config=llm_config,
            aggregated_client=McpAggregatedClient([])  # Default empty client
        )
        pass
