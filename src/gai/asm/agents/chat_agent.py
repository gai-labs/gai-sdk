from typing import AsyncGenerator, Optional
from gai.asm import AsyncStateMachine
from gai.mcp.client import McpAggregatedClient
from gai.lib.logging import getLogger
from gai.lib.config import GaiClientConfig
from gai.messages.monologue import Monologue
from gai.asm.agents.base import AgentBase

logger = getLogger(__name__)


class ChatAgent(AgentBase):

    def __init__(
        self,
        agent_name: str,
        llm_config: GaiClientConfig,
        monologue: Optional[Monologue]=None,
        #path: Optional[str] = None,
        aggregated_client: Optional[McpAggregatedClient]=None,
    ):
        super().__init__(
            agent_name=agent_name, 
            monologue=monologue,
            llm_config=llm_config
            )
                
        with AsyncStateMachine.StateMachineBuilder(
            """
            INIT --> CHAT
            CHAT --> FINAL
            """
        ) as builder:
            self.fsm = builder.build(
                {
                    "INIT": {
                        "input_data": {
                            "llm_config": {
                                "type": "getter",
                                "dependency": "get_llm_config",
                            }
                        }
                    },
                    "CHAT": {
                        "module_path": "gai.asm.states",
                        "class_name": "ChatState",
                        "title": "CHAT",
                        "input_data": {
                            "llm_config": {
                                "type": "state_bag",
                                "dependency": "llm_config",
                            },
                        },
                        "output_data": ["streamer", "get_assistant_message"],
                    },                    
                    "FINAL": {
                        "output_data": ["monologue","get_assistant_message"],
                    },
                },
                agent_name=agent_name,
                get_llm_config=lambda state: llm_config.model_dump(),
                monologue=self.monologue
            )
    
    def run(self, user_message: Optional[str]=None)-> AsyncGenerator[str, None]:
        self.fsm.state = "INIT"
        self.fsm.user_message = user_message

        async def streamer():
            logger.info(f"ChatAgent({self.fsm.monologue.agent_name}).run: inside streamer()")
            
            # LOOP UNTIL FINAL STATE
            while self.fsm.state != "FINAL":
                current_state = self.fsm.state
                await self.fsm.run_async()
                logger.info(f"Final state: {current_state} --> {self.fsm.state}")
                if self.fsm.state_bag.get("streamer"):
                    async for chunk in self.fsm.state_bag["streamer"]:
                        if chunk:
                            if isinstance(chunk, str):
                                yield (chunk)
                else:
                    yield None

        return streamer()

    def final_output(self):
        get_assistant_message = self.fsm.state_history[-1]["output"]["get_assistant_message"]
        return get_assistant_message()