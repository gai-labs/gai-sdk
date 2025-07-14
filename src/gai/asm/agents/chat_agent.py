import os
from typing import overload, Union
from typing import Optional
from gai.asm import AsyncStateMachine
from gai.mcp.client import McpAggregatedClient
from gai.lib.logging import getLogger
from gai.lib.config import GaiClientConfig
from gai.messages.monologue import Monologue
from gai.messages.dialogue import Dialogue

logger = getLogger(__name__)


class ChatAgent:
    # @classmethod
    # @overload
    # def reset(cls, project_name: str, agent_name: str) -> None: ...

    # @classmethod
    # @overload
    # def reset(cls, path: str) -> None: ...

    # @classmethod
    # def reset(
    #     cls,
    #     project_name: Optional[str] = None,
    #     agent_name: Optional[str] = None,
    #     *,
    #     path: Optional[str] = None,
    # ) -> None:
    #     if path:
    #         pass
    #     elif project_name and agent_name:
    #         path = os.path.expanduser(f"~/.gai/logs/{project_name}_{agent_name}.log")
    #     else:
    #         raise TypeError(
    #             "reset() takes either (project_name, agent_name) or (path,)"
    #         )

    #     monologue = FileMonologue(file_path=path) if path else FileMonologue()
    #     monologue.reset()

    def __init__(
        self,
        agent_name: str,
        llm_config: GaiClientConfig,
        monologue: Optional[Monologue]=None,
        #path: Optional[str] = None,
        aggregated_client: Optional[McpAggregatedClient]=None,
        recap: Optional[str] = "",
    ):
        # Initialize monologue
        self.monologue = monologue
        if not self.monologue:
            self.monologue = Monologue(agent_name=agent_name)

        # # Initialize dialogue
        # self.dialogue = dialogue
        # recap = ""
        # if self.dialogue:
        #     recap = self.dialogue.extract_recap()
                
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
                            },
                            "recap": {
                                "type": "getter",
                                "dependency": "get_recap",
                            },
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
                            "recap": {
                                "type": "state_bag",
                                "dependency": "recap",
                            },
                        },
                        "output_data": ["streamer", "get_assistant_message"],
                    },                    
                    "FINAL": {
                        "output_data": ["monologue","get_assistant_message"],
                    },
                },
                get_llm_config=lambda state: llm_config.model_dump(),
                get_recap=lambda state: recap,
                monologue=self.monologue
            )

    async def run_async(self, user_message: str):
        self.fsm.state = "INIT"
        self.fsm.user_message = user_message

        async def streamer():
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

        return streamer

    def final_output(self):
        get_assistant_message = self.fsm.state_history[-1]["output"]["get_assistant_message"]
        return get_assistant_message()