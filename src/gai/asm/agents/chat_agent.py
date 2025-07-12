import os
from typing import overload, Union
from typing import Optional
from gai.asm import AsyncStateMachine, FileMonologue
from gai.mcp.client import McpAggregatedClient
from gai.lib.logging import getLogger
from gai.lib.config import GaiClientConfig

logger = getLogger(__name__)


class ChatAgent:
    @classmethod
    @overload
    def reset(cls, project_name: str, agent_name: str) -> None: ...

    @classmethod
    @overload
    def reset(cls, path: str) -> None: ...

    @classmethod
    def reset(
        cls,
        project_name: Optional[str] = None,
        agent_name: Optional[str] = None,
        *,
        path: Optional[str] = None,
    ) -> None:
        if path:
            pass
        elif project_name and agent_name:
            path = os.path.expanduser(f"~/.gai/logs/{project_name}_{agent_name}.log")
        else:
            raise TypeError(
                "reset() takes either (project_name, agent_name) or (path,)"
            )

        monologue = FileMonologue(file_path=path) if path else FileMonologue()
        monologue.reset()

    def __init__(
        self,
        agent_name: str,
        project_name: str,
        llm_config: GaiClientConfig,
        path: Optional[str] = None,
        aggregated_client: Optional[McpAggregatedClient]=None,
    ):
        log_file_path = path
        if not path:
            log_file_path = os.path.expanduser(
                f"~/.gai/logs/{project_name}_{agent_name}.log"
            )
        monologue = (
            FileMonologue(file_path=log_file_path) if log_file_path else FileMonologue()
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
                        "class_name": "AnthropicChatState",
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
                        "output_data": ["monologue"],
                    },
                },
                get_llm_config=lambda state: llm_config.model_dump(),
                monologue=monologue,
            )

    async def run_async(self, user_message: Optional[str] = None):
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
