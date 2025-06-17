import os
from gai.asm import AsyncStateMachine, FileMonologue
from gai.mcp.client import McpAggregatedClient
from gai.lib.logging import getLogger
from gai.lib.config import GaiClientConfig


logger = getLogger(__name__)


class ToolUseAgent:
    def __init__(
        self,
        user_message: str,
        agent_name: str,
        project_name: str,
        llm_config: GaiClientConfig,
        mcp_client: McpAggregatedClient,
    ):
        log_file_path = os.path.expanduser(
            f"~/.gai/logs/{project_name}_{agent_name}.log"
        )
        monologue = (
            FileMonologue(file_path=log_file_path) if log_file_path else FileMonologue()
        )
        with AsyncStateMachine.StateMachineBuilder(
            """
            INIT --> TOOL_CALL
            TOOL_CALL--> TOOL_USE
            TOOL_USE --> CONTINUE_TOOL_USE
            CONTINUE_TOOL_USE --> TOOL_USE: condition_true
            CONTINUE_TOOL_USE --> FINAL: condition_false            
            """
        ) as builder:
            self.fsm = builder.build(
                {
                    "INIT": {
                        "input_data": {
                            "user_message": {
                                "type": "getter",
                                "dependency": "get_user_message",
                            },
                            "llm_config": {
                                "type": "getter",
                                "dependency": "get_llm_config",
                            },
                            "mcp_client": {
                                "type": "getter",
                                "dependency": "get_mcp_client",
                            },
                        }
                    },
                    "TOOL_CALL": {
                        "module_path": "gai.asm.states",
                        "class_name": "AnthropicToolCallState",
                        "title": "TOOL_CALL",
                        "input_data": {
                            "user_message": {
                                "type": "state_bag",
                                "dependency": "user_message",
                            },
                            "llm_config": {
                                "type": "state_bag",
                                "dependency": "llm_config",
                            },
                            "mcp_client": {
                                "type": "state_bag",
                                "dependency": "mcp_client",
                            },
                        },
                        "output_data": ["streamer", "get_assistant_message"],
                    },
                    "TOOL_USE": {
                        "module_path": "gai.asm.states",
                        "class_name": "AnthropicToolUseState",
                        "title": "TOOL_USE",
                        "input_data": {
                            "llm_config": {
                                "type": "state_bag",
                                "dependency": "llm_config",
                            },
                            "mcp_client": {
                                "type": "state_bag",
                                "dependency": "mcp_client",
                            },
                        },
                        "output_data": ["tool_result"],
                    },
                    "CONTINUE_TOOL_USE": {
                        "module_path": "gai.asm.states",
                        "class_name": "PurePredicateState",
                        "title": "CONTINUE_TOOL_USE",
                        "predicate": "continue_tool_use",
                        "output_data": ["predicate_result"],
                        "conditions": ["condition_true", "condition_false"],
                    },
                    "FINAL": {
                        "output_data": ["monologue"],
                    },
                },
                get_user_message=lambda state: user_message,
                get_llm_config=lambda state: llm_config.model_dump(),
                get_mcp_client=lambda state: mcp_client,
                monologue=monologue,
                continue_tool_use=self.continue_tool_use,
            )

    def continue_tool_use(self, state):
        messages = state.machine.monologue.list_messages()
        last_message = messages[-1] if messages else None

        while last_message and last_message.body.role != "assistant":
            logger.warning(
                "Last message is not from assistant or no messages found. Dropping message and retry."
            )
            state.machine.monologue.pop()
            messages = state.machine.monologue.list_messages()
            if not messages:
                raise ValueError(
                    "No valid previous message were found for predicate to work. Messages might be corrupted."
                )
            last_message = messages[-1] if messages else None

        if not last_message or last_message.body.role != "assistant":
            raise ValueError("Last message is not from assistant or no messages found.")

        try:
            state.machine.state_bag["predicate_result"] = False
            for item in last_message.body.content:
                if item["type"] == "tool_use":
                    state.machine.state_bag["predicate_result"] = True
                    break
            return state.machine.state_bag["predicate_result"]

        except Exception as e:
            logger.error(f"[red]Error processing last message content: {e}[/red]")
            raise e

    @classmethod
    def reset(cls, project_name: str, agent_name: str):
        log_file_path = os.path.expanduser(
            f"~/.gai/logs/{project_name}_{agent_name}.log"
        )
        monologue = (
            FileMonologue(file_path=log_file_path) if log_file_path else FileMonologue()
        )
        monologue.reset()

    async def run_once_async(self):
        async def streamer():
            async for chunk in self.fsm.state_bag["streamer"]:
                if isinstance(chunk, str):
                    yield chunk

        if self.fsm.state != "FINAL":
            current_state = self.fsm.state
            await self.fsm.run_async()
            logger.info(f"Final state: {current_state} --> {self.fsm.state}")
            return streamer
        else:
            logger.info("Agent is already in the final state, no action taken.")

    async def run_until_final_async(self):
        async def streamer():
            # LOOP UNTIL FINAL STATE
            while self.fsm.state != "FINAL":
                await self.run_once_async()
                async for chunk in self.fsm.state_bag["streamer"]:
                    if chunk:
                        if isinstance(chunk, str):
                            yield (chunk)

        return streamer
