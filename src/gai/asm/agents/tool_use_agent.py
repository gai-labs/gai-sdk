from gai.asm import AsyncStateMachine
from gai.lib.logging import getLogger

logger = getLogger(__name__)


class ToolUseAgent:
    def __init__(self, user_message: str):
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
                            "user_message": user_message,
                            "llm_config": {
                                "type": "getter",
                                "dependency": "get_llm_config",
                            },
                            "mcp_server_names": ["mcp-filesystem", "mcp-web"],
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
                            "mcp_server_names": {
                                "type": "state_bag",
                                "dependency": "mcp_server_names",
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
                            "mcp_server_names": {
                                "type": "state_bag",
                                "dependency": "mcp_server_names",
                            },
                        },
                        "output_data": ["streamer", "get_assistant_message"],
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
                get_llm_config=lambda state: {
                    "client_type": "anthropic",
                    # "model": "claude-opus-4-20250514",
                    "model": "claude-sonnet-4-20250514",
                    "max_tokens": 32000,
                    "temperature": 0.7,
                    "top_p": 0.95,
                    "tools": True,
                },
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
            state.machine.monologue.save()
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

    async def run_async(self):
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
