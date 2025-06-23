import os
from typing import Optional
from gai.asm import AsyncStateMachine, FileMonologue
from gai.mcp.client import McpAggregatedClient
from gai.lib.logging import getLogger
from gai.lib.config import GaiClientConfig


logger = getLogger(__name__)


class ToolUseAgent:
    def __init__(
        self,
        agent_name: str,
        prompt_template: str,
        project_name: str,
        llm_config: GaiClientConfig,
        mcp_client: McpAggregatedClient,
    ):
        self.prompt_template = prompt_template
        log_file_path = os.path.expanduser(
            f"~/.gai/logs/{project_name}_{agent_name}.log"
        )
        monologue = (
            FileMonologue(file_path=log_file_path) if log_file_path else FileMonologue()
        )
        with AsyncStateMachine.StateMachineBuilder(
            """
            INIT --> IS_TERMINATED
            IS_TERMINATED --> FINAL: condition_true
            IS_TERMINATED --> HAS_MESSAGE
            HAS_MESSAGE --> TOOL_CALL: condition_true
            HAS_MESSAGE --> TOOL_USE: condition_false
            TOOL_CALL--> TOOL_USE
            TOOL_USE --> CONTINUE_TOOL_USE
            CONTINUE_TOOL_USE --> FINAL: condition_true
            CONTINUE_TOOL_USE --> TERMINATE: condition_false
            TERMINATE --> FINAL
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
                    "IS_TERMINATED": {
                        "module_path": "gai.asm.states",
                        "class_name": "PurePredicateState",
                        "title": "IS_TERMINATED",
                        "predicate": "is_terminated",
                        "output_data": ["predicate_result"],
                        "conditions": ["condition_true", "condition_false"],
                    },
                    "HAS_MESSAGE": {
                        "module_path": "gai.asm.states",
                        "class_name": "PurePredicateState",
                        "title": "HAS_MESSAGE",
                        "predicate": "has_message",
                        "output_data": ["predicate_result"],
                        "conditions": ["condition_true", "condition_false"],
                    },
                    "TERMINATE": {
                        "module_path": "gai.asm.states",
                        "class_name": "PureActionState",
                        "title": "TERMINATE",
                        "action": "terminate",
                    },
                    "FINAL": {
                        "output_data": ["monologue"],
                    },
                },
                get_llm_config=lambda state: llm_config.model_dump(),
                get_mcp_client=lambda state: mcp_client,
                monologue=monologue,
                terminate=self.terminate,
                is_terminated=self.is_terminated,
                has_message=self.has_message,
                continue_tool_use=self.continue_tool_use,
            )

    async def terminate(self, state):
        logger.info("Terminating the state machine.")
        state.machine.monologue.add_user_message(state=state, content="TERMINATE")
        return state

    def is_terminated(self, state):
        state.machine.state_bag["predicate_result"] = self.fsm.monologue.is_terminated()
        return state.machine.state_bag["predicate_result"]

    def has_message(self, state):
        state.machine.state_bag["predicate_result"] = False

        if not state.machine.state_bag.get("user_message", None):
            logger.info("user_message not provided.")
            return state.machine.state_bag["predicate_result"]

        state.machine.state_bag["predicate_result"] = True
        return state.machine.state_bag["predicate_result"]

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

    def make_user_message(self, goal: str):
        return (
            f"""
        1. Goal
        
        {goal}
        
        """
            + self.prompt_template
        )

    async def run_async(self, goal: Optional[str] = None):
        self.fsm.state = "INIT"
        if goal:
            # self.fsm.user_message = self.prompt_template.format(goal=goal)
            self.fsm.user_message = self.make_user_message(goal)
        else:
            self.fsm.user_message = None

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

    async def interrupt_async(self, message):
        self.fsm.state = "INIT"

        # hijack the agent's instruction
        interrupt_template = """
        I am going to deviate a little and talk about something adhoc. 
        But I want you to come back on track after responding to this. 
        What I want to talk about is this - {message}
        """
        self.fsm.user_message = interrupt_template.format(message=message)

        # Remove the last tool_use from assistant since the user has interrupted the flow.

        messages = self.fsm.monologue.list_messages()
        if messages:
            last_message = messages[-1]
            if isinstance(last_message.body.content, list):
                # Create a new list without tool_use blocks
                last_message.body.content = [
                    content_block
                    for content_block in last_message.body.content
                    if content_block["type"] != "tool_use"
                ]
            if last_message.body.content == []:
                # If the content is empty, remove the message
                self.fsm.monologue.pop()

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
