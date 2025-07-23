from typing import AsyncGenerator, Optional
from gai.asm import AsyncStateMachine
from gai.messages import Monologue
from gai.mcp.client import McpAggregatedClient
from gai.lib.logging import getLogger
from gai.lib.config import GaiClientConfig
from gai.asm.agents.base import AgentBase

logger = getLogger(__name__)


class ToolUseAgent(AgentBase):
    def __init__(
        self,
        agent_name: str,
        llm_config: GaiClientConfig,
        aggregated_client: McpAggregatedClient,
        monologue: Optional[Monologue] = None,
    ):
        super().__init__(
            agent_name=agent_name, monologue=monologue, llm_config=llm_config
        )

        with AsyncStateMachine.StateMachineBuilder(
            """
            INIT --> HAS_MESSAGE
            HAS_MESSAGE --> CHAT: condition_true
            HAS_MESSAGE --> TOOL_USE: condition_false

            CHAT--> IS_TOOL_CALL
            
            IS_TOOL_CALL --> TOOL_USE: condition_true
            IS_TOOL_CALL --> FINAL: condition_false

            TOOL_USE --> FINAL
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
                    "HAS_MESSAGE": {
                        "module_path": "gai.asm.states",
                        "class_name": "PurePredicateState",
                        "title": "HAS_MESSAGE",
                        "predicate": "has_message",
                        "output_data": ["predicate_result"],
                        "conditions": ["condition_true", "condition_false"],
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
                        "output_data": ["tool_result", "get_assistant_message"],
                    },
                    "IS_TOOL_CALL": {
                        "module_path": "gai.asm.states",
                        "class_name": "PurePredicateState",
                        "title": "IS_TOOL_CALL",
                        "predicate": "is_tool_call",
                        "output_data": ["predicate_result"],
                        "conditions": ["condition_true", "condition_false"],
                    },
                    "FINAL": {
                        "output_data": ["monologue", "get_assistant_message"],
                    },
                },
                agent_name=agent_name,
                get_llm_config=lambda state: llm_config.model_dump(),
                get_mcp_client=lambda state: aggregated_client,
                monologue=self.monologue,
                has_message=self.has_message,
                is_tool_call=self.is_tool_call,
            )

    def has_message(self, state):
        state.machine.state_bag["predicate_result"] = False
        state.machine.state_bag["streamer"] = None

        if not state.machine.state_bag.get("user_message", None):
            logger.info("user_message not provided.")
            return state.machine.state_bag["predicate_result"]

        state.machine.state_bag["predicate_result"] = True
        return state.machine.state_bag["predicate_result"]

    def is_tool_call(self, state):
        messages = state.machine.monologue.list_messages()
        last_message = messages[-1] if messages else None
        if not last_message:
            raise ValueError("ToolUseAgent.is_tool_call: Monologue has no messages.")
        if last_message.body.role != "assistant":
            raise ValueError(
                "ToolUseAgent.is_tool_call: Last message is not from assistant."
            )
        try:
            tool_use = None
            if isinstance(last_message.body.content, list):
                for t in last_message.body.content:
                    if t.get("type") == "tool_use":
                        tool_use = t

            state.machine.state_bag["predicate_result"] = tool_use is not None
            return state.machine.state_bag["predicate_result"]
        except Exception as e:
            logger.error(f"[red]Error processing last message content: {e}[/red]")
            raise e

    def start(
        self, user_message: str, recap: Optional[str] = None
    ) -> AsyncGenerator[str, None]:
        """
        The user_message in this case contains the "goal" message.
        """
        if recap:
            user_message = f"""
            {user_message}

            Here is a recap of the conversation:
            {recap}
            """
        return self.run(user_message=user_message)

    def resume(self, user_message: Optional[str] = None) -> AsyncGenerator[str, None]:
        """
        If there is no user_message, then this is a regular call
        to continue with tool_use.

        If there is a user_message, then this user_message is
        a response to the llm interrupted flow by the tool 'user_input'.
        """
        return self.run(user_message=user_message)

    def run(self, user_message: Optional[str] = None) -> AsyncGenerator[str, None]:
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
                            # if isinstance(chunk, str):
                            yield (chunk)

        return streamer()

    def interrupt(self, user_message) -> AsyncGenerator[str, None]:
        """ """

        self.fsm.state = "INIT"

        # hijack the agent's instruction
        interrupt_template = """
        I am going to deviate a little and talk about something adhoc. 
        But I want you to come back on track after responding to this. 
        What I want to talk about is this - {user_message}
        """
        self.fsm.user_message = interrupt_template.format(user_message=user_message)

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

        return streamer()

    def final_output(self):
        get_assistant_message = self.fsm.state_history[-1]["output"][
            "get_assistant_message"
        ]
        return get_assistant_message()
