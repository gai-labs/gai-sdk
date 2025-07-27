from typing import AsyncGenerator, Optional
from gai.asm.base import StateBase
from gai.asm import AsyncStateMachine
from gai.asm.agents.base import AgentBase
from gai.lib.logging import getLogger
from gai.llm.lib import LLMGeneratorRetryPolicy
from gai.llm.openai import AsyncOpenAI
from gai.lib.config import GaiClientConfig
from gai.messages import Monologue
from gai.mcp.client import McpAggregatedClient

logger = getLogger(__name__)


class AnthropicStateBase(StateBase):

    async def _raw_llm_stream(self, llm_client, llm_model, messages, tools):
        """Call the LLM once and yield raw (extracted) chunks."""
        async def _gen():
            resp = await llm_client.chat.completions.create(
                model=llm_model,
                messages=messages,
                tools=tools,
                stream=True,
            )
            async for c in resp:
                if not c:
                    continue
                try:
                    chunk = c.extract()
                except Exception:
                    logger.warning("chunk.extract() failed", exc_info=True)
                    chunk = c
                yield chunk
        return LLMGeneratorRetryPolicy(self.machine).run(_gen)

    def _make_streamer(self, llm_client, llm_model, messages, tools):
        """
        Main function is to stream text followed by the last chunk for the completed object.
        """
        async def _streamer():
            has_text = False
            async for chunk in await self._raw_llm_stream(
                llm_client, llm_model, messages, tools
            ):
                # streaming‐text
                if isinstance(chunk, str):
                    has_text = True
                    yield chunk
                    continue

                # final (non‐str) chunk → insert newline or “Thinking…”
                yield "Thinking...\n" if not has_text else "\n"

                # record assistant message, history & getter
                self.machine.monologue.add_assistant_message(
                    state=self, content=chunk
                )
                self.machine.state_history[-1]["output"]["monologue"] = (
                    self.machine.monologue.copy()
                )

                # build get_assistant_message
                if (
                    isinstance(chunk, list)
                    and chunk
                    and isinstance(chunk[0], dict)
                    and "text" in chunk[0]
                ):
                    self.machine.state_bag["get_assistant_message"] = lambda: chunk[0]["text"]
                else:
                    self.machine.state_bag["get_assistant_message"] = lambda: chunk.copy(
                    )

                # yield the tool‐call object itself, then exit
                yield chunk
                return
        return _streamer()


"""
AnthropicChatState

Make a chat call to Claude models.

"""


class AnthropicChatState(AnthropicStateBase):
    """
    state schema:
    {
        "CHAT": {
            "module_path": "gai.asm.states",
            "class_name": "AnthropicChatState",
            "title": "CHAT",
            "input_data": {
                "llm_config": {"type": "state_bag", "dependency": "llm_config"},
                "mcp_server_names": {
                    "type": "state_bag",
                    "dependency": "mcp_server_names",
                },
            },
            "output_data": ["streamer", "get_assistant_message"],
        }
    }
    """

    def __init__(self, machine):
        super().__init__(machine)

    async def run_async(self):
        # Get User Message
        if not self.machine.user_message:
            raise Exception("AnthropicToolCallState: user_message is missing.")

        # Get llm client
        llm_config = self.input["llm_config"]
        llm_client = AsyncOpenAI(llm_config)

        # Get model
        llm_model = llm_config["model"]

        # Create system message from user message
        system_message = f"""
            Your name is {self.machine.agent_name} within the context of this conversation and you will always respond as such.
            Do not refer to yourself as an AI or a bot or confuse your name with other agents.
           
            You may respond to my following message using the context you have learnt.
            {self.machine.user_message}
            """

        mcp_client = self.input.get("mcp_client")
        tools = []
        if mcp_client:
            tools = await mcp_client.list_tools()

        # Case 1: LLM interrupt flow.
        # If previous message contains "user_input" tool use,
        # and user_message exists, this is to resume with user input.
        # Exit and forward to the "tool_use" state for processing.

        last_tool_calls = self.machine.monologue.get_last_toolcalls()
        if (
            last_tool_calls
            and any(result["tool_name"] == "user_input" for result in last_tool_calls)
            and self.machine.user_message
        ):
            self.machine.state_bag["streamer"] = None
            return  # Exit the state early
        # End of Case 1

        self.machine.monologue.add_user_message(
            state=self, content=system_message)
        messages = self.machine.monologue.list_chat_messages()

        self.machine.state_bag["streamer"] = self._make_streamer(
            llm_client,
            llm_model,
            messages,
            tools
        )


"""
AnthropicToolUseState

This state is made up of 2 actions. The first is to make a tool call and the second is the marshall the result from the
tool call and send it to the LLM for response.
"""


class AnthropicToolUseState(AnthropicStateBase):
    """
    state schema:
    {
        "TOOL_CALL": {
            "module_path": "gai.asm.states",
            "class_name": "AnthropicToolCallState",
            "title": "TOOL_CALL",
            "input_data": {
                "user_message": {"type": "state_bag", "dependency": "user_message"},
                "llm_config": {"type": "state_bag", "dependency": "llm_config"},
                "mcp_server_names": {
                    "type": "state_bag",
                    "dependency": "mcp_server_names",
                },
            },
            "output_data": ["streamer", "get_assistant_message"],
        }
    }
    """

    def __init__(self, machine):
        super().__init__(machine)

    async def _use_tool(self, last_tool_calls):
        """
        This function is used to make a tool call to the MCP client and return the result.
        """

        tool_calls = last_tool_calls

        mcp_client = self.input["mcp_client"]

        try:
            tool_results = []
            for item in tool_calls:
                logger.debug(
                    f"Using tool: {item['tool_name']} with input: {item['arguments']}"
                )

                tool_result = await mcp_client.call_tool(
                    tool_name=item["tool_name"], **item["arguments"]
                )
                logger.debug(f"Tool result: {tool_result}")

                # Extract just the text content from MCP tool result, not the full structure
                if hasattr(tool_result, "content") and tool_result.content:
                    result_content = tool_result.content
                    if isinstance(result_content, list) and len(result_content) > 0:
                        # Get the text from the first content block
                        result_text = (
                            result_content[0].text
                            if hasattr(result_content[0], "text")
                            else str(result_content[0])
                        )
                    else:
                        result_text = str(result_content)
                else:
                    result_text = str(tool_result)

                tool_result = {
                    "type": "tool_result",
                    "tool_use_id": item["tool_use_id"],
                    "content": result_text,
                }

                tool_results.append(tool_result)

            self.machine.state_bag["tool_results"] = tool_results
            return tool_results

        except Exception as e:
            logger.error(f"Error processing last message content: {e}")
            raise e

    def _make_user_input_tool_result(self, last_tool_calls):
        """
        This function is used to artificially create a tool result for user_input using user_message as opposed to using MCP tool.
        """

        # This function is used to create a tool result for user input
        item = next(
            (t for t in last_tool_calls if t.get("tool_name") == "user_input"),
            None,
        )
        if item:
            logger.info("AnthropicToolUseState: user_input tool found.")
            tool_result = {
                "type": "tool_result",
                "tool_use_id": item["tool_use_id"],
                "content": self.machine.state_bag["user_message"],
            }
            return tool_result
        return None

    async def run_async(self):
        # Get llm client
        llm_config = self.input["llm_config"]
        llm_client = AsyncOpenAI(llm_config)

        # Get mcp client
        mcp_client = self.input["mcp_client"]
        tools = await mcp_client.list_tools()

        # Get model
        llm_model = llm_config["model"]

        # Case 1: Either user terminated or LLM terminated. Stream nothing.

        if self.machine.monologue.is_terminated():
            logger.info(
                "AnthropicToolUseState: Task completed, nothing to continue.")
            self.machine.state_bag["streamer"] = None
            return  # Exit the state early

        # If it is not terminated, then
        # last_tool_calls should exist.
        last_tool_calls = self.machine.monologue.get_last_toolcalls()

        if any(result["tool_name"] == "user_input" for result in last_tool_calls):
            if self.machine.state_bag.get("user_message", None) is None:
                # Case 2a: LLM interrupt flow.
                # LLM request input from user by responding with a tool call of "user_input"
                # but user_message is None. Stream nothing.
                logger.info(
                    "AnthropicToolUseState: Pending user input, nothing to continue."
                )
                self.machine.state_bag["streamer"] = None
                return

            # Case 2b: LLM interrupt flow.
            # LLM request input from user by responding with a tool call of "user_input"
            # and user_message is provided. Stream LLM response.
            # tool_result is created from user_input instead of using any tools.
            # That is why "user_input" is a pseudo tool.
            tool_result = self._make_user_input_tool_result(
                last_tool_calls=last_tool_calls
            )
            tool_results = [tool_result]

        else:
            # Case 3: Normal flow. Proceed to use MCP tools and stream LLM response.
            tool_results = await self._use_tool(last_tool_calls=last_tool_calls)

        # At this point, tool_results should either be a list of real tool results or psuedo tool result.

        assistant_message = ""

        self.machine.monologue.add_user_message(
            state=self, content=tool_results)
        messages = self.machine.monologue.list_chat_messages()

        self.machine.state_bag["streamer"] = self._make_streamer(
            llm_client,
            llm_model,
            messages,
            tools
        )


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
            raise ValueError(
                "ToolUseAgent.is_tool_call: Monologue has no messages.")
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
            logger.error(
                f"[red]Error processing last message content: {e}[/red]")
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
                logger.info(
                    f"Final state: {current_state} --> {self.fsm.state}")
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
        self.fsm.user_message = interrupt_template.format(
            user_message=user_message)

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
                logger.info(
                    f"Final state: {current_state} --> {self.fsm.state}")
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
