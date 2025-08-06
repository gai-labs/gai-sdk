from typing import AsyncGenerator, Optional
from gai.asm.base import StateBase
from gai.asm import AgenticStateMachine
from gai.asm.agents.base import AgentBase
from gai.lib.logging import getLogger
from gai.llm.lib import LLMGeneratorRetryPolicy
from gai.llm.openai import AsyncOpenAI
from gai.lib.config import GaiClientConfig
from gai.messages import Monologue
from gai.mcp.client import McpAggregatedClient

logger = getLogger(__name__)


class PendingUserInputError(Exception):
    """
    Raised when the LLM has requested user input via the ‘user_input’ tool
    and no input has yet been provided.
    """

    pass


class AutoResumeError(Exception):
    """
    Raised when agent's task is completed and cannot resume()
    """

    pass


class MissingUserMessageError(Exception):
    """
    Raised when chat state is run without a user message.
    """

    pass


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

        from gai.messages import message_helper

        chat_messages = message_helper.convert_to_chat_messages(messages)
        chat_messages = message_helper.shrink_messages(chat_messages)

        async def _streamer():
            has_text = False

            async def process_last_chunk(last_chunk):
                self.machine.state_bag["get_assistant_message"] = None
                self.machine.state_bag["user_message"] = None

                # The last chunk is a special case. It is either a dict that contains the finish reason.
                # Or it is a list of content blocks from Anthropic.
                # Or it could be the last chunk of the text stream which just means that there is no object to return.

                if isinstance(last_chunk, list):
                    # record assistant message, history & getter
                    self.machine.monologue.add_assistant_message(
                        state=self, content=last_chunk
                    )

                    self.machine.state_history[-1]["output"]["monologue"] = (
                        self.machine.monologue.copy()
                    )

                    for item in last_chunk:
                        if item.get("type") == "text":
                            # The final text contain the complete assistant message
                            text = item.get("text", "")
                            self.machine.state_bag["get_assistant_message"] = (
                                lambda: text
                            )
                        if item.get("type") == "tool_use":
                            if item.get("name") == "user_input":
                                self.machine.state_bag["is_user_input"] = True
                            else:
                                self.machine.state_bag["is_user_input"] = False

                elif (
                    isinstance(last_chunk, dict)
                    and last_chunk.get("type") == "finish_reason"
                ):
                    # The last chunk is a dict with type=finish_reason
                    # This means that the LLM has finished its response.
                    pass
                elif isinstance(last_chunk, str):
                    pass
                else:
                    raise ValueError(
                        f"AnthropicStateBase._streamer: The last chunk should be a str or list, got {last_chunk} instead."
                    )

                return last_chunk

            has_text = False
            async for chunk in await self._raw_llm_stream(
                llm_client, llm_model, chat_messages, tools
            ):
                # Stream until encountering a list type chunk.

                if isinstance(chunk, str):
                    has_text = True
                    yield chunk
                    continue

                # We know that this iteration is the last chunk so we need to say something if not already done so.

                yield "Thinking...\n" if not has_text else "\n"
                if isinstance(chunk, list):
                    break

            last_chunk = await process_last_chunk(chunk)
            yield last_chunk
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
            raise MissingUserMessageError(
                "AnthropicChatState.run_async: user_message is missing."
            )

        # Get llm client
        llm_config = self.input["llm_config"]
        if isinstance(llm_config, GaiClientConfig):
            llm_config = llm_config.model_dump()
        llm_client = AsyncOpenAI(llm_config)

        # Get model
        llm_model = llm_config["model"]

        # Create system message from user message
        system_message = f"""
            Your name is {self.machine.agent_name} within the context of this conversation and you will always respond as such.
            Do not refer to yourself as an AI or a bot or confuse your name with other agents.
           
            You may respond to my following message using the context you have learnt.
            {self.machine.user_message}
            
            You may ask me for more information if you need to clarify my request but ask just enough to get the information you need to get started.
            If you need to ask for more information, please use the "user_input" tool to get the information from me.
            Never call "user_input" without providing a description of what you need from me.
            Do not be vague and expect an input from me, be specific about what you want.
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

        self.machine.monologue.add_user_message(state=self, content=system_message)

        messages = self.machine.monologue.list_messages()

        self.machine.state_bag["streamer"] = self._make_streamer(
            llm_client, llm_model, messages, tools
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
        if isinstance(llm_config, GaiClientConfig):
            llm_config = llm_config.model_dump()
        llm_client = AsyncOpenAI(llm_config)

        # Get mcp client
        mcp_client = self.input.get("mcp_client")
        tools = []
        if mcp_client:
            tools = await mcp_client.list_tools()

        # Get model
        llm_model = llm_config["model"]

        # Case 1: Either user terminated or LLM terminated. Streamer is `None`

        if self.machine.monologue.is_terminated():
            logger.info("AnthropicToolUseState: Task completed, nothing to continue.")
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
                raise PendingUserInputError(
                    "AnthropicToolUseState.run_async: pending user input"
                )

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

        self.machine.monologue.add_user_message(state=self, content=tool_results)
        messages = self.machine.monologue.list_messages()

        self.machine.state_bag["streamer"] = self._make_streamer(
            llm_client, llm_model, messages, tools
        )


class ToolUseAgent:
    def __init__(
        self,
        agent_name: str,
        llm_config: GaiClientConfig,
        aggregated_client: Optional[McpAggregatedClient] = None,
        monologue: Optional[Monologue] = None,
    ):
        if not llm_config:
            raise ValueError("ChatAgent: llm_config is required.")

        self.agent_name = agent_name
        self.aggregated_client = aggregated_client
        self.llm_config = llm_config

        with AgenticStateMachine.StateMachineBuilder(
            """
            INIT --> IS_TOOL_CALL
            IS_TOOL_CALL --> CHAT: condition_false
            IS_TOOL_CALL --> TOOL_USE: condition_true
            
            CHAT --> IS_TERMINATE
            TOOL_USE --> IS_TERMINATE
            
            IS_TERMINATE --> IS_TOOL_CALL: condition_false
            IS_TERMINATE --> FINAL: condition_true
            
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
                        "module_path": "gai.asm.agents.tool_use_agent",
                        "class_name": "AnthropicChatState",
                        "title": "CHAT",
                        "input_data": {
                            "llm_config": {
                                "type": "getter",
                                "dependency": "get_llm_config",
                            },
                            "mcp_client": {
                                "type": "getter",
                                "dependency": "get_mcp_client",
                            },
                        },
                        "output_data": ["streamer", "get_assistant_message"],
                    },
                    "TOOL_USE": {
                        "module_path": "gai.asm.agents.tool_use_agent",
                        "class_name": "AnthropicToolUseState",
                        "title": "TOOL_USE",
                        "input_data": {
                            "llm_config": {
                                "type": "getter",
                                "dependency": "get_llm_config",
                            },
                            "mcp_client": {
                                "type": "getter",
                                "dependency": "get_mcp_client",
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
                    "IS_TERMINATE": {
                        "module_path": "gai.asm.states",
                        "class_name": "PurePredicateState",
                        "title": "IS_TERMINATE",
                        "predicate": "is_terminate",
                        "output_data": ["predicate_result"],
                        "conditions": ["condition_true", "condition_false"],
                    },
                    "FINAL": {
                        "output_data": ["monologue", "get_assistant_message"],
                    },
                },
                agent_name=self.agent_name,
                get_llm_config=lambda state: self.llm_config.model_dump(),
                get_mcp_client=lambda state: self.aggregated_client
                or McpAggregatedClient([]),
                monologue=monologue,
                has_message=self.has_message,
                is_tool_call=self.is_tool_call,
                is_terminate=self.is_terminate,
            )

    @property
    def monologue(self):
        """
        Returns the monologue associated with the agent.
        """
        return self.fsm.monologue

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
        result = False
        if (
            last_message
            and last_message.body.role == "assistant"
            and isinstance(last_message.body.content, list)
        ):
            if any(
                item.get("type") == "tool_use" for item in last_message.body.content
            ):
                result = True

        state.machine.state_bag["is_tool_call_result"] = result
        return result

    def is_terminate(self, state):
        result = bool(
            state.machine.user_message
            and state.machine.user_message.lower() == "terminate"
        )
        state.machine.state_bag["is_terminate_result"] = result
        return result

    async def _init_async(self):
        self.fsm.monologue.reset()
        self.fsm.restart()
        current_state = self.fsm.state
        await self.fsm.run_async()
        logger.info(f"Final state: {current_state} --> {self.fsm.state}")

    async def _run_async(
        self, user_message: Optional[str] = None, recap: Optional[str] = None
    ) -> AsyncGenerator[str, None]:
        """
        If there is no user_message, then this is a regular call
        to continue with tool_use.

        If there is a user_message, then this user_message is
        a response to the llm interrupted flow by the tool 'user_input'.
        """

        if (
            self.fsm.state_bag.get("is_user_input") == True
            and (self.fsm.state == "IS_TOOL_CALL")
            and user_message is None
        ):
            raise PendingUserInputError("ToolUseAgent._run_async: pending user input")

        current_state = self.fsm.state

        if recap:
            user_message = f"""
            {user_message}

            Here is a recap of the conversation. Note that the recap may include agents other than yourself.
            Do not confuse your identity and do not mention the recap. Just continue.
            {recap}
            """
        self.fsm.user_message = user_message

        await self.fsm.run_async()
        logger.info(f"Final state: {current_state} --> {self.fsm.state}")

        async def streamer():
            if (
                self.fsm.state_bag.get("is_user_input") == True
                and self.fsm.state == "TOOL_USE"
            ):
                self.fsm.state_bag["is_user_input"] = False

            if self.fsm.state_bag.get("streamer"):
                async for chunk in self.fsm.state_bag["streamer"]:
                    if chunk:
                        # if isinstance(chunk, str):
                        yield (chunk)
            # If the streamer is None, then reading from streamer will yield nothing.

        return streamer()

    async def start(self, user_message: str, recap: Optional[str] = None):
        """
        First call always require a user_message
        Call always ends with "IS_TERMINATE"
        """

        # INIT -> IS_TOOL_CALL
        await self._init_async()

        # IS_TOOL_CALL -> CHAT
        resp = await self._run_async(user_message=user_message, recap=recap)
        async for chunk in resp:
            yield chunk

        # CHAT -> IS_TERMINATE
        await self._run_async()

    async def resume(
        self, user_message: Optional[str] = None
    ) -> AsyncGenerator[str, None]:
        """
        Subsequent calls does not require user_message if task is not completed.
        Call always ends with "IS_TERMINATE"
        """

        # If continuing from previous state
        if self.fsm.state == "IS_TERMINATE":
            await self._run_async()

        # Run until LLM call
        try:
            while self.fsm.state != "IS_TERMINATE":
                if self.fsm.state == "IS_TOOL_CALL":
                    resp = await self._run_async(user_message=user_message)
                    async for chunk in resp:
                        yield chunk
                else:
                    await self._run_async()
        except PendingUserInputError as e:
            self.fsm.state_bag["streamer"] = None
            logger.error(f"ToolUserAgent.resume: {e}")

            # Move to IS_TERMINATE state
            prev = self.fsm.state
            while self.fsm.state != "IS_TERMINATE":
                try:
                    await self._run_async()
                except Exception:
                    pass
                if prev == self.fsm.state:
                    raise Exception(
                        "tool_user_agent.resume: Error while handling PendingUserInputError. Cannot fast forward to IS_TERMINATE."
                    )

            raise
        except MissingUserMessageError:
            # Move to IS_TERMINATE state
            prev = self.fsm.state
            while self.fsm.state != "IS_TERMINATE":
                try:
                    await self._run_async()
                except Exception:
                    pass
                if prev == self.fsm.state:
                    raise Exception(
                        "tool_user_agent.resume: Error while handling MissingUserMessageError. Cannot fast forward to IS_TERMINATE."
                    )

            # This error is only raised when the chat state is run without a user message and
            # since this is a resume() operation, that means the agent has completed its task and expecting a new user message.
            # In this case, we will raise AutoResumeError to indicate that the agent is ready for a new user message.
            raise AutoResumeError(
                "ToolUseAgent.resume: Cannot resume() as agent has completed its task. Either resume(user_message) to update the task or start a new task with start(user_message)."
            ) from None

    def final_output(self):
        get_assistant_message = self.fsm.state_bag["get_assistant_message"]
        return get_assistant_message()

    def _undo(self):
        """
        Undo the last state and return to the previous state.
        This is useful for undoing the last tool call or user message.
        """
        self.fsm.undo()
        logger.info(f"Undo: current state: {self.fsm.state}")
        return self.fsm.state

    def undo(self):
        """
        Public method to undo the last state.
        """

        if self.fsm.state == "IS_TOOL_CALL":
            # If we are already in IS_TOOL_CALL state, then undo one step.
            self._undo()

        # Undo until the last IS_TOOL_CALL state
        while self.fsm.state != "IS_TOOL_CALL":
            self._undo()

        return self.fsm.state
