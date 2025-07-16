from ..base import StateBase
from gai.lib.logging import getLogger
from gai.llm.lib import LLMGeneratorRetryPolicy
from gai.llm.openai import AsyncOpenAI
from gai.mcp.client import McpAggregatedClient
from rich.console import Console

console = Console()

logger = getLogger(__name__)

"""
AnthropicToolUseState

This state is made up of 2 actions. The first is to make a tool call and the second is the marshall the result from the
tool call and send it to the LLM for response.
"""


class AnthropicToolUseState(StateBase):
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

        self.machine.monologue.add_user_message(state=self, content=tool_results)

        async def streamer():
            nonlocal assistant_message

            async def stream_with_retry():
                response = await llm_client.chat.completions.create(
                    model=llm_model,
                    messages=self.machine.monologue.list_chat_messages(),
                    tools=tools,
                    stream=True,
                )

                async for chunk in response:
                    if chunk:
                        try:
                            chunk = chunk.extract()
                        except Exception as e:
                            logger.warning(
                                f"AnthropicToolUseState.streamer: chunk extract error={str(e)}"
                            )
                        yield chunk  # Just yield everything, control flow handled outside

            # Retry the entire streaming operation
            retry_policy = LLMGeneratorRetryPolicy(self.machine)
            has_text = False
            async for chunk in retry_policy.run(stream_with_retry):
                # The LLM client will return either of the following results:

                ##  * a stream of strings followed by a tool call. This means the response will be
                ##    streamed to the user and AthropicToolUseState will use a tool.
                ##    The session will continue.

                ##  - a tool call only. This means there is nothing to stream to the user, and
                ##    AnthropicToolUseState will silently use a tool.
                ##    The session will continue.

                ##  - a stream of strings only. This means the response will be streamed to the user
                ##    and AnthropicToolUseState will not use a tool.
                ##    This signifies the session has ended.

                if chunk:
                    if isinstance(chunk, str):
                        has_text = True
                        yield chunk
                    else:
                        if not has_text:
                            # That means the LLM has no text to stream but has content returned.
                            # So we stream "thinking..." instead to show its still working.
                            yield "Thinking...\n"
                        else:
                            yield "\n"

                        self.machine.monologue.add_assistant_message(
                            state=self, content=chunk
                        )
                        # Need to update the stale history due to delayed output
                        self.machine.state_history[-1]["output"]["monologue"] = (
                            self.machine.monologue.copy()
                        )

                        if isinstance(chunk, list) and chunk:
                            if isinstance(chunk[0], dict) and "text" in chunk[0]:
                                self.machine.state_bag["get_assistant_message"] = (
                                    lambda: chunk[0]["text"]
                                )
                        else:
                            self.machine.state_bag["get_assistant_message"] = (
                                lambda: chunk.copy()
                            )
                        yield chunk

                        # Exit after receiving first non-str token
                        return  # This will now work correctly

        self.machine.state_bag["streamer"] = streamer()
