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

    async def _use_tool(self):
        # Get last assistant message

        messages = self.machine.monologue.list_messages()
        last_message = messages[-1] if messages else None
        if not last_message or last_message.body.role != "assistant":
            raise ValueError("Last message is not from assistant or no messages found.")

        # First Pass: Create list of tool calls

        tool_calls = []
        if isinstance(last_message.body.content, list):
            for item in last_message.body.content:
                if isinstance(item, str):
                    continue

                if not isinstance(item, dict):
                    if item.type == "tool_use":
                        tool_calls.append(
                            {
                                "tool_use_id": item.id,
                                "tool_name": item.name,
                                "arguments": item.input,
                            }
                        )
                else:
                    if item["type"] == "tool_use":
                        tool_calls.append(
                            {
                                "tool_use_id": item["id"],
                                "tool_name": item["name"],
                                "arguments": item["input"],
                            }
                        )
        if not tool_calls:
            raise ValueError("Last message does not contain a tool_use content block.")

        # Second Pass: Make calls and get results

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

    async def run_async(self):
        # Get llm client
        llm_config = self.input["llm_config"]
        llm_client = AsyncOpenAI(llm_config)

        # Get mcp client
        mcp_client = self.input["mcp_client"]
        tools = await mcp_client.list_tools()

        # Get model
        llm_model = llm_config["model"]

        tool_results = await self._use_tool()

        assistant_message = ""

        async def streamer():
            nonlocal assistant_message

            self.machine.monologue.add_user_message(state=self, content=tool_results)

            async def stream_with_retry():
                response = await llm_client.chat.completions.create(
                    model=llm_model,
                    messages=self.machine.monologue.list_chat_messages(),
                    tools=tools,
                    stream=True,
                )

                async for chunk in response:
                    if chunk:
                        chunk = chunk.extract()
                        yield chunk  # Just yield everything, control flow handled outside

            # Retry the entire streaming operation
            retry_policy = LLMGeneratorRetryPolicy(self.machine)

            async for chunk in retry_policy.run(stream_with_retry):
                if isinstance(chunk, str):
                    yield chunk
                else:
                    self.machine.monologue.add_assistant_message(
                        state=self, content=chunk
                    )
                    # Need to update the stale history due to delayed output
                    self.machine.state_history[-1]["output"]["monologue"] = (
                        self.machine.monologue.copy()
                    )
                    self.machine.state_bag["get_assistant_message"] = (
                        lambda: chunk.copy()
                    )
                    yield chunk
                    # Exit after receiving first non-str token
                    return  # This will now work correctly

        self.machine.state_bag["streamer"] = streamer()
