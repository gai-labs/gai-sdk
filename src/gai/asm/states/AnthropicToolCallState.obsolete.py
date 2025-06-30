from ..base import StateBase
from gai.lib.logging import getLogger
from gai.llm.lib import LLMGeneratorRetryPolicy
from gai.llm.openai import AsyncOpenAI
from gai.mcp.client import McpAggregatedClient

logger = getLogger(__name__)

"""
AnthropicToolCallState

This state is to generating tool calls with Anthropic models only. Do not use with other models.

"""


class AnthropicToolCallState(StateBase):
    """
    state schema:
    {
        "TOOL_CALL": {
            "module_path": "gai.asm.states",
            "class_name": "AnthropicToolCallState",
            "title": "TOOL_CALL",
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

        # If user_message is missing, the machine should transition into AnthropicToolUseState
        # directly instead of here.

        if not self.input.get("user_message", None):
            raise Exception("AnthropicToolCallState: user_message is missing.")

        # Get llm client
        llm_config = self.input["llm_config"]
        llm_client = AsyncOpenAI(llm_config)

        # Get mcp client
        mcp_client = self.input["mcp_client"]
        tools = await mcp_client.list_tools()

        # Get model
        llm_model = llm_config["model"]

        assistant_message = ""

        async def streamer():
            nonlocal assistant_message

            async def stream_with_retry():
                user_message = self.machine.user_message
                self.machine.monologue.add_user_message(
                    state=self, content=user_message
                )
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
                #
                # The LLM will always return:

                ##  * a stream of strings followed by a tool call. This means the response will be
                ##    streamed to the user and AthropicToolUseState will use a tool.
                ##    The session will continue.
                ##    ContinueToolUseState will return True

                ##  - a tool call only. This means there is nothing to stream to the user, and
                ##    AnthropicToolUseState will silently use a tool.
                ##    The session will continue.
                ##    ContinueToolUseState will return True

                ##  - a stream of strings only. This means the response will be streamed to the user
                ##    and AnthropicToolUseState will not use a tool.
                ##    This signifies the session has ended.
                ##    ContinueToolUseState will return False

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
