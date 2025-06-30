from ..base import StateBase
from gai.lib.logging import getLogger
from gai.llm.lib import LLMGeneratorRetryPolicy
from gai.llm.openai import AsyncOpenAI

logger = getLogger(__name__)

"""
AnthropicChatState

Make a chat call to Claude models.

"""


class AnthropicChatState(StateBase):
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

        # If user_message is missing, the machine should transition into AnthropicToolUseState
        # directly instead of here.

        if not self.input.get("user_message", None):
            raise Exception("AnthropicToolCallState: user_message is missing.")

        # Get llm client
        llm_config = self.input["llm_config"]
        llm_client = AsyncOpenAI(llm_config)

        # Get mcp client
        mcp_client = self.input.get("mcp_client")
        tools = []
        if mcp_client:
            tools = await mcp_client.list_tools()

        # Get model
        llm_model = llm_config["model"]

        last_tool_calls = []
        if self.machine.monologue.list_messages():
            last_tool_calls = self.machine.monologue.get_last_toolcalls()

        async def stream_nothing():
            # This will yield nothing, effectively ending the state
            yield

        if (
            last_tool_calls
            and any(result["tool_name"] == "user_input" for result in last_tool_calls)
            and self.machine.user_message
        ):
            # Case 1: LLM interrupt flow.
            # LLM request input from user using "user_input" and user_message is provided.
            # Stream nothing and forward to the "tool_use" state for processing.
            self.machine.state_bag["streamer"] = stream_nothing()
            return  # Exit the state early

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
                    # streaming continues
                    yield chunk

                else:
                    # streaming ended when a non-str chunk is received.
                    # The non-str chunk may contain a tool_call
                    # Note: This doesn't mean the LLM task is completed. It just means there is nothing else to stream.

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

                    # Exit
                    return

        self.machine.state_bag["streamer"] = streamer()
