from ..base import StateBase
from gai.lib.logging import getLogger
from gai.llm.lib import LLMGeneratorRetryPolicy
from gai.messages import message_helper
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

        assistant_message = ""
        self.machine.monologue.add_user_message(
            state=self, content=system_message)
        messages = self.machine.monologue.list_chat_messages()

        async def streamer():
            nonlocal assistant_message

            async def stream_with_retry():

                response = await llm_client.chat.completions.create(
                    model=llm_model,
                    messages=messages,
                    tools=tools,
                    stream=True,
                )

                async for chunk in response:
                    if chunk:
                        try:
                            chunk = chunk.extract()
                        except Exception as e:
                            logger.warning(
                                f"AnthropicChatState.streamer: chunk extract error={str(e)}"
                            )
                        yield chunk  # Just yield everything, control flow handled outside

            # Retry the entire streaming operation
            retry_policy = LLMGeneratorRetryPolicy(self.machine)
            has_text = False
            if not messages:
                raise ValueError(
                    "AnthropicChatState: Cannot pass empty messages to LLM. Find out why messages are empty."
                )
            async for chunk in retry_policy.run(stream_with_retry):
                # The LLM client will return either of the following results:

                # * a stream of strings followed by a tool call. This means the response will be
                # streamed to the user and AthropicToolUseState will use a tool.
                # The session will continue.

                # - a tool call only. This means there is nothing to stream to the user, and
                # AnthropicToolUseState will silently use a tool.
                # The session will continue.

                # - a stream of strings only. This means the response will be streamed to the user
                # and AnthropicToolUseState will not use a tool.
                # This signifies the session has ended.

                if isinstance(chunk, str):
                    # streaming continues
                    has_text = True
                    yield chunk

                else:
                    if not has_text:
                        # That means the LLM has no text to stream but has content returned.
                        # So we stream "thinking..." instead to show its still working.
                        yield "Thinking...\n"
                    else:
                        yield "\n"

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
                    return

        self.machine.state_bag["streamer"] = streamer()
