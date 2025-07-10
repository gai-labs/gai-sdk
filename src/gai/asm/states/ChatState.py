from ..base import StateBase
from gai.lib.logging import getLogger
from gai.llm.lib import LLMGeneratorRetryPolicy
from gai.llm.openai import AsyncOpenAI

logger = getLogger(__name__)

"""
AnthropicChatState

Make a chat call to Claude models.

"""


class ChatState(StateBase):
    """
    state schema:
    {
        "CHAT": {
            "module_path": "gai.asm.states",
            "class_name": "ChatState",
            "title": "CHAT",
            "input_data": {
                "llm_config": {"type": "state_bag", "dependency": "llm_config"},
            },
            "output_data": ["streamer", "get_assistant_message"],
        }
    }
    """

    def __init__(self, machine):
        super().__init__(machine)

    async def run_async(self):
        # Get User Message

        if not self.input.get("user_message", None):
            raise Exception("ChatState: user_message is missing.")

        # Get llm client
        llm_config = self.input["llm_config"]
        llm_client = AsyncOpenAI(llm_config)

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
