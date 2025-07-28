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
                "recap": {"type": "state_bag", "dependency": "recap"},
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
            raise Exception("ChatState: user_message is missing.")

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

        assistant_message = ""

        self.machine.monologue.add_user_message(state=self, content=system_message)

        from gai.messages import message_helper

        messages = self.machine.monologue.list_messages()
        chat_messages = message_helper.convert_to_chat_messages(messages)
        chat_messages = message_helper.shrink_messages(chat_messages)

        async def streamer():
            nonlocal assistant_message
            logger.info(f"ChatState.run_async: inside streamer()")

            async def stream_with_retry():
                response = await llm_client.chat.completions.create(
                    model=llm_model,
                    messages=chat_messages,
                    stream=True,
                )

                async for chunk in response:
                    if chunk:
                        chunk = chunk.extract()
                        yield chunk  # Just yield everything, control flow handled outside

            # Retry the entire streaming operation
            retry_policy = LLMGeneratorRetryPolicy(self.machine)
            if not chat_messages:
                raise ValueError(
                    "AnthropicChatState: Cannot pass empty messages to LLM. Find out why messages are empty."
                )

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

                    # Exit
                    return

        self.machine.state_bag["streamer"] = streamer()
