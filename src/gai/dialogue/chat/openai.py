from typing import Optional, Union
from gai.lib.config import GaiClientConfig, config_helper
from gai.dialogue import DialogueBus
from gai.dialogue.nodes import AgentNode, UserNode
from gai.lib.utils import run_async_function
from gai.llm.openai import OpenAI as Openai


class OpenAI:
    def __init__(
        self,
        client_config: Optional[Union[GaiClientConfig, dict, str]] = None,
        dialogue: Optional[DialogueBus] = None,
        name: Optional[str] = "Assistant",
        **kwargs,
    ):
        self.name = name
        self.client_config = config_helper.get_client_config(client_config)
        self.dialogue = dialogue
        self.kwargs = kwargs

        # Initialize the underlying client
        self.client = Openai(**kwargs, client_config=self.client_config)

        # Set up dialogue handling if provided
        if dialogue:
            self._setup_dialogue()

    def _setup_dialogue(self):
        """Set up dialogue bus integration"""
        original_create = self.client.chat.completions.create

        def handle_send_cb(message):
            """
            This handler must make LLM call
            """
            recap = self.dialogue.extract_recap()
            prompt = f"{self.name}, this is a short recap of your conversation so far <recap>{recap}</recap>.\nRefer to this recap to understand the background of the conversation. You will continue from where you left off as {self.name}. {message.body.content}"
            messages = [{"role": "user", "content": prompt}]
            response = original_create(
                model=self.client_config.model, messages=messages, stream=True
            )
            return response

        self.agent_node = AgentNode(
            name=self.name, dialogue=self.dialogue, handle_send_cb=handle_send_cb
        )

        def handle_reply_cb(message):
            print(message)

        self.user_node = UserNode(
            dialogue=self.dialogue, handle_reply_cb=handle_reply_cb
        )

        def patch_chat(**kwargs):
            messages = kwargs.pop("messages")
            run_async_function(
                self.user_node.chat,
                message=messages[-1]["content"],
                recipient=self.name,
            )

        # Patch the client method
        self.client.chat.completions.create = patch_chat

    def __getattr__(self, name):
        """Delegate attribute access to the underlying client"""
        return getattr(self.client, name)
