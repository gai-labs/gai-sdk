import asyncio
from typing import Optional, Union
from gai.lib.config import GaiClientConfig, config_helper
from gai.dialogue import DialogueBus
from gai.dialogue.nodes import AsyncAgentNode, AsyncUserNode
from gai.llm.openai import AsyncOpenAI as AsyncOpenai


class AsyncOpenAI:
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
        self._initialized = False

        # Initialize the underlying client
        self.client = AsyncOpenai(**kwargs, client_config=self.client_config)

        # Set up dialogue handling if provided
        if dialogue:
            self._setup_dialogue()

    def _setup_dialogue(self):
        """Set up dialogue bus integration"""
        original_create = self.client.chat.completions.create

        async def handle_send_cb(message):
            """
            This handler must make LLM call
            """
            recap = self.dialogue.extract_recap()
            prompt = f"{self.name}, this is a short recap of your conversation so far <recap>{recap}</recap>.\nRefer to this recap to understand the background of the conversation. You will continue from where you left off as {self.name}. {message.body.content}"
            messages = [{"role": "user", "content": prompt}]
            response = await original_create(
                model=self.client_config.model, messages=messages, stream=True
            )
            return response

        self.agent_node = AsyncAgentNode(name=self.name, handle_send_cb=handle_send_cb)

        # Create a queue for chunks
        self.chunk_queue = asyncio.Queue()

        async def handle_reply_cb(message):
            await self.chunk_queue.put(message.body.chunk)

        self.user_node = AsyncUserNode(handle_reply_cb=handle_reply_cb)

        async def patch_chat(**kwargs):
            # Initialize async components on first call
            if not self._initialized:
                await self.agent_node.subscribe(dialogue=self.dialogue)
                await self.user_node.subscribe(dialogue=self.dialogue)
                self._initialized = True

            messages = kwargs.pop("messages")

            # Send the message
            await self.user_node.chat(
                message=messages[-1]["content"], recipient=self.name
            )

            # Create async generator that yields until <eom>
            async def chunk_generator():
                while True:
                    chunk = await self.chunk_queue.get()
                    if chunk == "<eom>":
                        break
                    yield chunk

            return chunk_generator()

        # Patch the client method
        self.client.chat.completions.create = patch_chat

    def __getattr__(self, name):
        """Delegate attribute access to the underlying client"""
        return getattr(self.client, name)


# async def AsyncOpenAI(
#     name: Optional[str] = "Assistant",
#     client_config: Optional[Union[GaiClientConfig, dict, str]] = None,
#     dialogue: Optional[DialogueBus] = None,
#     **kwargs,
# ):
#     client_config = config_helper.get_client_config(client_config)
#     client = AsyncOpenai(**kwargs, client_config=client_config)

#     if dialogue:
#         original_create = client.chat.completions.create

#         async def handle_send_cb(message):
#             """
#             This handler must make LLM call
#             """

#             recap = dialogue.extract_recap()
#             prompt = f"{name}, this is a short recap of your conversation so far <recap>{recap}</recap>.\nRefer to this recap to understand the background of the conversation. You will continue from where you left off as {name}. {message.body.content}"
#             messages = [{"role": "user", "content": prompt}]
#             response = await original_create(
#                 model=client_config.model, messages=messages, stream=True
#             )
#             return response

#         agent_node = AsyncAgentNode(name=name, handle_send_cb=handle_send_cb)
#         await agent_node.subscribe(dialogue=dialogue)

#         # Create a queue for chunks
#         chunk_queue = asyncio.Queue()

#         async def handle_reply_cb(message):
#             await chunk_queue.put(message.body.chunk)

#         user_node = AsyncUserNode(handle_reply_cb=handle_reply_cb)
#         await user_node.subscribe(dialogue=dialogue)

#         # async def patch_chat(**kwargs):
#         #     messages = kwargs.pop("messages")
#         #     await user_node.chat(message=messages[-1]["content"],recipient=name)

#         async def patch_chat(**kwargs):
#             messages = kwargs.pop("messages")

#             # Send the message
#             await user_node.chat(message=messages[-1]["content"], recipient=name)

#             # Create async generator that yields until <eom>
#             async def chunk_generator():
#                 while True:
#                     chunk = await chunk_queue.get()
#                     if chunk == "<eom>":
#                         break
#                     yield chunk

#             return chunk_generator()

#         client.chat.completions.create = patch_chat
#         client.agent_node = agent_node
#         client.user_node = user_node

#     return client
