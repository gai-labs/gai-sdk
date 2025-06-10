import asyncio
from typing import Optional,Union
from gai.lib.config import GaiClientConfig, config_helper
from gai.dialogue import DialogueBus
from gai.dialogue.nodes import AgentNode, UserNode, AsyncAgentNode, AsyncUserNode
from gai.lib.utils import run_async_function
from gai.chat.openai import OpenAI as Openai, AsyncOpenAI as AsyncOpenai

def OpenAI(
    name:Optional[str]="Assistant",
    client_config:Optional[Union[GaiClientConfig,dict,str]]=None,
    dialogue:Optional[DialogueBus]=None,
    **kwargs):


    client_config = config_helper.get_client_config(client_config)
    client = Openai(**kwargs, client_config=client_config)

    if dialogue:
        
        original_create = client.chat.completions.create

        def handle_send_cb(message):
            """
            This handler must make LLM call
            """
            
            recap = dialogue.extract_recap()
            prompt = f"{name}, this is a short recap of your conversation so far <recap>{recap}</recap>.\nRefer to this recap to understand the background of the conversation. You will continue from where you left off as {name}. {message.body.content}"
            messages = [
                {"role":"user","content":prompt}
            ]
            response = original_create(
                model=client_config.model, 
                messages=messages,
                stream=True
            )
            return response            
        
        agent_node = AgentNode(name=name,dialogue=dialogue,handle_send_cb=handle_send_cb)
        
        def handle_reply_cb(message):
            print(message)
        
        user_node = UserNode(dialogue=dialogue, handle_reply_cb=handle_reply_cb)
        
        def patch_chat(**kwargs):
            messages = kwargs.pop("messages")
            run_async_function( user_node.chat,message=messages[-1]["content"],recipient=name)
            
        client.chat.completions.create=patch_chat
        client.agent_node=agent_node
        client.user_node=user_node

    return client

async def AsyncOpenAI(
    name:Optional[str]="Assistant",
    client_config:Optional[Union[GaiClientConfig,dict,str]]=None,
    dialogue:Optional[DialogueBus]=None,
    **kwargs):

    client_config = config_helper.get_client_config(client_config)
    client = AsyncOpenai(**kwargs, client_config=client_config)

    if dialogue:
        
        original_create = client.chat.completions.create

        async def handle_send_cb(message):
            """
            This handler must make LLM call
            """
            
            recap = dialogue.extract_recap()
            prompt = f"{name}, this is a short recap of your conversation so far <recap>{recap}</recap>.\nRefer to this recap to understand the background of the conversation. You will continue from where you left off as {name}. {message.body.content}"
            messages = [
                {"role":"user","content":prompt}
            ]
            response = await original_create(
                model=client_config.model, 
                messages=messages,
                stream=True
            )        
            return response            
        
        agent_node = AsyncAgentNode(name=name,handle_send_cb=handle_send_cb)
        await agent_node.subscribe(dialogue=dialogue)
        
        # Create a queue for chunks
        chunk_queue = asyncio.Queue()
        
        async def handle_reply_cb(message):
            await chunk_queue.put(message.body.chunk)
        
        user_node = AsyncUserNode(handle_reply_cb=handle_reply_cb)
        await user_node.subscribe(dialogue=dialogue)
        
        # async def patch_chat(**kwargs):
        #     messages = kwargs.pop("messages")
        #     await user_node.chat(message=messages[-1]["content"],recipient=name)
            
        async def patch_chat(**kwargs):
            messages = kwargs.pop("messages")
            
            # Send the message
            await user_node.chat(message=messages[-1]["content"], recipient=name)
            
            # Create async generator that yields until <eom>
            async def chunk_generator():
                while True:
                    chunk = await chunk_queue.get()
                    if chunk == "<eom>":
                        break
                    yield chunk
            
            return chunk_generator()

        client.chat.completions.create = patch_chat
        client.agent_node = agent_node
        client.user_node = user_node

    return client
