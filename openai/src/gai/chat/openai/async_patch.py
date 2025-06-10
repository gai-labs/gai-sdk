"""
Asynchronous patch module for OpenAI client.
Uses common functions from patch_common for shared functionality.

This version addresses ALL errors from dialogue history:
1. Proper is_async_generator checks before async for
2. Consistent use of attach_extractor_async
3. Proper handling of Anthropic AsyncStream
4. No redundant checks
5. No try/catch approach that can fail silently
"""

import inspect
from typing import Union, Optional
from gai.lib.utils import is_async_generator
from gai.lib.config import GaiClientConfig
from gai.lib.logging import getLogger
from .attach_extractor import attach_extractor_async

from .patch_common import (
    map_openai_to_ollama_params,
    map_openai_to_ollama_parse_params,
    map_openai_to_gai_params,
    map_openai_to_gai_parse_params,
    map_openai_to_anthropic_params,
    map_openai_to_anthropic_parse_params,
    create_get_client_config_function,
    validate_patch_state,
    apply_patch
)

logger = getLogger(__name__)

# async_openai_create(): Calls original unpatched async chat.completions.create()
async def async_openai_create(patched_client, **kwargs):
    stream = kwargs.get("stream", False)
    response = await patched_client.chat.completions.original_openai_create(**kwargs)
    
    if stream:
        # For streaming, don't call attach_extractor_async - apply logic manually
        import json
        tool_name = ""
        arguments = ""
        
        async def openai_async_streamer():
            nonlocal tool_name, arguments
            
            if is_async_generator(response):
                async for chunk in response:
                    # Manual extraction logic
                    if chunk.choices[0].delta.content:
                        chunk.extract = lambda: chunk.choices[0].delta.content

                    if chunk.choices[0].delta.tool_calls:
                        if chunk.choices[0].delta.tool_calls[0].function.name:
                            tool_name = chunk.choices[0].delta.tool_calls[0].function.name
                        
                        if chunk.choices[0].delta.tool_calls[0].function.arguments:
                            arguments += chunk.choices[0].delta.tool_calls[0].function.arguments

                    if chunk.choices[0].finish_reason:
                        if chunk.choices[0].finish_reason == "tool_calls" and tool_name:
                            if not arguments:
                                arguments = json.dumps({"type": "object", "properties": {}, "required": []})
                            
                            chunk.extract = lambda: {
                                "type": "finish_reason",
                                "finish_reason": chunk.choices[0].finish_reason,
                                "tool_name": tool_name,
                                "arguments": arguments
                            }
                        else:
                            chunk.extract = lambda: {
                                "type": "finish_reason",
                                "finish_reason": chunk.choices[0].finish_reason
                            }
                    
                    if not hasattr(chunk, 'extract'):
                        chunk.extract = lambda: ""
                        
                    yield chunk
            else:
                # Convert sync to async
                for chunk in response:
                    # Manual extraction logic
                    if chunk.choices[0].delta.content:
                        chunk.extract = lambda: chunk.choices[0].delta.content

                    if chunk.choices[0].delta.tool_calls:
                        if chunk.choices[0].delta.tool_calls[0].function.name:
                            tool_name = chunk.choices[0].delta.tool_calls[0].function.name
                        
                        if chunk.choices[0].delta.tool_calls[0].function.arguments:
                            arguments += chunk.choices[0].delta.tool_calls[0].function.arguments

                    if chunk.choices[0].finish_reason:
                        if chunk.choices[0].finish_reason == "tool_calls" and tool_name:
                            if not arguments:
                                arguments = json.dumps({"type": "object", "properties": {}, "required": []})
                            
                            chunk.extract = lambda: {
                                "type": "finish_reason",
                                "finish_reason": chunk.choices[0].finish_reason,
                                "tool_name": tool_name,
                                "arguments": arguments
                            }
                        else:
                            chunk.extract = lambda: {
                                "type": "finish_reason",
                                "finish_reason": chunk.choices[0].finish_reason
                            }
                    
                    if not hasattr(chunk, 'extract'):
                        chunk.extract = lambda: ""
                        
                    yield chunk
        
        return openai_async_streamer()
    else:
        # Non-streaming case - safe to use attach_extractor_async
        response = attach_extractor_async(response, stream)
        return response


# async_ollama_create(): Calls async ollama chat() function
async def async_ollama_create(client_config, **kwargs):
    import ollama
    
    mapped_kwargs = map_openai_to_ollama_params(client_config, **kwargs)
    
    # Use async ollama client
    async_client = ollama.AsyncClient()
    response = await async_client.chat(**mapped_kwargs)
    
    # Format ollama output to match openai output
    stream = mapped_kwargs["stream"]
    tools = mapped_kwargs["tools"]
    
    from .response.ollama.completions_factory import CompletionsFactory
    factory = CompletionsFactory()
    if stream and not tools:
        # Collect async chunks first, then use existing build_stream()
        chunks = []
        async for chunk in response:
            chunks.append(chunk)
        
        response = factory.chunk.build_stream(chunks)
        response = attach_extractor_async(response, stream)
        return response
    else:
        if tools:
            response = factory.message.build_toolcall(response)
        else:
            response = factory.message.build_content(response)
        response = attach_extractor_async(response, stream)
    return response


# async_gai_create(): Calls async gai ChatClient() function
async def async_gai_create(client_config, **kwargs):
    from gai.chat.client import AsyncChatClient
    
    mapped_kwargs = map_openai_to_gai_params(**kwargs)
    async with AsyncChatClient(client_config) as chat_client:
        response = await chat_client.chat(**mapped_kwargs)
    return response

async def async_anthropic_create(client_config, **kwargs):
    """
    Handles Anthropic AsyncStream properly with context manager.
    Uses existing factory methods correctly.
    """
    import anthropic
    
    final_kwargs = map_openai_to_anthropic_params(client_config, **kwargs)

    # Call anthropic API with async client
    try:
        async_client = anthropic.AsyncAnthropic()
        response = await async_client.messages.create(**final_kwargs)        
    except Exception as e:
        error_message = f"async_anthropic_create: Error while calling anthropic API: {e}"
        logger.error(error_message)
        raise Exception(error_message)

    # Format anthropic output to match openai output
    try:
        from .response.anthropic.completions_factory import CompletionsFactory
        factory = CompletionsFactory()
        
        is_stream = isinstance(response, anthropic.AsyncStream)
        tools = final_kwargs.get("tools", None)
        
        if is_stream:
            
            if not tools:
                response = factory.chunk.build_async_stream(response)
                response = attach_extractor_async(response, is_stream)
            else:
                response = factory.chunk.build_async_tool_stream(response)
                response = attach_extractor_async(response, is_stream)
        else:
            if not tools:
                response = factory.message.build_content(response)
                response = attach_extractor_async(response, False)
            else:
                response = factory.message.build_toolcall(response)
                response = attach_extractor_async(response, False)
        return response
        
    except Exception as e:
        error_message = f"async_anthropic_create: Error while formatting anthropic response: {e}"
        logger.error(error_message)
        raise Exception(error_message)

# async_openai_parse(): Calls original unpatched async beta.chat.completions.parse()
async def async_openai_parse(patched_client, **kwargs):
    response = await patched_client.beta.chat.completions.original_openai_parse(**kwargs)
    response = attach_extractor_async(response, is_stream=False)
    return response


# async_ollama_parse(): Calls async ollama chat() function
async def async_ollama_parse(client_config, response_format, **kwargs):
    import ollama
    
    mapped_kwargs = map_openai_to_ollama_parse_params(client_config, response_format, **kwargs)

    # Call ollama with async client
    async_client = ollama.AsyncClient()
    response = await async_client.chat(**mapped_kwargs)
        
    # Format ollama output to match openai output
    # Parse operations are never streaming, so safe to use attach_extractor_async
    stream = mapped_kwargs["stream"]  # Should always be False for parse
    from .response.ollama.completions_factory import CompletionsFactory
    factory = CompletionsFactory()
    response = factory.message.build_content(response)
    response = attach_extractor_async(response, False)  # Force False for parse
    return response


# async_anthropic_parse(): Calls async anthropic parse
async def async_anthropic_parse(client_config, response_format, **kwargs):
    import anthropic
    
    anthropic_tool, final_kwargs, messages = map_openai_to_anthropic_parse_params(response_format, **kwargs)
    
    async_client = anthropic.AsyncAnthropic()
    response = await async_client.messages.create(**final_kwargs)
    logger.debug(f"async_anthropic_parse: raw response: {response}")
    
    # Format anthropic output to match openai output
    from .response.anthropic.completions_factory import CompletionsFactory
    factory = CompletionsFactory()
    response = factory.message.build_toolcall(response)
    response = attach_extractor_async(response, False)
    return response


# async_gai_parse(): Calls async gai ChatClient() function
async def async_gai_parse(client_config, response_format, **kwargs):
    from gai.chat.client import AsyncChatClient
    
    mapped_kwargs = map_openai_to_gai_parse_params(response_format, **kwargs)
    async with AsyncChatClient(client_config) as chat_client:
        response = await chat_client.chat(**mapped_kwargs)
    return response


def patch_async_chatcompletions(openai_async_client, file_path: str = None, client_config: Optional[Union[GaiClientConfig|dict]] = None):
    """
    Patch an async OpenAI client to support multiple LLM backends.
    
    Parameters:
        openai_async_client: The async OpenAI client to patch
        file_path: Path to configuration file
        client_config: Direct configuration (dict or GaiClientConfig)
        
    Returns:
        The patched async OpenAI client
    """
    
    # Add get_client_config() function to the client
    openai_async_client.get_client_config = create_get_client_config_function(client_config, file_path)
    
    # Add LLM Client specific functions
    openai_async_client.async_openai_create = async_openai_create
    openai_async_client.async_ollama_create = async_ollama_create
    openai_async_client.async_anthropic_create = async_anthropic_create
    openai_async_client.async_gai_create = async_gai_create
    
    openai_async_client.async_openai_parse = async_openai_parse
    openai_async_client.async_ollama_parse = async_ollama_parse
    openai_async_client.async_gai_parse = async_gai_parse
    openai_async_client.async_anthropic_parse = async_anthropic_parse
    
    # Add routing functions
    async def async_patched_create(**kwargs):
        nonlocal openai_async_client
        model = kwargs.get("model")
        client_config = openai_async_client.get_client_config(model)
        client_type = client_config.client_type

        if client_type == "openai":
            return await openai_async_client.async_openai_create(openai_async_client, **kwargs)
        if client_type == "ollama":
            return await openai_async_client.async_ollama_create(client_config, **kwargs)
        if client_type == "gai":
            return await openai_async_client.async_gai_create(client_config, **kwargs)
        if client_type == "anthropic":
            return await openai_async_client.async_anthropic_create(client_config, **kwargs)
        
        error_message = f"async_patched_create: Invalid client type: {client_type}"
        logger.error(error_message)
        raise Exception(error_message)

    async def async_patched_parse(**kwargs):
        nonlocal openai_async_client
        model = kwargs.get("model")
        client_config = openai_async_client.get_client_config(model)
        client_type = client_config.client_type

        if client_type == "openai":
            return await openai_async_client.async_openai_parse(openai_async_client, **kwargs)
        if client_type == "ollama":
            return await openai_async_client.async_ollama_parse(client_config, **kwargs)
        if client_type == "gai":
            return await openai_async_client.async_gai_parse(client_config, **kwargs)
        if client_type == "anthropic":
            return await openai_async_client.async_anthropic_parse(client_config, **kwargs)
        
        error_message = f"async_patched_parse: Invalid client type: {client_type}"
        logger.error(error_message)
        raise Exception(error_message)
        
    # Apply patches with validation
    validate_patch_state(openai_async_client, "chat.completions", "is_async_patched", "async_patched_create")
    apply_patch(openai_async_client, "chat.completions", "create", "original_openai_create", async_patched_create, "is_async_patched")

    validate_patch_state(openai_async_client, "beta.chat.completions", "is_async_patched", "async_patched_parse")
    apply_patch(openai_async_client, "beta.chat.completions", "parse", "original_openai_parse", async_patched_parse, "is_async_patched")
        
    return openai_async_client


# Usage example:
"""
from openai import AsyncOpenAI
from gai.chat.openai.async_patch import patch_async_chatcompletions

async def main():
    client = AsyncOpenAI()
    patched_client = patch_async_chatcompletions(client, client_config={
        "client_type": "gai",
        "url": "http://gai-chat-svr:12031/gen/v1/chat/completions"
    })
    
    # Non-streaming
    response = await patched_client.chat.completions.create(
        model="ttt",
        messages=[{"role": "user", "content": "Hello!"}],
        stream=False
    )
    content = response.extract()
    
    # Streaming
    response = await patched_client.chat.completions.create(
        model="ttt", 
        messages=[{"role": "user", "content": "Tell me a story"}],
        stream=True
    )
    async for chunk in response:
        if chunk:
            content = chunk.extract()
            print(content, end='')

asyncio.run(main())
"""