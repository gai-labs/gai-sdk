"""
Synchronous patch module for OpenAI client.
Uses common functions from patch_common for shared functionality.
"""

from typing import Union, Optional
from gai.lib.config import GaiClientConfig
from gai.lib.logging import getLogger

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
from .attach_extractor import attach_extractor

logger = getLogger(__name__)


# openai_create(): This function calls the original unpatched chat.completions.create() function.

def openai_create(patched_client, **kwargs):
    stream = kwargs.get("stream", False)
    response = patched_client.chat.completions.original_openai_create(**kwargs)
    response = attach_extractor(response, stream)
    return response


# ollama_create(): This function calls the ollama chat() function.

def ollama_create(client_config, **kwargs):
    from ollama import chat
    
    mapped_kwargs = map_openai_to_ollama_params(client_config, **kwargs)
    response = chat(**mapped_kwargs)
    
    # Format ollama output to match openai output
    stream = mapped_kwargs["stream"]
    tools = mapped_kwargs["tools"]
    
    from .response.ollama.completions_factory import CompletionsFactory
    factory = CompletionsFactory()
    if stream and not tools:
        response = factory.chunk.build_stream(response)
        response = attach_extractor(response, stream)  
        response = (chunk for chunk in response)
    else:
        if tools:
            response = factory.message.build_toolcall(response)
        else:
            response = factory.message.build_content(response)
        response = attach_extractor(response, stream)
    return response


# gai_create(): This function calls the gai ChatClient() function.

def gai_create(client_config, **kwargs):
    from gai.chat.client import ChatClient
    
    mapped_kwargs = map_openai_to_gai_params(**kwargs)
    chat_client = ChatClient(client_config)
    response = chat_client.chat(**mapped_kwargs)
    return response


def anthropic_create(client_config, **kwargs):
    """
    This function calls the Claude client.
    """
    import anthropic
    
    final_kwargs = map_openai_to_anthropic_params(client_config, **kwargs)

    # Call anthropic API
    response = None
    try:
        client = anthropic.Anthropic()
        response = client.messages.create(**final_kwargs)
    except Exception as e:
        error_message = f"patch.anthropic_create: Error while calling anthropic API: {e}"
        logger.error(error_message)
        raise Exception(error_message)

    # Format anthropic output to match openai output
    try:
        from .response.anthropic.completions_factory import CompletionsFactory
        factory = CompletionsFactory()
        stream = isinstance(response, anthropic.Stream)
        tools = final_kwargs.get("tools", None)
        
        if stream:
            if not tools:
                response = factory.chunk.build_stream(response)
                response = attach_extractor(response, stream)
            else:
                response = factory.chunk.build_tool_stream(response)
                response = attach_extractor(response, stream)  
                response = (chunk for chunk in response)            
        else:
            if not tools:
                response = factory.message.build_content(response)
                response = attach_extractor(response, stream)
            else:
                response = factory.message.build_toolcall(response)
                response = attach_extractor(response, stream)
        return response
        
    except Exception as e:
        error_message = f"patch.anthropic_create: Error while formatting anthropic response: {e}"
        logger.error(error_message)
        raise Exception(error_message)


# openai_parse(): This function calls the original unpatched beta.chat.completions.parse() function.

def openai_parse(patched_client, **kwargs):
    response = patched_client.beta.chat.completions.original_openai_parse(**kwargs)
    response = attach_extractor(response, is_stream=False)
    return response


# ollama_parse(): This function calls the ollama chat() function.

def ollama_parse(client_config, response_format, **kwargs):
    from ollama import chat
    
    mapped_kwargs = map_openai_to_ollama_parse_params(client_config, response_format, **kwargs)
    response = chat(**mapped_kwargs)
        
    # Format ollama output to match openai output
    stream = mapped_kwargs["stream"]
    from .response.ollama.completions_factory import CompletionsFactory
    factory = CompletionsFactory()
    response = factory.message.build_content(response)
    response = attach_extractor(response, stream)
    return response


def anthropic_parse(client_config, response_format, **kwargs):
    import anthropic
    
    anthropic_tool, final_kwargs, messages = map_openai_to_anthropic_parse_params(response_format, **kwargs)
    
    client = anthropic.Anthropic()    
    response = client.messages.create(**final_kwargs)
    logger.debug(f"anthropic_parse: raw response: {response}")
    
    # Format anthropic output to match openai output
    from .response.anthropic.completions_factory import CompletionsFactory
    factory = CompletionsFactory()
    response = factory.message.build_toolcall(response)
    response = attach_extractor(response, False)
    return response


# gai_parse(): This function calls the gai ChatClient() function.

def gai_parse(client_config, response_format, **kwargs):
    from gai.chat.client import ChatClient
    
    mapped_kwargs = map_openai_to_gai_parse_params(response_format, **kwargs)
    chat_client = ChatClient(client_config)
    response = chat_client.chat(**mapped_kwargs)
    return response


def patch_chatcompletions(openai_client, file_path: str = None, client_config: Optional[Union[GaiClientConfig|dict]] = None):
    """
    Patch an OpenAI client to support multiple LLM backends.
    
    Parameters:
        openai_client: The OpenAI client to patch
        file_path: Path to configuration file
        client_config: Direct configuration (dict or GaiClientConfig)
        
    Returns:
        The patched OpenAI client
    """
    
    # Add get_client_config() function to the client
    openai_client.get_client_config = create_get_client_config_function(client_config, file_path)
    
    # Add LLM Client specific functions
    openai_client.openai_create = openai_create
    openai_client.ollama_create = ollama_create
    openai_client.anthropic_create = anthropic_create
    openai_client.gai_create = gai_create
    
    openai_client.openai_parse = openai_parse
    openai_client.ollama_parse = ollama_parse
    openai_client.gai_parse = gai_parse
    openai_client.anthropic_parse = anthropic_parse
    
    # Add routing functions
    def patched_create(**kwargs):
        nonlocal openai_client
        model = kwargs.get("model")
        client_config = openai_client.get_client_config(model)
        client_type = client_config.client_type

        if client_type == "openai":
            return openai_client.openai_create(openai_client, **kwargs)
        if client_type == "ollama":
            return openai_client.ollama_create(client_config, **kwargs)
        if client_type == "gai":
            return openai_client.gai_create(client_config, **kwargs)
        if client_type == "anthropic":
            return openai_client.anthropic_create(client_config, **kwargs)
        
        error_message = f"patched_create: Invalid client type: {client_type}"
        logger.error(error_message)
        raise Exception(error_message)

    def patched_parse(**kwargs):
        nonlocal openai_client
        model = kwargs.get("model")
        client_config = openai_client.get_client_config(model)
        client_type = client_config.client_type

        if client_type == "openai":
            return openai_client.openai_parse(openai_client, **kwargs)
        if client_type == "ollama":
            return openai_client.ollama_parse(client_config, **kwargs)
        if client_type == "gai":
            return openai_client.gai_parse(client_config, **kwargs)
        if client_type == "anthropic":
            return openai_client.anthropic_parse(client_config, **kwargs)
        
        error_message = f"patched_parse: Invalid client type: {client_type}"
        logger.error(error_message)
        raise Exception(error_message)
        
    # Apply patches with validation
    validate_patch_state(openai_client, "chat.completions", "is_patched", "patched_create")
    apply_patch(openai_client, "chat.completions", "create", "original_openai_create", patched_create, "is_patched")

    validate_patch_state(openai_client, "beta.chat.completions", "is_patched", "patched_parse")
    apply_patch(openai_client, "beta.chat.completions", "parse", "original_openai_parse", patched_parse, "is_patched")
        
    return openai_client