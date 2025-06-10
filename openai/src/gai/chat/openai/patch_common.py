"""
Common functions shared between sync and async patch modules.
This module contains utility functions and configuration logic that are identical
between sync and async implementations.
"""

import os
import inspect
from pydantic import BaseModel
from typing import get_args, Union, Optional

from gai.lib.config import GaiClientConfig, config_helper
from gai.lib.logging import getLogger
from .types import ChatModel

logger = getLogger(__name__)

# Set default OpenAI API key if not present
api_key = os.environ.get("OPENAI_API_KEY", None)
if not api_key:
    os.environ["OPENAI_API_KEY"] = "please_set_your_openai_api_key"


def is_BaseModel(item):
    """
    Check if the given item is a subclass of BaseModel.
    This is used to validate response_format.

    Parameters:
        item: The item to check.

    Returns:
        bool: True if the item is a subclass of BaseModel, False otherwise.
    """
    return inspect.isclass(item) and issubclass(item, BaseModel)


def create_get_client_config_function(client_config: Optional[Union[GaiClientConfig|dict]], file_path: str):
    """
    Factory function that creates a get_client_config function with the provided config and file_path.
    This function is used by both sync and async patch modules.
    
    Parameters:
        client_config: The client configuration (dict or GaiClientConfig)
        file_path: Path to configuration file
        
    Returns:
        function: A get_client_config function bound to the provided config
    """
    def get_client_config(model: str):
        nonlocal client_config, file_path
        
        if client_config and file_path:
            raise ValueError(f"__init__: config and path cannot be provided at the same time")

        # If model is an openai model, return "openai"
        if model in get_args(ChatModel):
            return GaiClientConfig(client_type="openai", model=model)
        
        # If it is not an openai model, then check client_config
        # There are two ways to provide the client_config:
        # 1. Provide the client_config directly
        # 2. Provide the file_path to the client_config
        # But both cannot be provided at the same time.

        if client_config:
            if isinstance(client_config, dict):
                # Load default config and patch with provided config
                resolved_config = config_helper.get_client_config(client_config)
            elif isinstance(client_config, GaiClientConfig):
                resolved_config = client_config
            else:
                raise ValueError(f"__init__: Invalid config provided")
        else:
            # If no config is provided, load config from path
            resolved_config = config_helper.get_client_config(model, file_path=file_path)
            
        return resolved_config
    
    return get_client_config


def map_openai_to_ollama_params(client_config, **kwargs):
    """
    Map OpenAI parameters to Ollama parameters.
    
    Parameters:
        client_config: The client configuration
        **kwargs: OpenAI parameters
        
    Returns:
        dict: Mapped Ollama parameters
    """
    mapped_kwargs = {
        # Get actual model from config and not from model parameter
        "model": client_config.model,
        "messages": kwargs.get("messages", None),
        "options": {
            "temperature": kwargs.get("temperature", None),
            "top_k": kwargs.get("top_k", None),
            "top_p": kwargs.get("top_p", None),
            "num_predict": kwargs.get("max_tokens", None),
        },
        "stream": kwargs.get("stream", False),
        "tools": kwargs.get("tools", None),
    }
    
    # Change the default context length of 2048
    if client_config.extra and client_config.extra.get("num_ctx", None):
        mapped_kwargs["options"]["num_ctx"] = client_config.extra.get("num_ctx")
    
    if mapped_kwargs.get("tools"):
        mapped_kwargs["stream"] = False
    
    return mapped_kwargs


def map_openai_to_ollama_parse_params(client_config, response_format, **kwargs):
    """
    Map OpenAI parameters to Ollama parameters for parse operations.
    
    Parameters:
        client_config: The client configuration
        response_format: The response format (BaseModel or dict)
        **kwargs: OpenAI parameters
        
    Returns:
        dict: Mapped Ollama parameters
    """
    mapped_kwargs = {
        # Get actual model from config and not from model parameter
        "model": client_config.model,
        "messages": kwargs.get("messages", None),
        "options": {
            "temperature": 0,
            "num_predict": kwargs.get("max_tokens", None),
        },
        "stream": False,
    }

    # Change the default context length of 2048
    if client_config.extra and client_config.extra.get("num_ctx", None):
        mapped_kwargs["options"]["num_ctx"] = client_config.extra.get("num_ctx")

    # Convert pydantic BaseModel to json schema
    if is_BaseModel(response_format):
        schema = response_format.model_json_schema()
        mapped_kwargs["format"] = schema
    elif type(response_format) is dict:
        if response_format.get("json_schema"):
            mapped_kwargs["format"] = response_format["json_schema"]["schema"]
        else:
            mapped_kwargs["format"] = response_format
    else:
        raise Exception("response_format must be a dict or a pydantic BaseModel")

    return mapped_kwargs


def map_openai_to_gai_params(**kwargs):
    """
    Map OpenAI parameters to GAI parameters.
    
    Parameters:
        **kwargs: OpenAI parameters
        
    Returns:
        dict: Mapped GAI parameters
    """
    return {
        "model": kwargs.get("model", "ttt"),
        "messages": kwargs.get("messages", None),
        "stream": kwargs.get("stream", False),
        "max_tokens": kwargs.get("max_tokens", None),
        "temperature": kwargs.get("temperature", None),
        "top_p": kwargs.get("top_p", None),
        "top_k": kwargs.get("top_k", None),
        "tools": kwargs.get("tools", None),
        "tool_choice": kwargs.get("tool_choice", None),
        "stop": kwargs.get("stop", None),
        "timeout": kwargs.get("timeout", None),
    }


def map_openai_to_gai_parse_params(response_format, **kwargs):
    """
    Map OpenAI parameters to GAI parameters for parse operations.
    
    Parameters:
        response_format: The response format (BaseModel or dict)
        **kwargs: OpenAI parameters
        
    Returns:
        dict: Mapped GAI parameters
    """
    mapped_kwargs = {
        "model": kwargs.get("model", "ttt"),
        "messages": kwargs.get("messages", None),
        "stream": False,
        "max_tokens": kwargs.get("max_tokens", None),
        "timeout": kwargs.get("timeout", None),
    }
    
    if is_BaseModel(response_format):
        schema = response_format.model_json_schema()
        mapped_kwargs["json_schema"] = schema
    elif type(response_format) is dict:
        if response_format.get("json_schema"):
            mapped_kwargs["json_schema"] = response_format["json_schema"]["schema"]
        else:
            mapped_kwargs["json_schema"] = response_format
    else:
        raise Exception("response_format must be a dict or a pydantic BaseModel")

    return mapped_kwargs


def map_openai_to_anthropic_params(client_config, **kwargs):
    """
    Map OpenAI parameters to Anthropic parameters.
    
    Parameters:
        client_config: The client configuration
        **kwargs: OpenAI parameters
        
    Returns:
        dict: Mapped Anthropic parameters
    """
    config_model = client_config.model
    config_max_tokens = client_config.extra.get("max_tokens", 1000) if client_config.extra else 1000
    config_temperature = client_config.extra.get("temperature", None) if client_config.extra else None
    config_top_k = client_config.extra.get("top_k", None) if client_config.extra else None
    config_timeout = client_config.extra.get("timeout", None) if client_config.extra else None
    
    final_kwargs = {
        "model": kwargs.get("model", config_model),
        "max_tokens": kwargs.get("max_tokens", config_max_tokens),
        "messages": kwargs.get("messages", [{"role": "user", "content": ""}]),
        "stream": kwargs.get("stream", False)
    }
    
    if kwargs.get("temperature", config_temperature):
        final_kwargs["temperature"] = kwargs.get("temperature", config_temperature)

    if kwargs.get("top_k", config_top_k):
        final_kwargs["top_k"] = kwargs.get("top_k", config_top_k)
    
    if kwargs.get("timeout", config_timeout):
        final_kwargs["timeout"] = kwargs.get("timeout", config_timeout)

    if kwargs.get("tools", None):
        tools = []
        for tool in kwargs.get("tools", []):
            function = tool["function"]
            anthropic_tool = {
                "name": function["name"],
                "description": function["description"],
            }
            if function.get("parameters", None):
                anthropic_tool["input_schema"] = function["parameters"]
            else:
                anthropic_tool["input_schema"] = {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            tools.append(anthropic_tool)
        final_kwargs["tools"] = tools

    return final_kwargs


def map_openai_to_anthropic_parse_params(response_format, **kwargs):
    """
    Map OpenAI parameters to Anthropic parameters for parse operations.
    
    Parameters:
        response_format: The response format (BaseModel or dict)
        **kwargs: OpenAI parameters
        
    Returns:
        tuple: (anthropic_tool, final_kwargs, messages)
    """
    # Convert pydantic BaseModel to json schema
    if is_BaseModel(response_format):
        schema = response_format.model_json_schema()
    elif type(response_format) is dict:
        if response_format.get("json_schema"):
            schema = response_format["json_schema"]["schema"]
        else:
            schema = response_format
    else:
        raise Exception("response_format must be a dict or a pydantic BaseModel")

    # Convert from json_schema to anthropic tool format
    anthropic_tool = {
        "name": "structured_output",
        "description": "Structured output for the response",
        "input_schema": schema
    }
    
    # Hack the messages to tell claude to return result in json format.
    messages = kwargs.get("messages", [{"role": "user", "content": ""}])
    messages[-1]["content"] += "Return response in JSON format."
    
    # Map openai parameters to anthropic parameters
    final_kwargs = {
        "model": kwargs.get("model", "claude-opus-4-20250514"),
        "messages": messages,
        "max_tokens": kwargs.get("max_tokens", 1000),
        "stream": False,
        "tools": [anthropic_tool],
    }
    if kwargs.get("top_k", None):
        final_kwargs["top_k"] = kwargs.get("top_k")
    if kwargs.get("timeout", None):
        final_kwargs["timeout"] = kwargs.get("timeout")
    
    return anthropic_tool, final_kwargs, messages


def validate_patch_state(client, attribute_path: str, patch_flag: str, operation_name: str):
    """
    Validate that a client hasn't already been patched to prevent double-patching.
    
    Parameters:
        client: The client object to check
        attribute_path: Dot-separated path to the object containing the method (e.g., "chat.completions")
        patch_flag: The flag name to check (e.g., "is_patched")
        operation_name: Human-readable operation name for error messages
        
    Raises:
        Exception: If already patched
    """
    # Navigate to the target object
    target = client
    for attr in attribute_path.split('.'):
        target = getattr(target, attr)
    
    if hasattr(target, patch_flag):
        error_message = f"{operation_name}: Attempted to re-patch the client which is already patched."
        logger.error(error_message)
        raise Exception(error_message)


def apply_patch(client, target_object_path: str, method_name: str, original_attr: str, new_function, patch_flag: str):
    """
    Apply a patch to a client method, backing up the original.
    
    Parameters:
        client: The client object to patch
        target_object_path: Dot-separated path to the object containing the method (e.g., "chat.completions")
        method_name: Name of the method to patch (e.g., "create")
        original_attr: Name for the backup attribute (e.g., "original_openai_create")
        new_function: The new function to install
        patch_flag: The flag name to set (e.g., "is_patched")
    """
    # Navigate to the target object (the one that contains the method)
    target = client
    for attr in target_object_path.split('.'):
        target = getattr(target, attr)
    
    # Backup original method and install new function
    original_method = getattr(target, method_name)
    setattr(target, original_attr, original_method)
    setattr(target, method_name, new_function)
    setattr(target, patch_flag, True)