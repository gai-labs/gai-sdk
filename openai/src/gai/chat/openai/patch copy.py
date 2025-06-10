import os
import inspect
from pydantic import BaseModel
from typing import get_args,Union, Optional,Callable

api_key = os.environ.get("OPENAI_API_KEY",None)
if not api_key:
    os.environ["OPENAI_API_KEY"] = "please_set_your_openai_api_key"

from gai.lib.config import GaiClientConfig, config_helper
from gai.lib.logging import getLogger
logger = getLogger(__name__)

from .types import ChatModel
from .attach_extractor import attach_extractor

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

# openai_create(): This function calls the original unpatched chat.completions.create() function.

def openai_create(patched_client, **kwargs):
    stream=kwargs.get("stream",False)
    response = patched_client.chat.completions.original_openai_create(**kwargs)
    response = attach_extractor(response,stream)
    return response

# ollama_create(): This function calls the ollama chat() function.

def ollama_create(client_config, **kwargs):
    from ollama import chat
    
    # Map openai parameters to ollama parameters
    kwargs={
        # Get actual model from config and not from model parameter
        "model": client_config.model,
        "messages": kwargs.get("messages", None),
        "options": {
            "temperature": kwargs.get("temperature", None),
            "top_k": kwargs.get("top_k", None),
            "top_p": kwargs.get("top_p", None),
            "num_predict" : kwargs.get("max_tokens", None),
        },
        "stream": kwargs.get("stream", False),
        "tools": kwargs.get("tools", None),
    }
    
    # Change the default context length of 2048
    if client_config.extra and client_config.extra.get("num_ctx", None):
        kwargs["options"]["num_ctx"] = client_config.extra.get("num_ctx")
    
    if kwargs.get("tools"):
        kwargs["stream"] = False
    response = chat(**kwargs)
    
    # Format ollama output to match openai output
    stream = kwargs["stream"]
    tools = kwargs["tools"]
    
    from .response.ollama.completions_factory import CompletionsFactory
    factory = CompletionsFactory()
    if stream and not tools:
        response = factory.chunk.build_stream(response)
        response = attach_extractor(response,stream)  
        response = (chunk for chunk in response)
    else:
        if tools:
            response = factory.message.build_toolcall(response)
        else:
            response = factory.message.build_content(response)
        response = attach_extractor(response,stream)
    return response


# gai_create(): This function calls the gai ChatClient() function.

def gai_create(client_config, **kwargs):
    from gai.chat.client import ChatClient    
    
    # Map openai parameters to gai parameters
    kwargs = {
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

    chat_client = ChatClient(client_config)
    response = chat_client.chat(**kwargs)
    return response

def anthropic_create(client_config, **kwargs):
    """
    This function is a placeholder for the Claude client.
    It is not implemented yet, but it should be implemented to call the Claude API.
    """
    import anthropic
    
    config_model = client_config.model
    config_max_tokens = client_config.extra.get("max_tokens",1000) if client_config.extra else 1000
    config_temperature = client_config.extra.get("temperature",None)if client_config.extra else None
    config_top_k = client_config.extra.get("top_k",None)if client_config.extra else None
    config_timeout = client_config.extra.get("timeout",None)if client_config.extra else None
    
    final_kwargs = {
        "model": kwargs.get("model", config_model),
        "max_tokens": kwargs.get("max_tokens", config_max_tokens),
        "messages": kwargs.get("messages", [{"role":"user","content": ""}]),
        "stream": kwargs.get("stream", False)
    }
    if kwargs.get("temperature", config_temperature):
        final_kwargs["temperature"] = kwargs.get("temperature", config_temperature)

    if kwargs.get("top_k",config_top_k):
        final_kwargs["top_k"] = kwargs.get("top_k", config_top_k)
    
    if kwargs.get("timeout",config_timeout):
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
                response = attach_extractor(response,stream)
            else:
                response = factory.chunk.build_tool_stream(response)
                response = attach_extractor(response,stream)  
                response = (chunk for chunk in response)            
        else:
            if not tools:
                response = factory.message.build_content(response)
                response = attach_extractor(response,stream)
            else:
                response = factory.message.build_toolcall(response)
                response = attach_extractor(response,stream)
        return response
        
    except Exception as e:
        error_message = f"patch.anthropic_create: Error while formatting anthropic response: {e}"
        logger.error(error_message)
        raise Exception(error_message)

# openai_parse(): This function calls the original unpatched beta.chat.completions.parse() function.

def openai_parse(patched_client, **kwargs):
    response = patched_client.beta.chat.completions.original_openai_parse(**kwargs)
    response = attach_extractor(response,is_stream=False)
    return response

# ollama_parse(): This function calls the ollama chat() function.

def ollama_parse(client_config,response_format, **kwargs):
    from ollama import chat
    
    # Map openai parameters to ollama parameters
    kwargs={
        # Get actual model from config and not from model parameter
        "model": client_config.model,
        "messages": kwargs.get("messages", None),
        "options": {
            "temperature": 0,
            "num_predict" : kwargs.get("max_tokens", None),
        },
        "stream": False,
    }

    # We cannot use num_ctx using openai's parameter so in order to change the default context length of 2048,
    # we need to use the extra parameter in the Gai's client_config.
    if client_config.extra and client_config.extra.get("num_ctx", None):
        kwargs["options"]["num_ctx"] = client_config.extra.get("num_ctx")

    # Convert pydantic BaseModel to json schema    
    if is_BaseModel(response_format):
        schema = response_format.model_json_schema()
        kwargs["format"] = schema
    elif type(response_format) is dict:
        if response_format.get("json_schema"):
            kwargs["format"] = response_format["json_schema"]["schema"]
        else:
            kwargs["format"] = response_format        
    else:
        raise Exception("completions.patched_parse: response_format must be a dict or a pydantic BaseModel")    
    
    # Call ollama
    
    response = chat(**kwargs)
        
    # Format ollama output to match openai output
    
    stream = kwargs["stream"]
    from .response.ollama.completions_factory import CompletionsFactory
    factory = CompletionsFactory()
    response = factory.message.build_content(response)
    response = attach_extractor(response,stream)
    return response

def anthropic_parse(client_config,response_format, **kwargs):

    # Convert pydantic BaseModel to json schema    

    if is_BaseModel(response_format):
        response_format = response_format.model_json_schema()
    elif type(response_format) is dict:
        if response_format.get("json_schema"):
            response_format = response_format["json_schema"]["schema"]
        else:
            response_format = response_format        
    else:
        raise Exception("completions.patched_parse: response_format must be a dict or a pydantic BaseModel")    

    # Convert from json_schema to anthropic tool format

    anthropic_tool = {
        "name": "structured_output",
        "description": "Structured output for the response",
        "input_schema": response_format
    }
    
    # Hack the messages to tell claude to return result in json format.
    messages = kwargs.get("messages", [{"role":"user","content": ""}])
    messages[-1]["content"] += "Return response in JSON format."

    import anthropic
    
    # Map openai parameters to anthropic parameters
    final_kwargs = {
        "model": kwargs.get("model", "claude-opus-4-20250514"),
        "messages": messages,
        "max_tokens": kwargs.get("max_tokens", 1000),
        "stream": False,
        "tools": [anthropic_tool],
    }
    if kwargs.get("top_k",None):
        final_kwargs["top_k"] = kwargs.get("top_k")
    if kwargs.get("timeout",None):
        final_kwargs["timeout"] = kwargs.get("timeout")
    
    client = anthropic.Anthropic()    
    response = client.messages.create(**final_kwargs)
    logger.debug(f"anthropic_parse: raw response: {response}")
    
    # Format anthropic output to match openai output
    from .response.anthropic.completions_factory import CompletionsFactory
    factory = CompletionsFactory()
    response = factory.message.build_toolcall(response)
    response = attach_extractor(response,False)
    return response

# gai_parse(): This function calls the gai ChatClient() function.

def gai_parse(client_config,response_format, **kwargs):
    from gai.chat.client import ChatClient    
    
    # Map openai parameters to gai parameters
    kwargs = {
        "model": kwargs.get("model", "ttt"),
        "messages": kwargs.get("messages", None),
        "stream": False,
        "max_tokens": kwargs.get("max_tokens", None),
        "timeout": kwargs.get("timeout", None),
    }
    if is_BaseModel(response_format):
        schema = response_format.model_json_schema()
        kwargs["json_schema"] = schema
    elif type(response_format) is dict:
        if response_format.get("json_schema"):
            kwargs["json_schema"] = response_format["json_schema"]["schema"]
        else:
            kwargs["json_schema"] = response_format        
    else:
        raise Exception("completions.patched_parse: response_format must be a dict or a pydantic BaseModel")    

    chat_client = ChatClient(client_config)
    response = chat_client.chat(**kwargs)
    return response

# This class is used by the monkey patch to override the openai's chat.completions.create() function.
# This is also the class responsible for for GAI's text-to-text completion.
# The main driver is the create() function that can be used to generate or stream completions as JSON output.
# The output from create() should be indisguishable from the output of openai's chat.completions.create() function.
#
# Example:
# from openai import OpenAI
# client = OpenAI()
# from gai.chat.openai.patch import patch_chatcompletions
# openai=patch_chatcompletions(openai)
# openai.chat.completions.create(model="llama3.1", messages=[{"role": "system", "content": "You are a helpful assistant."}], max_tokens=100)

# override_get_client_from_model is meant to be used for unit testing
def patch_chatcompletions(openai_client, file_path:str=None, client_config: Optional[Union[GaiClientConfig|dict]]=None):

    # During patch time, the client is patched with the following new functions.

    # a) Add get_client_config() function to the client.
    
    def get_client_config(model:str):
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
                client_config = config_helper.get_client_config(client_config)
            elif not isinstance(client_config, GaiClientConfig):
                raise ValueError(f"__init__: Invalid config provided")
        else:
            # If not config is provided, load config from path
            client_config = config_helper.get_client_config(model,file_path=file_path)    
            
        return client_config
    
    openai_client.get_client_config = get_client_config
    
    # b) Add LLM Client specific functions
    
    ##  Step 1: Add completions.create()

    openai_client.openai_create = openai_create
    openai_client.ollama_create = ollama_create
    openai_client.anthropic_create = anthropic_create
    openai_client.gai_create = gai_create
    
    ## Step 2: Add completions.parse()
    
    openai_client.openai_parse = openai_parse
    openai_client.ollama_parse = ollama_parse
    openai_client.gai_parse = gai_parse
    openai_client.anthropic_parse = anthropic_parse
    
    ## Step 3: Add routing for completions.create()

    def patched_create(**kwargs):
        nonlocal openai_client
        patched_client = openai_client
        model = kwargs.get("model")
        client_config = patched_client.get_client_config(model)
        client_type = client_config.client_type

        if client_type == "openai":
            return patched_client.openai_create(patched_client, **kwargs)    
        
        if client_type == "ollama":
            return patched_client.ollama_create(client_config, **kwargs)
        
        if client_type == "gai":
            return patched_client.gai_create(client_config, **kwargs)
        
        if client_type == "anthropic":
            return patched_client.anthropic_create(client_config, **kwargs)
        
        error_message = f"patched_create: Invalid client type: {client_type}"
        logger.error(error_message)
        raise Exception(error_message)

    ## Step 4: Add routing for completions.parse()
    
    def patched_parse(**kwargs):

        nonlocal openai_client
        patched_client = openai_client
        model = kwargs.get("model")
        client_config = patched_client.get_client_config(model)
        client_type = client_config.client_type

        if client_type == "openai":
            return patched_client.openai_parse(patched_client, **kwargs)    
        
        if client_type == "ollama":
            return patched_client.ollama_parse(client_config, **kwargs)
        
        if client_type == "gai":
            return patched_client.gai_parse(client_config, **kwargs)

        if client_type == "anthropic":
            return patched_client.anthropic_parse(client_config, **kwargs)
        
        error_message = f"patched_parse: Invalid client type: {client_type}"
        logger.error(error_message)
        raise Exception(error_message)
        
    # Backup the original and patch the client with the patched_create() function.
    
    if not hasattr(openai_client.chat.completions, 'is_patched'):
        openai_client.chat.completions.original_openai_create = openai_client.chat.completions.create
        openai_client.chat.completions.create = patched_create
        openai_client.chat.completions.is_patched = True
    else:
        error_message = "patched_create: Attempted to re-patch the OpenAI client which is already patched."
        logger.error(error_message)
        raise Exception(error_message)

    if not hasattr(openai_client.beta.chat.completions.parse, 'is_patched'):      
        openai_client.beta.chat.completions.original_openai_parse = openai_client.beta.chat.completions.parse
        openai_client.beta.chat.completions.parse = patched_parse
        openai_client.beta.chat.completions.is_patched = True
        
    else:
        error_message = "patched_parse: Attempted to re-patch the OpenAI client which is already patched."
        logger.error(error_message)
        raise Exception(error_message)
        
    return openai_client