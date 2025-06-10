from .patch import patch_chatcompletions
from .json_schemas import boolean_schema
from .attach_extractor import ToolCallLastChunk, LastChunk

def OpenAI(client_config=None,**kwargs):
    from openai import OpenAI
    from gai.chat.openai import patch_chatcompletions
    return patch_chatcompletions(OpenAI(**kwargs), client_config=client_config)

def AsyncOpenAI(client_config=None, **kwargs):
    from openai import AsyncOpenAI
    from gai.chat.openai.async_patch import patch_async_chatcompletions
    return patch_async_chatcompletions(AsyncOpenAI(**kwargs),client_config=client_config)

__all__ = [
    "OpenAI",
    "boolean_schema",
    "patch_chatcompletions",
    "ToolCallLastChunk",
    "LastChunk"
]