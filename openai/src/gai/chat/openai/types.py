# Standalone openai types module extracted from https://github.com/openai/openai-python/archive/refs/tags/v1.43.9.tar.gz

import os
from typing import Optional, Literal, List
from typing_extensions import TypedDict, Required
import pydantic
from pydantic import Extra, BaseModel as PydanticBaseModel

# Default to using the library unless explicitly overridden
USE_HARDCODED_CLASSES = os.getenv("USE_HARDCODED_CLASSES", "false").lower() == "true"

if not USE_HARDCODED_CLASSES:

    # Import classes from the installed openai library
    from openai.types.chat.chat_completion_chunk import ChatCompletionChunk, Choice as ChunkChoice, ChoiceDelta
    from openai.types.chat.chat_completion_chunk import ChoiceDeltaToolCall, ChoiceDeltaToolCallFunction    
    from openai.types.chat.chat_completion import ChatCompletion, ChatCompletionMessage, Choice , CompletionUsage
    from openai.types.chat.chat_completion_message_tool_call import ChatCompletionMessageToolCall
    from openai.types.chat.chat_completion_message_tool_call_param import Function
    from openai.types.chat_model import ChatModel

else:

    try:
        import pydantic
        PYDANTIC_V2 = pydantic.VERSION.startswith("2.")
    except ImportError:
        PYDANTIC_V2 = False

    class BaseModel(PydanticBaseModel):
        """
        Stub BaseModel for extracted OpenAI types.
        Inherit from Pydantic’s BaseModel, allow extra fields.
        """

        if PYDANTIC_V2:
            # Use the new `model_config` attribute for Pydantic v2
            model_config = {
                "extra": "allow"
            }
        else:
            # For Pydantic v1, use the Config class
            class Config:
                extra = "allow"
                
    class Choice(BaseModel):
        delta: dict
        finish_reason: Optional[Literal["stop", "length", "tool_calls", "content_filter", "function_call"]] = None
        index: int
        
    class CompletionUsage(BaseModel):
        completion_tokens: int
        prompt_tokens: int
        total_tokens: int
        
    class ChatCompletion(BaseModel):
        id: str
        choices: List[Choice]
        created: int
        model: str
        object: Literal["chat.completion"]
        service_tier: Optional[Literal["scale", "default"]] = None
        system_fingerprint: Optional[str] = None
        usage: Optional[CompletionUsage] = None

    class Function(TypedDict, total=False):
        arguments: Required[str]
        name: Required[str]

    class ChatCompletionMessageToolCall(BaseModel):
        id: str
        function: Function
        type: Literal["function"]

    class FunctionCall(BaseModel):
        arguments: str
        name: str

    class ChatCompletionMessage(BaseModel):
        content: Optional[str] = None
        refusal: Optional[str] = None
        role: Literal["assistant"]
        function_call: Optional[FunctionCall] = None
        tool_calls: Optional[List[ChatCompletionMessageToolCall]] = None
        
    # ChatCompletionChunk

    class ChatCompletionChunk(BaseModel):
        id: str
        choices: List[Choice]
        created: int
        model: str
        object: Literal["chat.completion.chunk"]
        service_tier: Optional[Literal["scale", "default"]] = None
        system_fingerprint: Optional[str] = None
        usage: Optional[CompletionUsage] = None

    # ChoiceDeltaToolCallFunctionCall
        
    class ChoiceDeltaToolCallFunction(BaseModel):
        arguments: Optional[str] = None
        name: Optional[str] = None

    # ChoiceDeltaToolCall

    class ChoiceDeltaToolCall(BaseModel):
        index: int
        id: Optional[str] = None
        function: Optional[ChoiceDeltaToolCallFunction] = None
        type: Optional[Literal["function"]] = None

    # ChoiceDeltaFunctionCall

    class ChoiceDeltaFunctionCall(BaseModel):
        arguments: Optional[str] = None
        name: Optional[str] = None

    # ChoiceDelta

    class ChoiceDelta(BaseModel):
        content: Optional[str] = None
        function_call: Optional[ChoiceDeltaFunctionCall] = None
        refusal: Optional[str] = None
        role: Optional[Literal["system", "user", "assistant", "tool"]] = None
        tool_calls: Optional[List[ChoiceDeltaToolCall]] = None

    from typing_extensions import Literal, TypeAlias

    ChatModel: TypeAlias = Literal[
        "gpt-4o",
        "gpt-4o-2024-05-13",
        "gpt-4o-2024-08-06",
        "gpt-4o-mini",
        "gpt-4o-mini-2024-07-18",
        "gpt-4-turbo",
        "gpt-4-turbo-2024-04-09",
        "gpt-4-0125-preview",
        "gpt-4-turbo-preview",
        "gpt-4-1106-preview",
        "gpt-4-vision-preview",
        "gpt-4",
        "gpt-4-0314",
        "gpt-4-0613",
        "gpt-4-32k",
        "gpt-4-32k-0314",
        "gpt-4-32k-0613",
        "gpt-3.5-turbo",
        "gpt-3.5-turbo-16k",
        "gpt-3.5-turbo-0301",
        "gpt-3.5-turbo-0613",
        "gpt-3.5-turbo-1106",
        "gpt-3.5-turbo-0125",
        "gpt-3.5-turbo-16k-0613",
    ]

__all__ = ["ChatModel", "ChatCompletion", "ChatCompletionChunk", "ChunkChoice","ChoiceDeltaToolCallFunction","ChoiceDeltaToolCall","ChatCompletionMessage","Function","CompletionUsage"]