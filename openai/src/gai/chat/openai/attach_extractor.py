import json
import inspect
from .types import ChatCompletion
from pydantic import BaseModel

"""
This is a convenient function for extracting content from various OpenAI response components.
It works by injecting an `extract` method into the response object.

Example:
- For generation. This will return the text output.

    use `response.extract()` instead of using `response.choices[0].message.content`.

- For stream. This will return the text content of each chunk as it is streamed.

    for chunk in response:
        if chunk:
            chunk.extract()

- For streaming tool calls. This will return the tool name and arguments as a single chunk at the end of the stream.

    for chunk in response:
        if hasattr(chunk, 'extract'):
            chunk.extract()
"""

class LastChunk(BaseModel):
    """
    This is a marker object for the last chunk of a text stream.
    """
    
    type: str
    finish_reason: str

class ToolCallLastChunk(LastChunk):
    """
    This is a marker object for the last chunk of a text stream that contains a tool call.
    """   
    tool_name: str
    arguments: str

class ToolCallContent(BaseModel):
    """
    This is returned as the extracted content of an non-streaming tool call response.
    """
    tool_name: str
    arguments: str

def attach_extractor(response: ChatCompletion,is_stream:bool):
    tool_name = ""
    arguments = ""

    if not is_stream:
        
        # return message content
        if response.choices[0].message.content:
            
            # This returns both structured and unstructured output.
            
            response.extract = lambda: response.choices[0].message.content
            return response
        
        # return message toolcall
        if response.choices[0].message.tool_calls:

            response.extract = lambda: ToolCallContent(
                tool_name=response.choices[0].message.tool_calls[0].function.name,
                arguments=response.choices[0].message.tool_calls[0].function.arguments
            )                
            return response
        raise Exception("completions.attach_extractor: Response is neither content nor toolcall. Please verify the API response.")
    
    def streamer():
        nonlocal tool_name, arguments
        
        # Each chunk component has a different way of parsing so the extract function has to cater for each component.
        
        for chunk in response:
            if not chunk:
                continue

            if chunk.choices[0].delta.content:
                
                # This is used for parsing text streams.
                
                chunk.extract = lambda: chunk.choices[0].delta.content

            if chunk.choices[0].delta.tool_calls:
                
                # The way way handle tool streams is to combine the tool name and arguments into a single chunk
                
                if chunk.choices[0].delta.tool_calls[0].function.name:
                    
                    # The tool name is buffered for now and returned at the end of tool stream.
                    tool_name = chunk.choices[0].delta.tool_calls[0].function.name
                    #chunk.extract = lambda: tool_name
                
                if chunk.choices[0].delta.tool_calls[0].function.arguments:
                    
                    # The arguments is partial json at this point and is buffered for now and returned at the end of tool stream.
                    arguments += chunk.choices[0].delta.tool_calls[0].function.arguments
                    #chunk.extract = lambda: arguments

            if chunk.choices[0].finish_reason:
                
                if chunk.choices[0].finish_reason == "tool_calls" and tool_name:
                    
                    if not arguments:
                        arguments = json.dumps({
                            "type": "object",
                            "properties": {},
                            "required": []                            
                        })
                    
                    # If the finish reason is tool_calls, we return the buffered tool name and arguments.
                    chunk.extract = lambda: ToolCallLastChunk(
                        type="finish_reason",
                        finish_reason=chunk.choices[0].finish_reason,
                        tool_name=tool_name,
                        arguments=arguments
                        )
                        
                else:
                    chunk.extract = lambda: LastChunk(
                        type="finish_reason",
                        finish_reason=chunk.choices[0].finish_reason
                    )
            
            if not hasattr(chunk, 'extract'):
                # If no extract method is set, we set it to return ""
                chunk.extract = lambda: ""
                
            yield chunk

    return (chunk for chunk in streamer())   

def attach_extractor_async(response, is_stream: bool):
    """
    Async-aware version of attach_extractor.
    
    CRITICAL: Use try/catch as backup since is_async_generator() can miss edge cases.
    """
    import json
    
    tool_name = ""
    arguments = ""

    if not is_stream:
        # Non-streaming case - same as original
        if response.choices[0].message.content:
            response.extract = lambda: response.choices[0].message.content
            return response
        
        if response.choices[0].message.tool_calls:
            response.extract = lambda: {
                "tool_name": response.choices[0].message.tool_calls[0].function.name,
                "arguments": response.choices[0].message.tool_calls[0].function.arguments
            }
            return response
        raise Exception("attach_extractor_async: Response is neither content nor toolcall")
    
    # Streaming case - need robust detection
    def extract_chunk_logic(chunk):
        """Common extraction logic - prevents code duplication"""
        nonlocal tool_name, arguments
        
        if not chunk:
            return chunk
            
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
            
        return chunk
    
    # Robust async detection - check multiple indicators
    def is_truly_async_iterable(obj):
        """More comprehensive async detection"""
        # Direct async generator check
        if inspect.isasyncgen(obj):
            return True
        # Has async iterator protocol
        if hasattr(obj, '__aiter__'):
            return True
        # Check string representation for async_generator
        if 'async_generator' in str(type(obj)):
            return True
        # Check if trying regular iteration fails
        try:
            iter(obj)
            return False  # If iter() works, it's sync
        except TypeError:
            # If iter() fails, might be async-only
            return hasattr(obj, '__aiter__') or 'async' in str(type(obj))
    
    if is_truly_async_iterable(response):
        # Handle async generators/iterators - use async for
        async def async_streamer():
            async for chunk in response:
                yield extract_chunk_logic(chunk)
        return async_streamer()
    else:
        # Handle sync iterators - convert to async generator
        async def sync_to_async_streamer():
            for chunk in response:
                yield extract_chunk_logic(chunk)
        return sync_to_async_streamer()
