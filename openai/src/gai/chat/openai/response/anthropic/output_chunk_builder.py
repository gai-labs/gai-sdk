import json
from typing import Union
from ...types import (
    ChatCompletionChunk, 
    ChunkChoice, 
    ChoiceDelta, 
    ChoiceDeltaToolCall, 
    ChoiceDeltaToolCallFunction
    )

from gai.lib.logging import getLogger
logger = getLogger(__name__)

from datetime import datetime
from uuid import uuid4
from ..output_chunk_builder_base import OutputChunkBuilderBase
from anthropic.types import Message

class OutputChunkBuilder(OutputChunkBuilderBase):
    
    def build_content(self,result,generator_name="anthropic"):
        
        """
        Build a ChatCompletionChunk from an Anthropic result.
        But Anthropic uses 1st content block as text stream and 2nd content block as tool call,
        so we only want to handle the 1st content block only in this case.
        """
        
        if (result is None):
            return None

        # Add start chunk
        if (result.type == "message_start"):
            return OutputChunkBuilder(
                ).add_chunk(generator=generator_name
                    ).add_chunk_choice_delta(finish_reason=None, role="assistant"
                        ).add_chunk_choice_delta_content(content=''
                            ).build()

        # Add body chunk
        if (result.type == "content_block_delta" and result.index == 0):
            return OutputChunkBuilder(
                ).add_chunk(generator=generator_name
                    ).add_chunk_choice_delta(finish_reason=None, role=None
                        ).add_chunk_choice_delta_content(content=result.delta.text
                            ).build()

        # Add end chunk
        if (result.type == "message_delta"):
            finish_reason = "stop"
            if (result.delta.stop_reason == "max_tokens"):
                finish_reason = "length"
            elif (result.delta.stop_reason == "end_turn"):
                finish_reason = "stop"                
            else:
                raise ValueError(f"Unexpected stop reason: {result.delta.stop_reason}")                

            return OutputChunkBuilder(
                ).add_chunk(generator=generator_name
                    ).add_chunk_choice_delta(finish_reason=finish_reason, role=None
                        ).build() 

    def build_async_stream(self, async_streaming_response):
        async def async_streamer():
            async for chunk in async_streaming_response:
                processed_chunk = self.build_content(chunk)
                if processed_chunk is not None:
                    yield processed_chunk
        return async_streamer()

    tool_arguments = ""
    
    def build_toolcall(self,result:Union[Message,list],generator_name="anthropic"):
        """
        Build a ChatCompletionChunk from an Anthropic result.
        But Anthropic uses 1st content block as text stream and 2nd content block as tool call,
        so we only want to handle the 2nd content block only in this case.
        """
        
        if (result is None):
            return None

        if isinstance(result,list):

            # Return completed message

            return OutputChunkBuilder(
                ).add_chunk(generator=generator_name
                    ).add_chunk_choice_delta(finish_reason=None, role="assistant"
                        ).add_chunk_choice_delta_content(content=result
                            ).build()        

        # Add start
        
        if (result.type == "message_start"):
            return OutputChunkBuilder(
                ).add_chunk(generator=generator_name
                    ).add_chunk_choice_delta(finish_reason=None, role="assistant"
                        ).add_chunk_choice_delta_content(content=''
                            ).build()        

        # Add text chunk
        
        if (result.type == "content_block_delta" and result.delta.type=="text_delta"):
                return OutputChunkBuilder(
                    ).add_chunk(generator=generator_name
                        ).add_chunk_choice_delta(finish_reason=None, role=None
                            ).add_chunk_choice_delta_content(content=result.delta.text
                                ).build()
                            
        

        # Add end chunk
        if (result.type == "message_delta"):
            finish_reason = "stop"
            if (result.delta.stop_reason == "max_tokens"):
                finish_reason = "length"
            elif (result.delta.stop_reason == "tool_use"):
                finish_reason = "tool_calls"              
            elif (result.delta.stop_reason == "end_turn"):
                finish_reason = "stop"
            else:
                raise ValueError(f"Unexpected stop reason: {result.delta.stop_reason}")  

            return OutputChunkBuilder(
                ).add_chunk(generator=generator_name
                    ).add_chunk_choice_delta(finish_reason=finish_reason, role=None
                        ).build()  


    def build_async_tool_stream(self, async_streaming_response):
        async def async_streamer():
            completed_message = None
            content_block = None
            partial_json=""
            async for chunk in async_streaming_response:
                
                # assemble and yield completed content
                
                if chunk.type == "message_start":
                    completed_message=[]
                if chunk.type == "content_block_start":
                    content_block=chunk.content_block.model_dump()
                if chunk.type == "content_block_delta" and chunk.delta.type=="text_delta":
                    content_block["text"]+=chunk.delta.text
                if chunk.type=="content_block_delta" and chunk.delta.type=="input_json_delta":
                    partial_json+=chunk.delta.partial_json
                if chunk.type == "content_block_stop":
                    if partial_json:
                        content_block["input"]=json.loads(partial_json)
                        partial_json=""
                    completed_message.append(content_block)
                    content_block=None
                if chunk.type == "message_delta":
                    # Yield completed message
                    yield self.build_toolcall(completed_message)
                    # Return final chunk
                    yield processed_chunk
                if chunk.type == "message_stop":
                    pass
                else:
                    # yield printable chunk
                    processed_chunk = self.build_toolcall(chunk)
                    if processed_chunk is not None:
                        yield processed_chunk
        return async_streamer()