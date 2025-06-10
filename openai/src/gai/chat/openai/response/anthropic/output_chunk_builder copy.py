import json
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
    
    tool_arguments = ""
    
    def build_toolcall(self,result:Message,generator_name="anthropic"):
        """
        Build a ChatCompletionChunk from an Anthropic result.
        But Anthropic uses 1st content block as text stream and 2nd content block as tool call,
        so we only want to handle the 2nd content block only in this case.
        """
        
        if (result is None):
            return None

        # Add start
        
        if (result.type == "message_start"):
            return OutputChunkBuilder(
                ).add_chunk(generator=generator_name
                    ).add_chunk_choice_delta(finish_reason=None, role="assistant"
                        ).add_chunk_choice_delta_content(content=''
                            ).build()        

        # Add text chunk
        
        if (result.type == "content_block_delta" and result.index == 0):
            if not hasattr(result.delta,"text"):
                logger.warning(f"output_chunk_builder.build_toolcall: Anthropic content block delta does not have text attribute, skipping it. chunk={result}")
            else:
                content = result.delta.text
                return OutputChunkBuilder(
                    ).add_chunk(generator=generator_name
                        ).add_chunk_choice_delta(finish_reason=None, role=None
                            ).add_chunk_choice_delta_content(content=content
                                ).build()
        
        # For tool calling, the very first chunk will contain the tool name.
        
        if (result.type == "content_block_start" and result.index == 1):
            return OutputChunkBuilder(
                ).add_chunk(generator=generator_name
                    ).add_chunk_choice_delta(finish_reason=None, role="assistant"
                        ).add_chunk_choice_delta_toolcall_name(name=result.content_block.name
                            ).build()

        # Buffer body chunk
        
        if (result.type == "content_block_delta" and result.index == 1):
            # Buffer the partial JSON for the tool call
            return OutputChunkBuilder(
                ).add_chunk(generator=generator_name
                    ).add_chunk_choice_delta(finish_reason=None, role=None
                        ).add_chunk_choice_delta_toolcall_arguments(arguments=result.delta.partial_json
                            ).build()

        # if (result.type == "content_block_delta" and result.index == 1):
        #     # Buffer the partial JSON for the tool call
        #     self.tool_arguments += result.delta.partial_json
            
        #     # Return None forces the caller not to yield this chunk yet
        #     return None

        # # Add body chunk
        # if (result.type == "content_block_stop" and result.index == 1):
        #     # Return all buffered tool call arguments as a chunk
        #     return OutputChunkBuilder(
        #         ).add_chunk(generator=generator_name
        #             ).add_chunk_choice_delta(finish_reason=None, role=None
        #                 ).add_chunk_choice_delta_toolcall_arguments(arguments=self.tool_arguments
        #                     ).build()

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
            content = result.delta.text
            return OutputChunkBuilder(
                ).add_chunk(generator=generator_name
                    ).add_chunk_choice_delta(finish_reason=None, role=None
                        ).add_chunk_choice_delta_content(content=content
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

    def build_async_tool_stream(self, async_streaming_response):
        if not getattr(self, 'build_toolcall', None):
            raise NotImplementedError("build_toolcall method is not implemented")
        async def async_streamer():
            async for chunk in async_streaming_response:
                processed_chunk = self.build_toolcall(chunk)
                if processed_chunk is not None:
                    yield processed_chunk
        return async_streamer()