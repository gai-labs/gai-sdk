from ..types import (
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
from abc import ABC, abstractmethod

class OutputChunkBuilderBase(ABC):
    
    @abstractmethod
    def build_content(self,result,generator_name="anthropic"):
        pass 
    
    def build_stream(self,streaming_response):
        def streamer():
            for chunk in streaming_response:
                yield self.build_content(chunk)
        return (chunk for chunk in streamer())

    def build_tool_stream(self,streaming_response):
        if not getattr(self, 'build_toolcall', None):
            raise NotImplementedError("build_toolcall method is not implemented")
        def streamer():
            for chunk in streaming_response:
                chunk = self.build_toolcall(chunk)
                if chunk is not None:
                    yield chunk
        return streamer()

    def copy(self):
        return OutputChunkBuilder(self.result)

    def generate_chatcompletion_id(self):
        return "chatcmpl-"+str(uuid4())

    def generate_creationtime(self):
        return int(datetime.now().timestamp())
    
    def generate_toolCall_id(self):
        return "call_"+str(uuid4())
    
    def add_chunk(self,generator):
        try:
            chatcompletion_id = self.generate_chatcompletion_id()
            created = self.generate_creationtime()
            self.result = ChatCompletionChunk(
                id=chatcompletion_id,
                choices=[],
                created=created,
                model=generator,
                object='chat.completion.chunk'
            )
            return self
        except Exception as e:
            logger.error(f"OutputChunkBuilder.add_chunk: error={e}")
            raise
    
    def add_chunk_choice_delta(self, role=None, finish_reason=None):
        try:
            self.result.choices.append(
                ChunkChoice(
                    delta=ChoiceDelta(
                        content=None, 
                        role=role, 
                        tool_calls=None, 
                        function_call=None
                        ),
                    index=0,
                    logprobs=None,
                    finish_reason=finish_reason,
                )            
            )
            return self
        except Exception as e:
            logger.error(f"OutputChunkBuilder.add_chunk_choice_delta: error={e}")
            raise

    def add_chunk_choice_delta_content(self, content):
        try:
            self.result.choices[0].delta.content=content
            return self
        except Exception as e:
            logger.error(f"OutputChunkBuilder.add_chunk_choice_delta_content: error={e}")
            raise

    def add_chunk_choice_delta_toolcall_name(self, name):
        try:
            self.result.choices[0].delta.tool_calls=[ChoiceDeltaToolCall(
                index=0,
                id=self.generate_toolCall_id(),
                function=ChoiceDeltaToolCallFunction(
                    name=name,
                    arguments=''
                    ),
                type='function'
                )]
            return self
        except Exception as e:
            logger.error(f"OutputChunkBuilder.add_chunk_choice_delta_toolcall_name: error={e}")
            raise

    def add_chunk_choice_delta_toolcall_arguments(self, arguments):
        try:
            self.result.choices[0].delta.tool_calls=[ChoiceDeltaToolCall(
                index=0,
                id=None,
                function=ChoiceDeltaToolCallFunction(
                    name=None,
                    arguments=arguments
                    ),
                type='function'
                )]
            return self
        except Exception as e:
            logger.error(f"OutputChunkBuilder.add_chunk_choice_delta_toolcall_arguments: error={e}")
            raise
    
    def build(self):
        try:
            return self.result.copy()    
        except Exception as e:
            logger.error(f"OutputChunkBuilder.build: error={e}")
            raise
        


