import json
from ...types import ChatCompletion, ChatCompletionMessage, Choice , CompletionUsage, ChatCompletionMessageToolCall, Function

from datetime import datetime
from uuid import uuid4
from jsonschema import ValidationError

from anthropic.types import Message
from ..output_message_builder_base import OutputMessageBuilderBase

class OutputMessageBuilder(OutputMessageBuilderBase):
    """
    # Documentation
    Descriptions: 
    Concrete class for building an OpenAI-styled ChatCompletion object 
    from an Anthropic Message object.
    """     
    
    def build_toolcall(self,result,generator_name="anthropic"):
        
        if result.stop_reason != "tool_use":
            raise Exception("OutputMessageBuilder.build_toolcall: stop_reason is not 'tool_use'")
        
        if result.type == "message":
            prompt_tokens=result.usage.input_tokens
            new_tokens=result.usage.output_tokens
            
            #string_content = json.dumps([content_block.model_dump() for content_block in result.content])
            content = [content_block.model_dump() for content_block in result.content]
            
            try:
                return OutputMessageBuilder(
                    ).add_chat_completion(generator_name=generator_name
                        ).add_choice(finish_reason="stop",logprobs=None
                            ).add_content(content
                                ).add_usage(
                                    prompt_tokens=prompt_tokens,
                                    new_tokens=new_tokens
                                    ).build()
            except ValidationError as e:
                return
        
        raise Exception("OutputMessageBuilder.build_toolcall: result type is not 'message'")          
    
    def build_content(self,result: Message,generator_name="anthropic"):
    
        finish_reason="stop"   
        if (result.stop_reason == "max_tokens"):
            finish_reason = "length"                
        if (result.stop_reason == "tool_use"):
            raise Exception("OutputMessageBuilder.build_content: stop_reason is 'tool_use', use build_toolcall instead")
        
        content = result.content[0].text
        prompt_tokens = result.usage.input_tokens
        new_tokens = result.usage.output_tokens
        total_tokens = prompt_tokens + new_tokens
        
        return OutputMessageBuilder(
            ).add_chat_completion(generator_name=generator_name
                ).add_choice(finish_reason=finish_reason,logprobs=None
                    ).add_content(
                        content=content
                        ).add_usage(
                            prompt_tokens=prompt_tokens,
                            new_tokens=new_tokens
                            ).build()




        
