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
            tool_name = result.content[-1].name
            arguments = json.dumps(result.content[-1].input)
            if tool_name is None:
                raise Exception("OutputMessageBuilder.build_toolcall: tool_name returned None")
            
            if tool_name == "structured_output":
                #
                # Handling structured output for Anthropic:
                #
                # There is a special case when using Anthropic to return structured output.
                # Because Anthropic doesn't support structured output directly, we have to retrofit a tool_call request
                # instead. This tool_call request use the tool_name "structured_output" and the json schema is passed in as
                # 'arguments' field. So to retrieve the structured output, we need to extract from 'arguments' field of
                # the tool_call response and route it back as content output.
                #
                return OutputMessageBuilder(
                    ).add_chat_completion(generator_name=generator_name
                        ).add_choice(finish_reason="stop",logprobs=None
                            ).add_content(arguments
                                ).add_usage(
                                    prompt_tokens=prompt_tokens,
                                    new_tokens=new_tokens
                                    ).build()

            try:
                return OutputMessageBuilder(
                    ).add_chat_completion(generator_name=generator_name
                        ).add_choice(finish_reason='tool_calls'
                            ).add_tool(
                                function_name=tool_name,
                                function_arguments=arguments
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




        
