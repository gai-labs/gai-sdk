import json
from ...types import ChatCompletion, ChatCompletionMessage, Choice , CompletionUsage, ChatCompletionMessageToolCall, Function

from datetime import datetime
from uuid import uuid4
from jsonschema import ValidationError

from ollama import ChatResponse
from ..output_message_builder_base import OutputMessageBuilderBase

class OutputMessageBuilder(OutputMessageBuilderBase):
    """
    # Documentation
    Descriptions: This class is used to build an OpenAI-styled ChatCompletion object to be returned from text generation.
    It is used to maintain compatibility with the OpenAI API design to facilitate drop-in replacements.
    Example: Used by generating text generation and text streaming output.
    """

    def build_toolcall(self,result,generator_name="ollama"):
        if result.message.tool_calls is None:
            raise Exception("OutputMessageBuilder.build_toolcall: tool_call is None")

        prompt_tokens=result.prompt_eval_count
        new_tokens=result.eval_count

        try:
            function_name = result.message.tool_calls[0].function.name
            function_arguments = json.dumps(result.message.tool_calls[0].function.arguments)
            return OutputMessageBuilder(
                ).add_chat_completion(generator_name=generator_name
                    ).add_choice(finish_reason='tool_calls'
                        ).add_tool(
                            function_name=function_name,
                            function_arguments=function_arguments
                            ).add_usage(
                                prompt_tokens=prompt_tokens,
                                new_tokens=new_tokens
                                ).build()
        except ValidationError as e:
            return

    def build_content(self,result: ChatResponse,generator_name="ollama"):
       
        eos_reason=result.done_reason
        content=result.message.content
        prompt_tokens=result.prompt_eval_count
        new_tokens=result.eval_count
        total_tokens=result.prompt_eval_count + result.eval_count
        
        return OutputMessageBuilder(
            ).add_chat_completion(generator_name=generator_name
                ).add_choice(finish_reason=eos_reason,logprobs=None
                    ).add_content(
                        content=content
                        ).add_usage(
                            prompt_tokens=prompt_tokens,
                            new_tokens=new_tokens
                            ).build()
