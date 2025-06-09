import json
from uuid import uuid4
from openai.types.chat.chat_completion import ChatCompletion, ChatCompletionMessage, Choice , CompletionUsage
from openai.types.chat.chat_completion_message_tool_call import ChatCompletionMessageToolCall
from openai.types.chat.chat_completion_message_tool_call_param import Function
from datetime import datetime
from jsonschema import validate, ValidationError

from gai.chat.lib.generators_utils import get_tools_schema
from gai.lib.logging import getLogger
logger = getLogger(__name__)

class OutputMessageBuilder:
    """
    # Documentation
    Descriptions: This class is used to build an OpenAI-styled ChatCompletion object to be returned from text generation.
    It is used to maintain compatibility with the OpenAI API design to facilitate drop-in replacements.
    Example: Used by generating text generation and text streaming output.
    """

    def build_toolcall(self,result) -> ChatCompletion:
        state = "function_name"
        eos_reason=result["eos_reason"]
        if eos_reason=="stop_string":
            eos_reason="stop"
        if eos_reason=="stop_token":
            eos_reason="stop"
        if eos_reason=="max_new_tokens":
            eos_reason="length"
        
        text = result["full_completion"]
        jsoned = json.loads(text)
        schema = get_tools_schema()        
        try:
            validate(instance=jsoned, schema=schema)
        except ValidationError as e:
            id=str(uuid4())
            logger.error(f"OutputMessageBuilder.build_toolcall: Failed validate. error={e} text={text} id={id}")
            raise Exception(f"OutputMessageBuilder: id={id}")
    
        try:
            function_name = jsoned["function"]["name"]
            function_arguments = json.dumps(jsoned["function"]["arguments"])
            return OutputMessageBuilder(
                ).add_chat_completion(generator="exllamav2-mistral7b"
                    ).add_choice(finish_reason='tool_calls'
                        ).add_tool(
                            function_name=function_name,
                            function_arguments=function_arguments
                            ).add_usage(
                                prompt_tokens=result["prompt_tokens"],
                                new_tokens=result["new_tokens"]
                                ).build()
        except ValidationError as e:
            id=str(uuid4())
            logger.error(f"OutputMessageBuilder.build_toolcall: error={e} text={text} id={id}")
            raise Exception(f"OutputMessageBuilder: id={id}")

    def build_content(self,result) -> ChatCompletion:
        eos_reason=result["eos_reason"]
        if eos_reason=="stop_string":
            eos_reason="stop"
        if eos_reason=="stop_token":
            eos_reason="stop"
        if eos_reason=="max_new_tokens":
            eos_reason="length"
        return OutputMessageBuilder(
            ).add_chat_completion(generator="exllamav2-mistral7b"
                ).add_choice(finish_reason=eos_reason,logprobs=None
                    ).add_content(
                        content=result["full_completion"]
                        ).add_usage(
                            prompt_tokens=result["prompt_tokens"],
                            new_tokens=result["new_tokens"]
                            ).build()

    def generate_chatcompletion_id(self) -> str:
        return "chatcmpl-"+str(uuid4())

    def generate_creationtime(self) -> int:
        return int(datetime.now().timestamp())

    def generate_toolcall_id(self) -> str:
        return "call_"+str(uuid4())

    def add_chat_completion(self,generator) -> 'OutputMessageBuilder':
        try:
            chatcompletion_id = self.generate_chatcompletion_id()
            created = self.generate_creationtime()
            self.result = ChatCompletion(
                id=chatcompletion_id,
                choices=[],
                created=created,
                model=generator,
                object='chat.completion',
                usage=None
            )
            return self
        except Exception as e:
            print("OutputMessageBuilder.add_chat_completion:",e)
            raise e

    def add_choice(self,finish_reason,logprobs=None) -> 'OutputMessageBuilder':
        try:
            self.result.choices.append(Choice(
                finish_reason=finish_reason,
                index=0,
                message=ChatCompletionMessage(role='assistant',content=None, function_call=None, tool_calls=[]),
                logprobs=logprobs,
            ))
            return self
        except Exception as e:
            print("OutputMessageBuilder.add_choice:",e)
            raise e
        

    def add_tool(self,function_name,function_arguments) -> 'OutputMessageBuilder':
        try:
            toolcall_id = self.generate_toolcall_id()
            self.result.choices[0].message.tool_calls.append(ChatCompletionMessageToolCall(
                id = toolcall_id,
                function = Function(
                    name=function_name,
                    arguments=function_arguments
                ),
                type='function'
            ))
            return self
        except Exception as e:
            print("OutputMessageBuilder.add_tool:",e)
            raise e

    def add_content(self,content) -> 'OutputMessageBuilder':
        try:
            self.result.choices[0].message.content = content
            self.result.choices[0].message.tool_calls = None
            return self
        except Exception as e:
            print("OutputMessageBuilder.add_content:",e)
            raise e
    
    def add_usage(self, prompt_tokens, new_tokens) -> 'OutputMessageBuilder':
        try:
            total_tokens = prompt_tokens + new_tokens
            self.result.usage = CompletionUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=new_tokens,
                total_tokens=total_tokens
            )
            return self
        except Exception as e:
            print("OutputMessageBuilder.add_usage:",e)
            raise e
    
    def build(self) -> ChatCompletion:
        try:
            return self.result.copy()
        except Exception as e:
            print("OutputMessageBuilder.build:",e)
            raise e
