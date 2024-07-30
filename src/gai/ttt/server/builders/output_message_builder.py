import json
from openai.types.chat.chat_completion import ChatCompletion, ChatCompletionMessage, Choice , CompletionUsage
from openai.types.chat.chat_completion_message_tool_call import ChatCompletionMessageToolCall
from openai.types.chat.chat_completion_message_tool_call_param import Function
from datetime import datetime
from uuid import uuid4
import re
from jsonschema import validate, ValidationError

class OutputMessageBuilder:
    """
    # Documentation
    Descriptions: This class is used to build an OpenAI-styled ChatCompletion object to be returned from text generation.
    It is used to maintain compatibility with the OpenAI API design to facilitate drop-in replacements.
    Example: Used by generating text generation and text streaming output.
    """

    def build_toolcall(self,result):
        state = "function_name"
        eos_reason=result["eos_reason"]
        if eos_reason=="stop_string":
            eos_reason="stop"
        if eos_reason=="stop_token":
            eos_reason="stop"
        if eos_reason=="max_new_tokens":
            eos_reason="length"

        from gai.lib.common.generators_utils import get_tools_schema
        text = result["full_completion"]
        jsoned = json.loads(text)
        schema = get_tools_schema()        
        try:
            validate(instance={
                "type":"function","function":jsoned
            }, schema=schema)

            function_name = None
            if state == "function_name":
                text = result["full_completion"]
                text = re.sub(r'\s+', ' ', text)
                function_name_pattern = r'\"name\"\s*:\s*\"(.*?)\",'
                match = re.search(function_name_pattern, text, re.DOTALL)
                if match:
                    function_name=match.group(1)
                    state = "function_args"

            function_arguments = None
            if state == "function_args":
                text = result["full_completion"]
                text = re.sub(r'\s+', ' ', text)
                function_arguments_pattern = r'"(parameters|arguments)"\s*:\s*({.*?})'
                match = re.search(function_arguments_pattern, text, re.DOTALL)
                if match:
                    function_arguments=match.group(2)
                    # Remove escape sequence from keys
                    function_arguments = function_arguments.replace("\\","")
                    # If not parsable, return None and continue streaming.
                    try:
                        function_arguments = json.dumps(json.loads(function_arguments))
                    except Exception as e:
                        return None

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
            return



    def build_content(self,result):
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

    def generate_chatcompletion_id(self):
        return "chatcmpl-"+str(uuid4())

    def generate_creationtime(self):
        return int(datetime.now().timestamp())

    def generate_toolcall_id(self):
        return "call_"+str(uuid4())

    def add_chat_completion(self,generator):
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

    def add_choice(self,finish_reason,logprobs=None):
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
        

    def add_tool(self,function_name,function_arguments):
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

    def add_content(self,content):
        try:
            self.result.choices[0].message.content = content
            self.result.choices[0].message.tool_calls = None
            return self
        except Exception as e:
            print("OutputMessageBuilder.add_content:",e)
            raise e
    
    def add_usage(self, prompt_tokens, new_tokens):
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
    
    def build(self):
        try:
            return self.result.copy()
        except Exception as e:
            print("OutputMessageBuilder.build:",e)
            raise e
