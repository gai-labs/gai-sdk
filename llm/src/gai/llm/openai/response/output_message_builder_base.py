import json
from ..types import ChatCompletion, ChatCompletionMessage, Choice , CompletionUsage, ChatCompletionMessageToolCall, Function

from datetime import datetime
from uuid import uuid4
from jsonschema import ValidationError

from ollama import ChatResponse
from abc import ABC, abstractmethod
from gai.lib.logging import getLogger
logger = getLogger(__name__)

class OutputMessageBuilderBase(ABC):
    """
    # Documentation
    Descriptions: This class is used to build an OpenAI-styled ChatCompletion object to be returned from text generation.
    It is used to maintain compatibility with the OpenAI API design to facilitate drop-in replacements.
    Example: Used by generating text generation and text streaming output.
    """
    
    @abstractmethod
    def build_content(self,result,generator_name):
        """
        Builds the content of the ChatCompletion object based on the result from the generator.
        """
        pass

    @abstractmethod
    def build_toolcall(self,result,generator_name):
        """
        Builds the content of the ChatCompletion object based on the result from the generator.
        """
        pass

    def generate_chatcompletion_id(self):
        return "chatcmpl-"+str(uuid4())
    
    def generate_creationtime(self):
        return int(datetime.now().timestamp())

    def generate_toolcall_id(self):
        return "call_"+str(uuid4())

    def add_chat_completion(self,generator_name):
        try:
            chatcompletion_id = self.generate_chatcompletion_id()
            created = self.generate_creationtime()
            self.result = ChatCompletion(
                id=chatcompletion_id,
                choices=[],
                created=created,
                model=generator_name,
                object='chat.completion',
                usage=None
            )
            return self
        except Exception as e:
            logger.error(f"OutputMessageBuilder.add_chat_completion: error={str(e)}")
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
            print(f"OutputMessageBuilder.add_choice: error={str(e)}")
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
            print(f"OutputMessageBuilder.add_content: error={str(e)}")
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
            print(f"OutputMessageBuilder.add_usage: error={str(e)}")
            raise e        
        
    def build(self):
        try:
            return self.result.copy()
        except Exception as e:
            print(f"OutputMessageBuilder.build: error={str(e)}")
            raise e