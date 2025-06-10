import json
from dotenv import load_dotenv
from typing import Union, Optional

from openai.types.chat.chat_completion import ChatCompletion
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk

from gai.lib.prompts import chat_string_to_list
from gai.lib.http_utils import http_post
from gai.lib.errors import ApiException
from gai.lib.config import GaiClientConfig, config_helper
from gai.lib.logging import getLogger
from gai.chat.lib.dtos import ChatCompletionRequest, ModelDownloadRequest

load_dotenv()
logger = getLogger(__name__)


"""
This is a convenient function for extracting the content of the response object.
Example:
- For generation.
use `response.extract()` instead of using `response.choices[0].message.content`.
- For stream.
    for chunk in response:
        if chunk:
            chunk.extract()
"""
def attach_extractor(response: ChatCompletion,is_stream:bool):

    if not is_stream:
        # return message content
        if response.choices[0].message.content:
            response.extract = lambda: {
                "type":"content",
                "content": response.choices[0].message.content
            }
            return response
        # return message toolcall
        if response.choices[0].message.tool_calls:
            response.extract = lambda: {
                "type":"function",
                "name": response.choices[0].message.tool_calls[0].function.name,
                "arguments": response.choices[0].message.tool_calls[0].function.arguments
            }
            return response
        raise Exception("completions.attach_extractor: Response is neither content nor toolcall. Please verify the API response.")
    
    def streamer():

        for chunk in response:
            if not chunk:
                continue

            if chunk.choices[0].delta.content or chunk.choices[0].delta.role:
                chunk.extract = lambda: chunk.choices[0].delta.content

            if chunk.choices[0].delta.tool_calls:

                if chunk.choices[0].delta.tool_calls[0].function.name:
                    chunk.extract = lambda: {
                        "type":"function",
                        "name": chunk.choices[0].delta.tool_calls[0].function.name,
                    }

                if chunk.choices[0].delta.tool_calls[0].function.arguments:
                    chunk.extract = lambda: {
                        "type":"function",
                        "arguments": chunk.choices[0].delta.tool_calls[0].function.arguments,
                    }

            if chunk.choices[0].finish_reason:
                chunk.extract = lambda: {
                    "type":"finish_reason",
                    "finish_reason": chunk.choices[0].finish_reason
                }

            if not hasattr(chunk,"extract") or not chunk.extract:
                chunk.extract = lambda: ""
                #raise Exception(f"completions.streamer: Chunk response contains unexpected data that cannot be processed. chunk: {chunk.__dict__}")
            yield chunk

    return (chunk for chunk in streamer())   

class ChatClient:

    # config is either a string path or a component config
    def __init__(self, config: Optional[Union[GaiClientConfig|dict]]=None,name:Optional[str]="ttt", file_path:str=None):
        
        # Load from default config file
        self.config:GaiClientConfig = None
        
        # Convert to ClientLLMConfig
        if isinstance(config, dict):
            # Load default config and patch with provided config
            self.config = config_helper.get_client_config(config)
        elif isinstance(config, GaiClientConfig):
            self.config = config
        elif name:
            # If path is provided, load config from path            
            self.config = config_helper.get_client_config(name,file_path=file_path)
        else:
            raise ValueError(f"__init__: Invalid config or path provided")
        
        if self.config.client_type != "gai":
            raise ValueError(f"__init__: Invalid client type. client_type={self.config.client_type}")

    # Generate non stream dictionary response for easier unit testing
    def _generate_dict(self, **kwargs):
        response=None
        url = kwargs.pop("url")
        timeout = kwargs.pop("timeout",30.0)
        try:
            response = http_post(url, data={**kwargs},timeout=timeout)
            jsoned=response.json()
            completion = ChatCompletion(**jsoned)
        except ApiException as he:
                raise he
        except Exception as e:
            logger.error(f"completions._generate_dict: error={e} response={response}")
            raise e

        return completion

    # Generate streamed dictionary response for easier unit testing
    def _stream_dict(self, **kwargs):
        response=None
        url = kwargs.pop("url")
        timeout = kwargs.pop("timeout",30.0)
        try:
            response = http_post(url, data={**kwargs},timeout=timeout)
        except ApiException as he:
                raise he
        except Exception as e:
            logger.error(f"completions._stream_dict: error={e}")
            raise e

        for chunk in response.iter_lines():
            try:
                chunk = chunk.decode("utf-8")
                if type(chunk)==str:
                    yield ChatCompletionChunk(**json.loads(chunk))
            except Exception as e:
                # Report the error and continue
                logger.error(f"completions._stream_dict: error={e}")
                pass


    """
    Description:
    This function is a monkey patch for openai's chat.completions.create() function.
    It will override the default completions.create() function to call the local llm instead of gpt-4.
    Example:
    openai_client.chat.completions.create = create
    """

    def chat(self, 
                model: str,
                messages: str | list, 
                stream: bool = True, 
                max_tokens: int = None, 
                temperature: float = None, 
                top_p: float = None, 
                top_k: float = None,
                tools: list = None,
                tool_choice: str = None,
                stop: list = None,
                timeout: float = 30.0,
                json_schema: dict = None):
        
        # Prepare messages
        if not messages:
            raise Exception("Messages not provided")
        if isinstance(messages, str):
            messages = chat_string_to_list(messages)
        if messages[-1]["role"] != "assistant":
            messages.append({"role": "assistant", "content": ""})

        # Prepare payload
        kwargs = {
            "model": model,
            "url": self.config.url,
            "messages": messages,
            "stream": stream,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "json_schema": json_schema,
            "tools": tools,
            "tool_choice": tool_choice,
            "stop": stop,
            "timeout": timeout
        }

        if not stream:
            response = self._generate_dict(**kwargs)
        else:
            response = (chunk for chunk in self._stream_dict(**kwargs))

        # Attach extractor
        response = attach_extractor(response,stream)

        return response


    def pull(self,model_name,url,timeout):
        response=None
        try:
            response = http_post(url, data={"model":model_name},timeout=timeout)
            jsoned=response.json()
            completion = ChatCompletion(**jsoned)
        except ApiException as he:
                raise he
        except Exception as e:
            logger.error(f"completions._generate_dict: error={e} response={response}")
            raise e
        return completion
        
        
        
        