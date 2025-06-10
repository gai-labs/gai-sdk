import json
import httpx
from dotenv import load_dotenv
from typing import AsyncGenerator, Union, Optional

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
                "type": "content",
                "content": response.choices[0].message.content
            }
            return response
        # return message toolcall
        if response.choices[0].message.tool_calls:
            response.extract = lambda: {
                "type": "function",
                "name": response.choices[0].message.tool_calls[0].function.name,
                "arguments": response.choices[0].message.tool_calls[0].function.arguments
            }
            return response
        raise Exception("completions.attach_extractor: Response is neither content nor toolcall. Please verify the API response.")
      
    async def async_streamer():
        async for chunk in response:
            if not chunk:
                continue

            if chunk.choices[0].delta.content or chunk.choices[0].delta.role:
                chunk.extract = lambda: chunk.choices[0].delta.content

            if chunk.choices[0].delta.tool_calls:

                if chunk.choices[0].delta.tool_calls[0].function.name:
                    chunk.extract = lambda: {
                        "type": "function",
                        "name": chunk.choices[0].delta.tool_calls[0].function.name,
                    }

                if chunk.choices[0].delta.tool_calls[0].function.arguments:
                    chunk.extract = lambda: {
                        "type": "function",
                        "arguments": chunk.choices[0].delta.tool_calls[0].function.arguments,
                    }

            if chunk.choices[0].finish_reason:
                chunk.extract = lambda: {
                    "type": "finish_reason",
                    "finish_reason": chunk.choices[0].finish_reason
                }

            if not hasattr(chunk, "extract") or not chunk.extract:
                chunk.extract = lambda: ""
            yield chunk

    return async_streamer()

class AsyncChatClient:

    def __init__(self, config: Optional[Union[GaiClientConfig,dict]]=None, name: Optional[str]="ttt", file_path: Optional[str]=None):
        
      # Load from default config file
        self.config: GaiClientConfig = None
        
        # Convert to ClientLLMConfig
        if isinstance(config, dict):
            # Load default config and patch with provided config
            self.config = config_helper.get_client_config(config)
        elif isinstance(config, GaiClientConfig):
            self.config = config
        elif name:
            # If path is provided, load config from path            
            self.config = config_helper.get_client_config(name, file_path=file_path)
        else:
            raise ValueError(f"__init__: Invalid config or path provided")
        
        if self.config.client_type != "gai":
            raise ValueError(f"__init__: Invalid client type. client_type={self.config.client_type}")

        # HTTP client for connection pooling
        self._client: Optional[httpx.AsyncClient] = None

    async def __aenter__(self):
        """Async context manager entry"""
        await self._ensure_client()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()

    async def _ensure_client(self):
        """Ensure httpx client is initialized"""
        if self._client is None:
            # Create a new httpx.AsyncClient
            self._client = httpx.AsyncClient()
        return self._client

    async def close(self):
        """Close the httpx client"""
        if self._client:
            await self._client.aclose()
            self._client = None

    # Generate non stream dictionary response for easier unit testing
    async def _generate_dict(self, **kwargs) -> ChatCompletion:
        url = kwargs.pop("url")
        timeout = kwargs.pop("timeout", 30.0)
        
        await self._ensure_client()
        
        try:
            response = await self._client.post(url, json={**kwargs}, timeout=timeout)
            response.raise_for_status()
            jsoned = response.json()
            completion = ChatCompletion(**jsoned)
        except httpx.HTTPStatusError as e:
            raise ApiException(code=e.response.status_code, message=f"HTTP {e.response.status_code}: {e.response.text}")
        except httpx.TimeoutException:
            raise ApiException(code=408, message=f"Request timeout after {timeout} seconds")
        except httpx.RequestError as e:
            raise ApiException(code=500, message=f"HTTP request failed: {e}")
        except Exception as e:
            logger.error(f"completions._generate_dict: error={e}")
            raise e

        return completion

    # Generate streamed dictionary response for easier unit testing
    async def _stream_dict(self, **kwargs) -> AsyncGenerator[ChatCompletionChunk, None]:
        url = kwargs.pop("url")
        timeout = kwargs.pop("timeout", 30.0)
        
        await self._ensure_client()
        
        try:
            async with self._client.stream('POST', url, json={**kwargs}, timeout=timeout) as response:
                response.raise_for_status()
                
                async for line in response.aiter_lines():
                    try:
                        line = line.strip()
                        if line:
                            # Handle Server-Sent Events format
                            if line.startswith('data: '):
                                line = line[6:]  # Remove 'data: ' prefix
                            if line == '[DONE]':
                                break
                            if line:
                                yield ChatCompletionChunk(**json.loads(line))
                    except Exception as e:
                        # Report the error and continue
                        logger.error(f"completions._stream_dict: error={e}, line={line}")
                        continue
                        
        except httpx.HTTPStatusError as e:
            raise ApiException(f"HTTP {e.response.status_code}: {e.response.text}")
        except httpx.TimeoutException:
            raise ApiException(f"Request timeout after {timeout} seconds")
        except httpx.RequestError as e:
            raise ApiException(f"HTTP request failed: {e}")
        except Exception as e:
            logger.error(f"completions._stream_dict: error={e}")
            raise e

    async def chat(self, 
                model: str,
                messages: str | list, 
                stream: bool = True, 
                max_tokens: Optional[int] = None, 
                temperature: Optional[float] = None, 
                top_p: Optional[float] = None, 
                top_k: Optional[float] = None,
                tools: Optional[list] = None,
                tool_choice: Optional[str] = None,
                stop: Optional[list] = None,
                timeout: Optional[float] = 30.0,
                json_schema: Optional[dict] = None):
        
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
            response = await self._generate_dict(**kwargs)
        else:
            response = self._stream_dict(**kwargs)

        # Attach extractor
        response = attach_extractor(response, stream)

        return response

    async def pull(self, model_name: str, url: str, timeout: float) -> ChatCompletion:
        await self._ensure_client()
        
        try:
            response = await self._client.post(url, json={"model": model_name}, timeout=timeout)
            response.raise_for_status()
            jsoned = response.json()
            completion = ChatCompletion(**jsoned)
        except httpx.HTTPStatusError as e:
            raise ApiException(f"HTTP {e.response.status_code}: {e.response.text}")
        except httpx.TimeoutException:
            raise ApiException(f"Request timeout after {timeout} seconds")
        except httpx.RequestError as e:
            raise ApiException(f"HTTP request failed: {e}")
        except Exception as e:
            logger.error(f"completions.pull: error={e}")
            raise e
        return completion