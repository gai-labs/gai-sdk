import os
import sys

mock_dir = os.path.join(os.getcwd(), "mock_data")
if mock_dir not in sys.path:
    sys.path.insert(0, mock_dir)
from mock_data.mock_openai_patch import chat_completions_generate,chat_completions_stream ,chat_completions_toolcall, chat_completions_jsonschema

from unittest.mock import patch, MagicMock
from unittest.mock import ANY

from openai import OpenAI

from gai.lib.config import config_helper

from gai.chat.openai import OpenAI

"""
generate: Gai
"""
@patch("gai.chat.client.ChatClient._generate_dict")
def test_patch_chatcompletions_gai_generate(mock_ChatClient_call):
    mock_ChatClient_call.return_value = chat_completions_generate("gai")
    
    client_config = {
        "client_type": "gai",
        "url": "http://localhost:12031/gen/v1/chat/completions",
    }
    client = OpenAI(client_config=client_config)
    response = client.chat.completions.create(model="ttt", messages=[{"role":"user","content":"tell me a one sentence story"}])
    
    print(response)
    
    extracted=response.extract()
    assert extracted["content"] == "Under a tree, a little boy shared his last bread with a hungry crow. They became friends, teaching him that kindness can feed more than just Hunger."
    mock_ChatClient_call.assert_called_once_with(model="ttt", url='http://localhost:12031/gen/v1/chat/completions', messages=[{'role': 'user', 'content': 'tell me a one sentence story'}, {'role': 'assistant', 'content': ''}], stream=False, max_tokens=None, temperature=None, top_p=None, top_k=None, json_schema=None, tools=None, tool_choice=None, stop=None, timeout=None)


"""
stream: Gai
"""
@patch("gai.chat.client.ChatClient._stream_dict")
def test_patch_chatcompletions_gai_stream(mock_ChatClient_stream):
    mock_ChatClient_stream.return_value = chat_completions_stream("gai")
    
    from gai.chat.openai.patch import patch_chatcompletions
    client_config = {
        "client_type": "gai",
        "url": "http://localhost:12031/gen/v1/chat/completions",
    }
    client = OpenAI(client_config=client_config)
    response = client.chat.completions.create(model="gai", messages=[{"role":"user","content":"tell me a one sentence story"}], stream=True)

    content = ""
    for chunk in response:
        if hasattr(chunk,"extract"):
            extracted=chunk.extract()
            if extracted and type(extracted)==str:
                content+=extracted
    assert content=="An angry old drunk walks through the streets yelling at cars and throwing bottles."

"""
toolcall: Gai
"""
@patch("gai.chat.client.ChatClient._generate_dict")
def test_patch_chatcompletions_gai_toolcall(mock_ChatClient):
    mock_ChatClient.return_value = chat_completions_toolcall("gai")
    
    from gai.chat.openai.patch import patch_chatcompletions
    client_config = {
        "client_type": "gai",
        "url": "http://localhost:12031/gen/v1/chat/completions",
    }
    client = OpenAI(client_config=client_config)

    response = client.chat.completions.create(
        model="gai",
        messages=[{"role": "user", "content": "What is the current time in Singapore?"}],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "google",
                    "description": "The 'google' function is a powerful tool that allows the AI to gather external information from the internet using Google search. It can be invoked when the AI needs to answer a question or provide information that requires up-to-date, comprehensive, and diverse sources which are not inherently known by the AI. For instance, it can be used to find current date, current news, weather updates, latest sports scores, trending topics, specific facts, or even the current date and time. The usage of this tool should be considered when the user's query implies or explicitly requests recent or wide-ranging data, or when the AI's inherent knowledge base may not have the required or most current information. The 'search_query' parameter should be a concise and accurate representation of the information needed.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "search_query": {
                                "type": "string",
                                "description": "The search query to search google with. For example, to find the current date or time, use 'current date' or 'current time' respectively."
                            }
                        },
                        "required": ["search_query"]
                    }
                }
            }
        ],
        tool_choice="required",
        stream=False
    )
    
    assert response.choices[0].message.tool_calls[0].function.name == "google"
    assert response.choices[0].message.tool_calls[0].function.arguments == '{"search_query": "current time in Singapore"}'