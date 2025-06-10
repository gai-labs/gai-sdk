import os
import sys

mock_dir = os.path.join(os.getcwd(), "mock_data")
if mock_dir not in sys.path:
    sys.path.insert(0, mock_dir)
from mock_data.mock_openai_patch import chat_completions_generate,chat_completions_stream ,chat_completions_toolcall, chat_completions_jsonschema

from unittest.mock import patch, MagicMock
from unittest.mock import ANY

from openai import OpenAI

from gai.lib.config import GaiClientConfig

from gai.chat.openai import OpenAI

"""
GPT models require OPENAI_API_KEY to be set in the environment.
"""
@patch("os.environ.get",return_value="")
def test_patch_chatcompletions_openai_create_failed_without_apikey(mock_os_environ_get):
    try:
        client = OpenAI()
        client.chat.completions.create(model="gpt-4o", messages=[{"role":"user","content":"tell me a one sentence story"}])
    except Exception as e:
        assert str(e) == "Connection error."


"""
generate: OpenAI
"""
@patch("gai.lib.config.config_helper.get_client_config", new_callable=MagicMock)
def test_patch_chatcompletions_openai_generate(mock_from_dict):
    
    client = OpenAI()
    
    # Mock the original create function() with data generator
    
    client.chat.completions.original_openai_create = lambda **kwargs: chat_completions_generate("openai")
    
    response = client.chat.completions.create(model="gpt-4o", messages=[{"role":"user","content":"tell me a one sentence story"}])
    
    # Reading config is not required for openai models
    
    mock_from_dict.assert_not_called()
    
    # openai_create() is called and extract() is injected correctly
    
    assert response.extract()["content"] == '"Despite being lost in the dense, mystifying forest for hours, the brave little puppy finally managed to find his way back home, surprising his family who welcomed him with more love than ever before."'

"""
stream: OpenAI
"""
@patch("gai.lib.config.config_helper.get_client_config", new_callable=MagicMock)
def test_patch_chatcompletions_openai_stream(mock_from_dict):
    
    client = OpenAI()
    
    # Mock the original create function() with data generator
    
    client.chat.completions.original_openai_create = lambda **kwargs: chat_completions_stream("openai")
    
    response = client.chat.completions.create(model="gpt-4o", messages=[{"role":"user","content":"tell me a one sentence story"}], stream=True)

    content = ""
    for chunk in response:
        extracted=chunk.extract()
        if extracted and type(extracted)==str:
            content+=extracted
    
    assert content=='"Once upon a time, a tiny, curious frog set on a journey to reach the top of the mountain, and against all odds, found a kingdom of thriving frogs living beautifully above the clouds."'

"""
toolcall: OpenAI
"""
@patch("gai.lib.config.config_helper.get_client_config", new_callable=MagicMock)
def test_patch_chatcompletions_openai_toolcall(mock_from_dict):
    
    client = OpenAI()
    
    # Mock the original create function() with data generator
    
    client.chat.completions.original_openai_create = lambda **kwargs: chat_completions_toolcall("openai")
    
    response = client.chat.completions.create(
        model="gpt-4o", 
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
    )
    #print(response.choices[0].tool_calls[0].function)
    print(response.choices[0].message.tool_calls[0].function)
    
    assert response.choices[0].message.tool_calls[0].function.name == "google"
    assert response.choices[0].message.tool_calls[0].function.arguments == '{"search_query":"current time in Singapore"}'




