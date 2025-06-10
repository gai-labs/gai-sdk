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
generate: Ollama
"""
@patch("ollama.chat")
def test_patch_chatcompletions_ollama_generate(mock_ollama_chat):
    mock_ollama_chat.return_value = chat_completions_generate("ollama")
    
    client_config = {
        "client_type": "ollama",
        "model": "llama3.1",
    }
    client = OpenAI(client_config=client_config)
    response = client.chat.completions.create(model="llama3.1", messages=[{"role":"user","content":"tell me a one sentence story"}])
    mock_ollama_chat.assert_called_once_with(model="llama3.1", messages=[{"role":"user","content":"tell me a one sentence story"}], options={'temperature': None, 'top_k': None, 'top_p': None, 'num_predict': None}, stream=False, tools=None)


"""
stream: Ollama
"""
@patch("ollama.chat")
def test_patch_chatcompletions_ollama_stream(mock_ollama_chat):
    mock_ollama_chat.return_value = chat_completions_stream("ollama")
    
    client_config = {
        "client_type": "ollama",
        "model": "llama3.1",
    }
    client = OpenAI(client_config=client_config)
    response = client.chat.completions.create(model="llama3.1", messages=[{"role":"user","content":"tell me a one sentence story"}], stream=True)

    content = ""
    for chunk in response:
        if hasattr(chunk,"extract"):
            extracted=chunk.extract()
            if extracted and type(extracted)==str:
                content+=extracted
    print(content)
    assert content=="As she lay in bed, Emily couldn't shake the feeling that someone had been watching her from the shadows of her childhood home."

"""
toolcall: Ollama
"""
@patch("ollama.chat")
def test_patch_chatcompletions_ollama_toolcall(mock_ollama_chat):
    mock_ollama_chat.return_value = chat_completions_toolcall("ollama")
    
    client_config = {
        "client_type": "ollama",
        "model": "llama3.1",
    }
    client = OpenAI(client_config=client_config)

    response = client.chat.completions.create(
        model="llama3.1", 
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