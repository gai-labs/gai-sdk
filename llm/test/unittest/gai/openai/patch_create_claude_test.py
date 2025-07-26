from unittest.mock import MagicMock, patch, PropertyMock
from anthropic import Anthropic
from mock_data.mock_openai_patch import chat_completions_generate, chat_completions_stream, chat_completions_toolcall, chat_completions_jsonschema
from gai.llm.openai import OpenAI
from unittest.mock import ANY
from unittest.mock import patch, MagicMock
import os
import sys

mock_dir = os.path.join(os.getcwd(), "mock_data")
if mock_dir not in sys.path:
    sys.path.insert(0, mock_dir)

"""
Claude models require ANTHROPIC_API_KEY to be set in the environment.
"""


@patch("os.environ.get", return_value="")
def test_patch_chatcompletions_anthropic_create_failed_without_apikey(mock_os_environ_get):
    try:
        client = OpenAI()
        client.chat.completions.create(
            model="claude-sonnet-4-0",
            messages=[
                {"role": "user", "content": "tell me a one sentence story"}],
            stream=True)
    except Exception as e:
        assert 'Error while calling anthropic API: "Could not resolve authentication method.' in str(
            e)


"""
generate: Anthropic
"""


@patch.object(Anthropic, "messages", new_callable=PropertyMock)
def test_patch_anthropic_generate_Anthropic_API(mock_messages_prop):
    """
    This test is using the Athropic API directly but return a mocked response.
    """

    mock_messages = MagicMock()
    mock_messages.create.return_value = chat_completions_generate("anthropic")
    mock_messages_prop.return_value = mock_messages

    # Use Anthropic API

    client = Anthropic()
    response = client.messages.create(
        model="claude-sonnet-4-0",
        messages=[
            {"role": "user", "content": "tell me a one sentence story"}],
        max_tokens=100,
        stream=False
    )
    assert response.content[0].text == 'The last person on Earth sat alone in a room, then heard a knock at the door.'


@patch.object(Anthropic, "messages", new_callable=PropertyMock)
def test_patch_chatcompletions_anthropic_generate(mock_messages_prop):

    mock_messages = MagicMock()
    mock_messages.create.return_value = chat_completions_generate("anthropic")
    mock_messages_prop.return_value = mock_messages

    client_config = {
        "client_type": "anthropic",
        "model": "claude-sonnet-4-0",
    }
    client = OpenAI(client_config=client_config)
    response = client.chat.completions.create(
        model="claude-sonnet-4-0",
        messages=[{"role": "user", "content": "tell me a one sentence story"}],
        stream=False
    )
    assert response.choices[0].message.content == 'The last person on Earth sat alone in a room, then heard a knock at the door.'


"""
stream: Anthropic
"""


@patch.object(Anthropic, "messages", new_callable=PropertyMock)
def test_patch_anthropic_stream_Anthropic_API(mock_messages_prop):
    """
    This test is using the Athropic API directly but return a mocked response.
    """

    # Create a mock with a create() method that returns an iterable
    mock_messages = MagicMock()
    mock_messages.create.return_value = chat_completions_stream("anthropic")
    mock_messages_prop.return_value = mock_messages

    # Use Anthropic API

    client = Anthropic()
    response = client.messages.create(
        model="claude-sonnet-4-0",
        messages=[
            {"role": "user", "content": "tell me a one sentence story"}],
        max_tokens=100,
        stream=True
    )
    content = ""
    for chunk in response:
        if chunk.type == "content_block_delta":
            content += chunk.delta.text
    assert content == 'The old lighthouse keeper finally understood why the beacon had been calling to the sea every night for forty years when a glowing figure emerged from the waves and whispered, "Thank you for keeping your promise."'


@patch.object(Anthropic, "messages", new_callable=PropertyMock)
def test_patch_anthropic_stream_GAI_API(mock_messages_prop):
    """
    This test is using the same underlying Athropic API mocked response but called via the GAI client.
    """

    # Create a mock with a create() method that returns an iterable

    mock_messages = MagicMock()
    mock_messages.create.return_value = chat_completions_stream("anthropic")
    mock_messages_prop.return_value = mock_messages

    # Use GAI API

    client_config = {
        "client_type": "anthropic",
        "model": "claude-sonnet-4-0",
    }
    client = OpenAI(client_config=client_config)
    response = client.chat.completions.create(model="claude-sonnet-4-0",
                                              messages=[
                                                  {"role": "user", "content": "tell me a one sentence story"}],
                                              stream=True
                                              )
    content = ""
    for chunk in response:
        chunk = chunk.extract()
        if isinstance(chunk, str):
            content += chunk
    assert content == 'The old lighthouse keeper finally understood why the beacon had been calling to the sea every night for forty years when a glowing figure emerged from the waves and whispered, "Thank you for keeping your promise."'
