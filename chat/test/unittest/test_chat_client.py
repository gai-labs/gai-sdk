import pytest
from unittest.mock import patch, MagicMock
from gai.chat.client import ChatClient, attach_extractor
from openai.types.chat.chat_completion import ChatCompletion
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk


@pytest.fixture
def mock_config():
    return {
        "client_type": "gai",
        "url": "http://fake-server/chat"
    }


def test_chat_non_stream_with_content(mock_config):
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "created": 123456,
        "model": "fake-model",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "Hello there!",
                "tool_calls": None
            },
            "finish_reason": "stop"
        }]
    }

    with patch("gai.chat.client.chat_client.http_post", return_value=mock_response):
        client = ChatClient(config=mock_config)
        response = client.chat(model="fake-model", messages=[{"role": "user", "content": "Hello"}], stream=False)
        assert callable(response.extract)
        assert response.extract() == {
            "type": "content",
            "content": "Hello there!"
        }


def test_chat_non_stream_with_toolcall(mock_config):
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "created": 123456,
        "model": "fake-model",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": "toolcall-abc",
                    "type": "function",
                    "function": {
                        "name": "my_func",
                        "arguments": '{"param": "value"}'
                    }
                }]
            },
            "finish_reason": "stop"
        }]
    }

    with patch("gai.chat.client.chat_client.http_post", return_value=mock_response):
        client = ChatClient(config=mock_config)
        response = client.chat(model="fake-model", messages=[{"role": "user", "content": "Hello"}], stream=False)
        assert callable(response.extract)
        assert response.extract() == {
            "type": "function",
            "name": "my_func",
            "arguments": '{"param": "value"}'
        }


