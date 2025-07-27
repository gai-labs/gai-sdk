import pytest
import pytest_asyncio
import os
import sys
import json
from unittest.mock import MagicMock, patch, PropertyMock, AsyncMock
from anthropic import Anthropic
from anthropic.types import MessageStreamEvent
from pydantic import TypeAdapter
from typing import List

# Add the mock data directory to path
mock_dir = "/workspace/projects/gai-sdk/llm/test/unittest/gai/openai/mock_data"
if mock_dir not in sys.path:
    sys.path.insert(0, mock_dir)

# Import the mock data functions
from mock_openai_patch import chat_completions_generate, chat_completions_stream, chat_completions_streaming_toolcall

# Add main source to path for imports
main_src = "/workspace/projects/gai-sdk/src"
if main_src not in sys.path:
    sys.path.insert(0, main_src)

lib_src = "/workspace/projects/gai-sdk/lib/src"
if lib_src not in sys.path:
    sys.path.insert(0, lib_src)

llm_src = "/workspace/projects/gai-sdk/llm/src"
if llm_src not in sys.path:
    sys.path.insert(0, llm_src)

from gai.asm.states.AnthropicChatState import AnthropicChatState
from gai.asm.base import StateBase
from gai.messages.monologue import Monologue


class MockMachine:
    """Mock state machine for testing AnthropicChatState"""
    
    def __init__(self):
        self.user_message = None
        self.agent_name = "TestAgent"
        self.monologue = Monologue()
        self.state_bag = {}
        self.state_history = [
            {"output": {"monologue": self.monologue}},
            {"output": {"monologue": self.monologue}}
        ]
        self.state = "CHAT(AnthropicChatState)"
        self.state_manifest = {
            "CHAT": {
                "module_path": "gai.asm.states",
                "class_name": "AnthropicChatState",
                "title": "CHAT",
                "input_data": {
                    "llm_config": {"type": "state_bag", "dependency": "llm_config"},
                    "mcp_client": {"type": "state_bag", "dependency": "mcp_client"},
                },
                "output_data": ["streamer", "get_assistant_message"],
            }
        }
    
    def copy_state_bag(self, keys):
        """Mock method to copy state bag values"""
        return {key: self.state_bag.get(key) for key in keys}


class MockMcpClient:
    """Mock MCP client for testing tool integration"""
    
    async def list_tools(self):
        return [
            {
                "name": "google",
                "description": "Search Google for information",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "search_query": {
                            "type": "string",
                            "description": "The search query to send to Google"
                        }
                    },
                    "required": ["search_query"]
                }
            }
        ]


class MockAsyncOpenAI:
    """Mock AsyncOpenAI client for testing"""
    
    def __init__(self, config):
        self.config = config
        self.chat = MockChatCompletions()


class MockChatCompletions:
    """Mock chat completions for AsyncOpenAI"""
    
    def __init__(self):
        pass
    
    async def create(self, model, messages, tools=None, stream=False):
        if stream:
            return MockStreamResponse()
        else:
            return MockNonStreamResponse()


class MockStreamResponse:
    """Mock streaming response"""
    
    def __aiter__(self):
        return self
    
    async def __anext__(self):
        # Simulate streaming chunks
        for chunk in ["Hello", " ", "world", "!"]:
            mock_chunk = MagicMock()
            mock_chunk.extract.return_value = chunk
            yield mock_chunk
        # Simulate end of stream with final response
        final_response = [{"text": "Hello world!", "type": "text"}]
        yield final_response
        raise StopAsyncIteration


class MockNonStreamResponse:
    """Mock non-streaming response"""
    
    def extract(self):
        return "Hello world!"


class TestAnthropicChatState:
    """Test suite for AnthropicChatState"""

    def setup_method(self):
        """Set up test fixtures"""
        self.mock_machine = MockMachine()
        self.mock_machine.user_message = "Tell me a one sentence story."
        
        # Mock LLM config
        self.llm_config = {
            "client_type": "anthropic",
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 32000,
            "temperature": 0.7,
            "top_p": 0.95,
        }

    def test_anthropic_chat_state_initialization(self):
        """Test that AnthropicChatState initializes properly"""
        state = AnthropicChatState(self.mock_machine)
        assert isinstance(state, StateBase)
        assert state.machine == self.mock_machine

    @pytest.mark.asyncio
    async def test_missing_user_message_raises_exception(self):
        """Test that missing user message raises an exception"""
        self.mock_machine.user_message = None
        state = AnthropicChatState(self.mock_machine)
        state.input = {"llm_config": self.llm_config, "step": 1}
        
        with pytest.raises(Exception, match="user_message is missing"):
            await state.run_async()

    @pytest.mark.asyncio
    @patch('gai.llm.openai.async_patch.async_anthropic_create')
    async def test_basic_chat_flow_generate_text(self, mock_anthropic_create):
        """Test basic chat flow with text generation using mocked Anthropic response"""
        
        # Create a mock async generator that yields string chunks and final response
        async def mock_stream():
            # Yield string chunks first
            yield "The"
            yield " last"
            yield " person"
            yield " on"
            yield " Earth"
            yield " sat"
            yield " alone"
            yield " in"
            yield " a"
            yield " room"
            yield ","
            yield " then"
            yield " heard"
            yield " a"
            yield " knock"
            yield " at"
            yield " the"
            yield " door"
            yield "."
            # Yield final response (non-string)
            yield [{"text": "The last person on Earth sat alone in a room, then heard a knock at the door.", "type": "text"}]
        
        mock_anthropic_create.return_value = mock_stream()
        
        # Setup state
        state = AnthropicChatState(self.mock_machine)
        state.input = {"llm_config": self.llm_config, "step": 1}
        
        # Run the state
        await state.run_async()
        
        # Verify streamer was created
        assert "streamer" in self.mock_machine.state_bag
        streamer = self.mock_machine.state_bag["streamer"]
        
        # Collect streamed content
        collected_text = ""
        final_response = None
        
        async for chunk in streamer:
            if isinstance(chunk, str):
                collected_text += chunk
            else:
                final_response = chunk
                break
        
        # Verify the text was properly streamed (note: additional \n is added by the logic)
        expected_text = "The last person on Earth sat alone in a room, then heard a knock at the door.\n"
        assert collected_text == expected_text
        
        # Verify final response structure
        assert final_response is not None
        assert isinstance(final_response, list)
        assert final_response[0]["text"] == "The last person on Earth sat alone in a room, then heard a knock at the door."
        assert final_response[0]["type"] == "text"
        
        # Verify get_assistant_message function was created
        assert "get_assistant_message" in self.mock_machine.state_bag
        assert callable(self.mock_machine.state_bag["get_assistant_message"])

    @pytest.mark.asyncio
    @patch('gai.llm.openai.async_patch.async_anthropic_create')
    async def test_chat_flow_with_tools(self, mock_anthropic_create):
        """Test chat flow with tool calls using mocked Anthropic streaming response"""
        
        # Create a mock async generator that simulates the tool call flow
        async def mock_tool_stream():
            # First yield text chunks
            yield "I"
            yield "'ll help you find the current time in Singapore."
            yield "\\n"
            # Then yield the final response with tool call
            yield [
                {"text": "I'll help you find the current time in Singapore.", "type": "text"},
                {
                    "id": "toolu_01PsZcVuuQAU62ReyTuKYMyH",
                    "input": {"search_query": "current time in Singapore"},
                    "name": "google",
                    "type": "tool_use"
                }
            ]
        
        mock_anthropic_create.return_value = mock_tool_stream()
        
        # Setup state with MCP client
        mock_mcp_client = MockMcpClient()
        state = AnthropicChatState(self.mock_machine)
        state.input = {
            "llm_config": self.llm_config,
            "mcp_client": mock_mcp_client,
            "step": 1
        }
        
        # Set user message for tool scenario
        self.mock_machine.user_message = "What time is it in Singapore?"
        
        # Run the state
        await state.run_async()
        
        # Verify streamer was created
        assert "streamer" in self.mock_machine.state_bag
        streamer = self.mock_machine.state_bag["streamer"]
        
        # Collect streamed content
        collected_text = ""
        final_response = None
        
        async for chunk in streamer:
            if isinstance(chunk, str):
                collected_text += chunk
            else:
                final_response = chunk
                break
        
        # Verify the text was properly streamed (note: additional \n is added by the logic)
        expected_text = "I'll help you find the current time in Singapore.\\n\n"
        assert collected_text == expected_text
        
        # Verify final response contains tool call
        assert final_response is not None
        assert isinstance(final_response, list)
        assert len(final_response) == 2
        
        # Check text block
        assert final_response[0]["type"] == "text"
        assert "Singapore" in final_response[0]["text"]
        
        # Check tool call block
        tool_call = final_response[1]
        assert tool_call["type"] == "tool_use"
        assert tool_call["name"] == "google"
        assert "search_query" in tool_call["input"]
        assert "Singapore" in tool_call["input"]["search_query"]

    @pytest.mark.asyncio
    async def test_user_input_interrupt_flow(self):
        """Test that state exits early when user_input tool was previously called"""
        
        # Setup mock machine with previous tool calls containing user_input
        self.mock_machine.monologue.get_last_toolcalls = MagicMock(return_value=[
            {"tool_name": "user_input", "result": "pending"}
        ])
        
        state = AnthropicChatState(self.mock_machine)
        state.input = {"llm_config": self.llm_config, "step": 1}
        
        # Run the state
        await state.run_async()
        
        # Verify that streamer is set to None (early exit)
        assert "streamer" in self.mock_machine.state_bag
        assert self.mock_machine.state_bag["streamer"] is None

    @pytest.mark.asyncio
    async def test_no_user_input_interrupt_when_no_tool_calls(self):
        """Test that normal flow continues when no previous user_input tool calls"""
        
        # Setup mock machine with no previous tool calls
        self.mock_machine.monologue.get_last_toolcalls = MagicMock(return_value=None)
        
        state = AnthropicChatState(self.mock_machine)
        state.input = {"llm_config": self.llm_config, "step": 1}
        
        # Mock AsyncOpenAI to avoid actual API calls
        with patch('gai.llm.openai.AsyncOpenAI') as mock_async_openai:
            mock_client_instance = MagicMock()
            mock_async_openai.return_value = mock_client_instance
            
            # Create a simple mock response
            async def mock_simple_stream():
                yield "Test response"
                yield [{"text": "Test response", "type": "text"}]
            
            mock_client_instance.chat.completions.create.return_value = mock_simple_stream()
            
            # Run the state
            await state.run_async()
            
            # Verify that streamer was created (not set to None)
            assert "streamer" in self.mock_machine.state_bag
            assert self.mock_machine.state_bag["streamer"] is not None

    @pytest.mark.asyncio
    async def test_system_message_creation(self):
        """Test that system message is properly formatted with agent name"""
        
        # Custom agent name
        self.mock_machine.agent_name = "CustomAgent"
        custom_message = "Hello, how are you?"
        self.mock_machine.user_message = custom_message
        
        state = AnthropicChatState(self.mock_machine)
        state.input = {"llm_config": self.llm_config, "step": 1}
        
        # Mock the monologue and AsyncOpenAI
        with patch('gai.llm.openai.AsyncOpenAI') as mock_async_openai:
            mock_client_instance = MagicMock()
            mock_async_openai.return_value = mock_client_instance
            
            # Mock the stream response
            async def mock_stream():
                yield "Response"
                yield [{"text": "Response", "type": "text"}]
            
            mock_client_instance.chat.completions.create.return_value = mock_stream()
            
            # Mock the monologue methods
            self.mock_machine.monologue.add_user_message = MagicMock()
            self.mock_machine.monologue.list_chat_messages = MagicMock(return_value=[])
            self.mock_machine.monologue.add_assistant_message = MagicMock()
            
            # Run the state
            await state.run_async()
            
            # Verify that add_user_message was called with the system message
            self.mock_machine.monologue.add_user_message.assert_called_once()
            call_args = self.mock_machine.monologue.add_user_message.call_args
            
            # Check that the system message contains agent name and user message
            system_message = call_args[1]['content']  # keyword argument 'content'
            assert "CustomAgent" in system_message
            assert custom_message in system_message
            assert "Do not refer to yourself as an AI or a bot" in system_message

    @pytest.mark.asyncio
    @patch('gai.llm.openai.async_patch.async_anthropic_create')
    async def test_thinking_message_for_tool_only_response(self, mock_anthropic_create):
        """Test that 'Thinking...' is displayed when LLM returns tool calls without text"""
        
        # Create a mock async generator that yields only tool calls (no text)
        async def mock_tool_only_stream():
            # Yield tool call directly without any text chunks
            yield [
                {
                    "id": "toolu_123",
                    "input": {"query": "test"},
                    "name": "test_tool",
                    "type": "tool_use"
                }
            ]
        
        mock_anthropic_create.return_value = mock_tool_only_stream()
        
        # Setup state
        state = AnthropicChatState(self.mock_machine)
        state.input = {"llm_config": self.llm_config, "step": 1}
        
        # Mock monologue methods
        self.mock_machine.monologue.add_assistant_message = MagicMock()
        self.mock_machine.monologue.list_chat_messages = MagicMock(return_value=[{"role": "user", "content": "test"}])
        
        # Run the state
        await state.run_async()
        
        # Verify streamer was created
        assert "streamer" in self.mock_machine.state_bag
        streamer = self.mock_machine.state_bag["streamer"]
        
        # Collect streamed content
        collected_chunks = []
        async for chunk in streamer:
            collected_chunks.append(chunk)
        
        # Verify that "Thinking..." was yielded first
        assert len(collected_chunks) >= 2
        assert collected_chunks[0] == "Thinking...\n"
        
        # Verify that the tool call was yielded
        final_chunk = collected_chunks[-1]
        assert isinstance(final_chunk, list)
        assert final_chunk[0]["type"] == "tool_use"

    @pytest.mark.asyncio
    async def test_empty_messages_raises_exception(self):
        """Test that empty messages list raises ValueError"""
        
        state = AnthropicChatState(self.mock_machine)
        state.input = {"llm_config": self.llm_config, "step": 1}
        
        # Mock AsyncOpenAI
        with patch('gai.llm.openai.AsyncOpenAI') as mock_async_openai:
            mock_client_instance = MagicMock()
            mock_async_openai.return_value = mock_client_instance
            
            # Mock empty messages list
            self.mock_machine.monologue.add_user_message = MagicMock()
            self.mock_machine.monologue.list_chat_messages = MagicMock(return_value=[])
            
            # Create a mock async generator that simulates the error condition
            async def mock_error_stream():
                raise ValueError("Cannot pass empty messages to LLM")
            
            mock_client_instance.chat.completions.create.return_value = mock_error_stream()
            
            # Run the state
            await state.run_async()
            
            # Get the streamer and try to iterate
            streamer = self.mock_machine.state_bag["streamer"]
            
            with pytest.raises(ValueError, match="Cannot pass empty messages to LLM"):
                async for chunk in streamer:
                    pass

    @pytest.mark.asyncio
    async def test_state_history_update(self):
        """Test that state history is properly updated with monologue"""
        
        state = AnthropicChatState(self.mock_machine)
        state.input = {"llm_config": self.llm_config, "step": 1}
        
        # Mock monologue methods
        self.mock_machine.monologue.add_user_message = MagicMock()
        self.mock_machine.monologue.list_chat_messages = MagicMock(return_value=[{"role": "user", "content": "test"}])
        self.mock_machine.monologue.add_assistant_message = MagicMock()
        self.mock_machine.monologue.copy = MagicMock(return_value="copied_monologue")
        
        # Mock at the async_anthropic_create level
        with patch('gai.llm.openai.async_patch.async_anthropic_create') as mock_anthropic_create:
            async def mock_stream():
                yield "Test"
                yield [{"text": "Test", "type": "text"}]
            
            mock_anthropic_create.return_value = mock_stream()
            
            # Run the state
            await state.run_async()
            
            # Get the streamer and consume it
            streamer = self.mock_machine.state_bag["streamer"]
            async for chunk in streamer:
                pass  # Consume all chunks
            
            # Verify that state history was updated
            assert self.mock_machine.state_history[-1]["output"]["monologue"] == "copied_monologue"
            self.mock_machine.monologue.copy.assert_called_once()

if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])