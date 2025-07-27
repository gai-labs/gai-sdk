import sys
import pytest
from unittest.mock import MagicMock, patch, PropertyMock, AsyncMock

# Add the projects directory to the Python path to import gai modules
project_path = "/workspace/projects/gai-sdk/src"
if project_path not in sys.path:
    sys.path.insert(0, project_path)

# Add the llm test directory for mock data
mock_data_path = "/workspace/projects/gai-sdk/llm/test/unittest/gai/openai/mock_data"
if mock_data_path not in sys.path:
    sys.path.insert(0, mock_data_path)

from gai.asm.states.AnthropicToolUseState import AnthropicToolUseState
from gai.messages import Monologue
from gai.llm.openai import AsyncOpenAI


class MockMachine:
    """Mock state machine for testing"""

    def __init__(self):
        self.state_bag = {}
        self.state_history = [{"output": {"monologue": Monologue()}}]
        self.monologue = Monologue()
        self.state = "TOOL_USE"
        self.state_manifest = {
            "TOOL_USE": {
                "module_path": "gai.asm.states",
                "class_name": "AnthropicToolUseState",
                "title": "TOOL_USE",
            }
        }


class MockMcpClient:
    """Mock MCP client for testing"""

    def __init__(self):
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "current_time",
                    "description": "Get current time",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "format": {"type": "string"},
                            "timezone": {"type": "string"},
                        },
                    },
                },
            }
        ]

    async def list_tools(self):
        return self.tools

    async def call_tool(self, tool_name, **kwargs):
        # Mock tool result
        mock_result = MagicMock()
        mock_result.content = [MagicMock()]
        mock_result.content[0].text = f"Tool {tool_name} executed with {kwargs}"
        return mock_result


class TestAnthropicToolUseState:
    """Test cases for AnthropicToolUseState"""

    def setup_method(self):
        """Setup test environment"""
        self.machine = MockMachine()
        self.state = AnthropicToolUseState(self.machine)
        self.llm_config = {
            "client_type": "anthropic",
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 1000,
        }
        self.mcp_client = MockMcpClient()

        # Setup state input
        self.state.input = {
            "llm_config": self.llm_config,
            "mcp_client": self.mcp_client,
        }


@pytest.mark.asyncio
async def test_terminated_monologue_returns_none_streamer():
    """Test Case 1: When monologue is terminated, should return None streamer"""
    # Setup
    machine = MockMachine()
    machine.monologue.add_user_message("test")
    machine.monologue.add_assistant_message([{"text": "response", "type": "text"}])
    # Mock the terminated check
    machine.monologue.is_terminated = MagicMock(return_value=True)

    state = AnthropicToolUseState(machine)
    state.input = {
        "llm_config": {"client_type": "anthropic", "model": "claude-sonnet-4-20250514"},
        "mcp_client": MockMcpClient(),
    }

    # Execute
    await state.run_async()

    # Assert
    assert state.machine.state_bag["streamer"] is None
    machine.monologue.is_terminated.assert_called_once()


@pytest.mark.asyncio
async def test_user_input_pending_returns_none_streamer():
    """Test Case 2a: When LLM requests user input but no user message is provided, should return None streamer"""
    # Setup
    machine = MockMachine()
    machine.monologue.add_user_message("What time is it?")
    # Add assistant message with user_input tool call
    machine.monologue.add_assistant_message(
        [
            {"text": "Are you asking for local time or UTC?", "type": "text"},
            {"id": "tool_123", "name": "user_input", "input": {}, "type": "tool_use"},
        ]
    )

    # Mock the methods
    machine.monologue.is_terminated = MagicMock(return_value=False)
    machine.monologue.get_last_toolcalls = MagicMock(
        return_value=[
            {"tool_use_id": "tool_123", "tool_name": "user_input", "arguments": {}}
        ]
    )

    # No user_message provided in state_bag
    machine.state_bag["user_message"] = None

    state = AnthropicToolUseState(machine)
    state.input = {
        "llm_config": {"client_type": "anthropic", "model": "claude-sonnet-4-20250514"},
        "mcp_client": MockMcpClient(),
    }

    # Execute
    await state.run_async()

    # Assert
    assert state.machine.state_bag["streamer"] is None
    machine.monologue.is_terminated.assert_called_once()
    machine.monologue.get_last_toolcalls.assert_called_once()


@pytest.mark.asyncio
async def test_make_user_input_tool_result():
    """Test _make_user_input_tool_result method creates pseudo tool result"""
    # Setup
    machine = MockMachine()
    machine.state_bag["user_message"] = "Use SGT timezone"

    state = AnthropicToolUseState(machine)

    # Mock tool calls with user_input
    last_tool_calls = [
        {"tool_use_id": "tool_123", "tool_name": "user_input", "arguments": {}}
    ]

    # Execute
    result = state._make_user_input_tool_result(last_tool_calls)

    # Assert
    expected_result = {
        "type": "tool_result",
        "tool_use_id": "tool_123",
        "content": "Use SGT timezone",
    }
    assert result == expected_result


@pytest.mark.asyncio
async def test_make_user_input_tool_result_no_user_input_tool():
    """Test _make_user_input_tool_result returns None when no user_input tool found"""
    # Setup
    machine = MockMachine()
    state = AnthropicToolUseState(machine)

    # Mock tool calls without user_input
    last_tool_calls = [
        {"tool_use_id": "tool_123", "tool_name": "current_time", "arguments": {}}
    ]

    # Execute
    result = state._make_user_input_tool_result(last_tool_calls)

    # Assert
    assert result is None


@pytest.mark.asyncio
async def test_use_tool_method():
    """Test _use_tool method executes MCP tools and formats results"""
    # Setup
    machine = MockMachine()
    mcp_client = MockMcpClient()

    state = AnthropicToolUseState(machine)
    state.input = {"mcp_client": mcp_client}

    # Mock tool calls
    last_tool_calls = [
        {
            "tool_use_id": "tool_123",
            "tool_name": "current_time",
            "arguments": {"format": "YYYY-MM-DD"},
        }
    ]

    # Execute
    result = await state._use_tool(last_tool_calls)

    # Assert
    expected_result = [
        {
            "type": "tool_result",
            "tool_use_id": "tool_123",
            "content": "Tool current_time executed with {'format': 'YYYY-MM-DD'}",
        }
    ]

    assert result == expected_result
    assert machine.state_bag["tool_results"] == expected_result


@pytest.mark.asyncio
async def test_use_tool_with_multiple_tools():
    """Test _use_tool method with multiple tool calls"""
    # Setup
    machine = MockMachine()
    mcp_client = MockMcpClient()

    state = AnthropicToolUseState(machine)
    state.input = {"mcp_client": mcp_client}

    # Mock multiple tool calls
    last_tool_calls = [
        {
            "tool_use_id": "tool_123",
            "tool_name": "current_time",
            "arguments": {"format": "YYYY-MM-DD"},
        },
        {
            "tool_use_id": "tool_456",
            "tool_name": "current_time",
            "arguments": {"timezone": "SGT"},
        },
    ]

    # Execute
    result = await state._use_tool(last_tool_calls)

    # Assert
    assert len(result) == 2
    assert result[0]["tool_use_id"] == "tool_123"
    assert result[1]["tool_use_id"] == "tool_456"
    assert all(item["type"] == "tool_result" for item in result)


@pytest.mark.asyncio
async def test_use_tool_handles_mcp_client_error():
    """Test _use_tool method handles MCP client errors properly"""
    # Setup
    machine = MockMachine()
    mcp_client = MockMcpClient()

    # Mock MCP client to raise an error
    mcp_client.call_tool = AsyncMock(side_effect=Exception("MCP error"))

    state = AnthropicToolUseState(machine)
    state.input = {"mcp_client": mcp_client}

    # Mock tool calls
    last_tool_calls = [
        {
            "tool_use_id": "tool_123",
            "tool_name": "current_time",
            "arguments": {"format": "YYYY-MM-DD"},
        }
    ]

    # Execute and Assert
    with pytest.raises(Exception, match="MCP error"):
        await state._use_tool(last_tool_calls)


@pytest.mark.asyncio
async def test_use_tool_handles_different_content_formats():
    """Test _use_tool method handles different MCP result content formats"""
    # Setup
    machine = MockMachine()

    # Mock MCP client with different result format
    mcp_client = MagicMock()
    mcp_client.call_tool = AsyncMock()

    # Test case 1: result.content is a list with text attribute
    mock_result = MagicMock()
    mock_result.content = [MagicMock()]
    mock_result.content[0].text = "Time result"
    mcp_client.call_tool.return_value = mock_result

    state = AnthropicToolUseState(machine)
    state.input = {"mcp_client": mcp_client}

    last_tool_calls = [
        {"tool_use_id": "tool_123", "tool_name": "current_time", "arguments": {}}
    ]

    result = await state._use_tool(last_tool_calls)

    assert result[0]["content"] == "Time result"

    # Test case 2: result.content without text attribute
    mock_result.content[0] = "Direct string content"
    result = await state._use_tool(last_tool_calls)

    assert result[0]["content"] == "Direct string content"

    # Test case 3: result without content attribute
    mock_result = "Direct result"
    mcp_client.call_tool.return_value = mock_result

    result = await state._use_tool(last_tool_calls)

    assert result[0]["content"] == "Direct result"


class MockAsyncStreamResponse:
    """Mock for async streaming response"""

    def __init__(self):
        self.chunks = ["Hello", " world", [{"type": "tool_use", "id": "123"}]]
        self.index = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self.index >= len(self.chunks):
            raise StopAsyncIteration
        chunk = self.chunks[self.index]
        self.index += 1

        if isinstance(chunk, str):
            # Mock string chunk
            mock_chunk = MagicMock()
            mock_chunk.extract.return_value = chunk
            return mock_chunk
        else:
            # Mock tool call chunk
            mock_chunk = MagicMock()
            mock_chunk.extract.return_value = chunk
            return mock_chunk


@pytest.mark.asyncio
async def test_user_input_case_2b_creates_streamer():
    """Test Case 2b: User input provided creates streamer and processes pseudo tool result"""
    # Setup
    machine = MockMachine()
    machine.monologue.is_terminated = MagicMock(return_value=False)
    machine.monologue.get_last_toolcalls = MagicMock(
        return_value=[
            {"tool_use_id": "tool_123", "tool_name": "user_input", "arguments": {}}
        ]
    )
    machine.monologue.add_user_message = MagicMock()
    machine.monologue.add_assistant_message = MagicMock()
    machine.monologue.list_chat_messages = MagicMock(
        return_value=[{"role": "user", "content": "What time is it?"}]
    )

    # User message provided
    machine.state_bag["user_message"] = "Use SGT timezone"

    state = AnthropicToolUseState(machine)
    state.input = {
        "llm_config": {"client_type": "anthropic", "model": "claude-sonnet-4-20250514"},
        "mcp_client": MockMcpClient(),
    }

    # Mock the LLM client to avoid actual API calls
    mock_response = MockAsyncStreamResponse()

    with patch("gai.llm.openai.AsyncOpenAI") as mock_llm_class:
        mock_llm_instance = MagicMock()
        mock_llm_instance.chat.completions.create = AsyncMock(
            return_value=mock_response
        )
        mock_llm_class.return_value = mock_llm_instance

        # Execute
        await state.run_async()

        # Assert pseudo tool result was created and added to monologue
        machine.monologue.add_user_message.assert_called_once()
        added_content = machine.monologue.add_user_message.call_args[1]["content"]
        expected_tool_result = [
            {
                "type": "tool_result",
                "tool_use_id": "tool_123",
                "content": "Use SGT timezone",
            }
        ]
        assert added_content == expected_tool_result

        # Assert streamer was created
        assert state.machine.state_bag["streamer"] is not None


@pytest.mark.asyncio
async def test_normal_tool_execution_creates_streamer():
    """Test Case 3: Normal MCP tool execution creates streamer"""
    # Setup
    machine = MockMachine()
    machine.monologue.is_terminated = MagicMock(return_value=False)
    machine.monologue.get_last_toolcalls = MagicMock(
        return_value=[
            {
                "tool_use_id": "tool_123",
                "tool_name": "current_time",
                "arguments": {"format": "YYYY-MM-DD"},
            }
        ]
    )
    machine.monologue.add_user_message = MagicMock()
    machine.monologue.add_assistant_message = MagicMock()
    machine.monologue.list_chat_messages = MagicMock(
        return_value=[{"role": "user", "content": "What time is it?"}]
    )

    state = AnthropicToolUseState(machine)
    state.input = {
        "llm_config": {"client_type": "anthropic", "model": "claude-sonnet-4-20250514"},
        "mcp_client": MockMcpClient(),
    }

    # Mock the LLM client to avoid actual API calls
    mock_response = MockAsyncStreamResponse()

    with patch("gai.llm.openai.AsyncOpenAI") as mock_llm_class:
        mock_llm_instance = MagicMock()
        mock_llm_instance.chat.completions.create = AsyncMock(
            return_value=mock_response
        )
        mock_llm_class.return_value = mock_llm_instance

        # Execute
        await state.run_async()

        # Assert tool was executed
        assert "tool_results" in machine.state_bag
        tool_results = machine.state_bag["tool_results"]
        assert len(tool_results) == 1
        assert tool_results[0]["tool_use_id"] == "tool_123"
        assert tool_results[0]["type"] == "tool_result"
        assert "Tool current_time executed" in tool_results[0]["content"]

        # Assert tool results were added to monologue
        machine.monologue.add_user_message.assert_called_once()
        added_content = machine.monologue.add_user_message.call_args[1]["content"]
        assert added_content == tool_results

        # Assert streamer was created
        assert state.machine.state_bag["streamer"] is not None


if __name__ == "__main__":
    pytest.main([__file__])
