import os
import sys
import json
import pytest
from unittest.mock import MagicMock, patch, PropertyMock, AsyncMock
from anthropic import Anthropic

# Add the mock data directory to path
mock_dir = os.path.join(os.path.dirname(__file__), "..", "..", "..", "projects", "gai-sdk", "llm", "test", "unittest", "gai", "openai", "mock_data")
if mock_dir not in sys.path:
    sys.path.insert(0, mock_dir)

from mock_openai_patch import chat_completions_generate, chat_completions_stream, chat_completions_streaming_toolcall

# Import the classes we need to test
from gai.asm.agents.tool_use_agent import ToolUseAgent
from gai.lib.config import GaiClientConfig
from gai.mcp.client import McpAggregatedClient
from gai.messages import Monologue
from gai.lib.logging import getLogger

logger = getLogger(__name__)


class TestToolUseAgent:
    """Test suite for ToolUseAgent class."""

    @pytest.fixture
    def mock_llm_config(self):
        """Create a mock LLM configuration."""
        return GaiClientConfig(
            client_type="anthropic",
            model="claude-sonnet-4-0",
            extra={
                "max_tokens": 32000,
                "temperature": 0.7,
                "top_p": 0.95,
                "tools": True,
                "stream": True,
            }
        )

    @pytest.fixture
    def mock_mcp_client(self):
        """Create a mock MCP aggregated client."""
        client = MagicMock(spec=McpAggregatedClient)
        client.list_tools = AsyncMock(return_value=[
            {
                "name": "search",
                "description": "Search for information",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "search_query": {"type": "string"}
                    },
                    "required": ["search_query"]
                }
            }
        ])
        return client

    @pytest.fixture
    def mock_monologue(self):
        """Create a mock monologue."""
        monologue = MagicMock(spec=Monologue)
        monologue.list_messages.return_value = []
        monologue.reset.return_value = None
        monologue.pop.return_value = None
        return monologue

    def test_tool_use_agent_initialization(self, mock_llm_config, mock_mcp_client, mock_monologue):
        """Test that ToolUseAgent initializes correctly."""
        agent = ToolUseAgent(
            agent_name="TestAgent",
            llm_config=mock_llm_config,
            aggregated_client=mock_mcp_client,
            monologue=mock_monologue
        )
        
        # Test that the agent was created successfully
        assert isinstance(agent, ToolUseAgent)
        assert agent.monologue == mock_monologue
        assert agent.fsm is not None
        # The FSM starts in INIT state
        assert hasattr(agent.fsm, 'state')

    def test_has_message_predicate_with_no_user_message(self, mock_llm_config, mock_mcp_client, mock_monologue):
        """Test has_message predicate when no user message is provided."""
        agent = ToolUseAgent(
            agent_name="TestAgent",
            llm_config=mock_llm_config,
            aggregated_client=mock_mcp_client,
            monologue=mock_monologue
        )
        
        # Create a mock state
        mock_state = MagicMock()
        mock_state.machine.state_bag = {}
        
        # Call the predicate
        result = agent.has_message(mock_state)
        
        # Should return False when no user_message
        assert result is False
        assert mock_state.machine.state_bag["predicate_result"] is False
        assert mock_state.machine.state_bag["streamer"] is None

    def test_has_message_predicate_with_user_message(self, mock_llm_config, mock_mcp_client, mock_monologue):
        """Test has_message predicate when user message is provided."""
        agent = ToolUseAgent(
            agent_name="TestAgent",
            llm_config=mock_llm_config,
            aggregated_client=mock_mcp_client,
            monologue=mock_monologue
        )
        
        # Create a mock state with user_message
        mock_state = MagicMock()
        mock_state.machine.state_bag = {"user_message": "Hello, world!"}
        
        # Call the predicate
        result = agent.has_message(mock_state)
        
        # Should return True when user_message exists
        assert result is True
        assert mock_state.machine.state_bag["predicate_result"] is True
        assert mock_state.machine.state_bag["streamer"] is None