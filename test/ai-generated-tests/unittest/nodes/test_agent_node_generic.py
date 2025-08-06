"""
Test AgentNode generic functionality.
"""

import pytest
from unittest.mock import Mock, AsyncMock
from gai.nodes.agent_node import AgentNode
from gai.asm.agents import ToolUseAgent, ChatAgent
from gai.lib.config import GaiClientConfig
from gai.sessions import SessionManager
from gai.messages.dialogue import Dialogue


class TestAgentNodeGeneric:
    """Test that AgentNode works with different agent types."""

    @pytest.fixture
    def mock_session_mgr(self):
        """Mock session manager."""
        return Mock(spec=SessionManager)

    @pytest.fixture
    def mock_llm_config(self):
        """Mock LLM configuration."""
        return Mock(spec=GaiClientConfig)

    def test_agent_node_with_default_tool_use_agent(
        self, mock_session_mgr, mock_llm_config
    ):
        """Test AgentNode with default ChatAgent."""
        # Act
        agent_node = AgentNode(
            agent_name="test_agent",
            session_mgr=mock_session_mgr,
            llm_config=mock_llm_config,
        )

        # Assert
        assert agent_node.agent_class == ChatAgent
        assert agent_node.agent_name == "test_agent"
        assert agent_node.session_mgr == mock_session_mgr
        assert agent_node.llm_config == mock_llm_config

    def test_agent_node_with_explicit_tool_use_agent(
        self, mock_session_mgr, mock_llm_config
    ):
        """Test AgentNode with explicitly specified ToolUseAgent."""
        # Act
        agent_node = AgentNode(
            agent_name="test_agent",
            session_mgr=mock_session_mgr,
            llm_config=mock_llm_config,
            agent_class=ToolUseAgent,
        )

        # Assert
        assert agent_node.agent_class == ToolUseAgent

    def test_agent_node_with_chat_agent(self, mock_session_mgr, mock_llm_config):
        """Test AgentNode with ChatAgent."""
        # Act
        agent_node = AgentNode(
            agent_name="test_agent",
            session_mgr=mock_session_mgr,
            llm_config=mock_llm_config,
            agent_class=ChatAgent,
        )

        # Assert
        assert agent_node.agent_class == ChatAgent

    def test_agent_node_type_annotation(self, mock_session_mgr, mock_llm_config):
        """Test that type annotations work correctly."""
        # This test mainly ensures the generic type system works
        # Act
        tool_use_node: AgentNode[ToolUseAgent] = AgentNode(
            agent_name="tool_agent",
            session_mgr=mock_session_mgr,
            llm_config=mock_llm_config,
            agent_class=ToolUseAgent,
        )

        chat_node: AgentNode[ChatAgent] = AgentNode(
            agent_name="chat_agent",
            session_mgr=mock_session_mgr,
            llm_config=mock_llm_config,
            agent_class=ChatAgent,
        )

        # Assert
        assert tool_use_node.agent_class == ToolUseAgent
        assert chat_node.agent_class == ChatAgent
