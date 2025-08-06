"""
Test AgentNode runtime functionality with different agent types.
"""
import pytest
from unittest.mock import Mock, AsyncMock, patch
from gai.nodes.agent_node import AgentNode
from gai.asm.agents import ToolUseAgent, ChatAgent
from gai.lib.config import GaiClientConfig
from gai.sessions import SessionManager


class TestAgentNodeRuntime:
    """Test that AgentNode instantiates different agent types correctly at runtime."""
    
    @pytest.fixture
    def mock_session_mgr(self):
        """Mock session manager."""
        return Mock(spec=SessionManager)
    
    @pytest.fixture  
    def mock_llm_config(self):
        """Mock LLM configuration."""
        return Mock(spec=GaiClientConfig)
    
    def test_creates_tool_use_agent_instance(self, mock_session_mgr, mock_llm_config):
        """Test that AgentNode creates ToolUseAgent instances correctly at runtime."""
        # Arrange
        agent_node = AgentNode(
            agent_name="test_agent",
            session_mgr=mock_session_mgr,
            llm_config=mock_llm_config,
            agent_class=ToolUseAgent
        )
        
        # Act - simulate agent creation as done in _input_handler
        agent_kwargs = {
            "agent_name": agent_node.agent_name,
            "llm_config": agent_node.llm_config,
        }
        if agent_node.aggregated_client is not None:
            agent_kwargs["aggregated_client"] = agent_node.aggregated_client
        if agent_node.monologue is not None:
            agent_kwargs["monologue"] = agent_node.monologue
            
        agent_instance = agent_node.agent_class(**agent_kwargs)
        
        # Assert
        assert isinstance(agent_instance, ToolUseAgent)
        assert agent_instance.agent_name == "test_agent"
        assert agent_instance.llm_config == mock_llm_config
    
    def test_creates_chat_agent_instance(self, mock_session_mgr, mock_llm_config):
        """Test that AgentNode creates ChatAgent instances correctly at runtime."""
        # Arrange
        agent_node = AgentNode(
            agent_name="chat_agent",
            session_mgr=mock_session_mgr,
            llm_config=mock_llm_config,
            agent_class=ChatAgent
        )
        
        # Act - simulate agent creation
        agent_kwargs = {
            "agent_name": agent_node.agent_name,
            "llm_config": agent_node.llm_config,
        }
        if agent_node.aggregated_client is not None:
            agent_kwargs["aggregated_client"] = agent_node.aggregated_client
        if agent_node.monologue is not None:
            agent_kwargs["monologue"] = agent_node.monologue
            
        agent_instance = agent_node.agent_class(**agent_kwargs)
        
        # Assert
        assert isinstance(agent_instance, ChatAgent)
        assert agent_instance.agent_name == "chat_agent"
        assert agent_instance.llm_config == mock_llm_config
    
    def test_agent_class_flexibility(self, mock_session_mgr, mock_llm_config):
        """Test that different agent classes can be used."""
        # Test with ToolUseAgent
        tool_agent_node = AgentNode(
            agent_name="tool_agent",
            session_mgr=mock_session_mgr,
            llm_config=mock_llm_config,
            agent_class=ToolUseAgent
        )
        assert tool_agent_node.agent_class == ToolUseAgent
        
        # Test with ChatAgent
        chat_agent_node = AgentNode(
            agent_name="chat_agent",
            session_mgr=mock_session_mgr,
            llm_config=mock_llm_config,
            agent_class=ChatAgent
        )
        assert chat_agent_node.agent_class == ChatAgent
        
        # Test with custom agent class (mock)
        class CustomAgent:
            def __init__(self, agent_name, llm_config, **kwargs):
                self.agent_name = agent_name
                self.llm_config = llm_config
        
        custom_agent_node = AgentNode(
            agent_name="custom_agent",
            session_mgr=mock_session_mgr,
            llm_config=mock_llm_config,
            agent_class=CustomAgent
        )
        assert custom_agent_node.agent_class == CustomAgent