import os
import sys
import json
import pytest
from anthropic.types import MessageStreamEvent
from pydantic import TypeAdapter
from typing import List
from unittest.mock import MagicMock, patch, PropertyMock, AsyncMock
from gai.lib.tests import get_local_datadir
from gai.asm.agents.tool_use_agent import ToolUseAgent
from gai.lib.config import GaiClientConfig
from gai.mcp.client import McpAggregatedClient
from gai.messages import Monologue
from gai.lib.logging import getLogger


# Add the mock data directory to path
mock_dir = os.path.join(
    os.path.dirname(__file__),
    "..",
    "..",
    "..",
    "projects",
    "gai-sdk",
    "llm",
    "test",
    "unittest",
    "gai",
    "openai",
    "mock_data",
)
if mock_dir not in sys.path:
    sys.path.insert(0, mock_dir)

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
            },
        )

    @pytest.fixture
    def mock_mcp_client(self):
        """Create a mock MCP aggregated client."""
        client = MagicMock(spec=McpAggregatedClient)
        client.list_tools = AsyncMock(
            return_value=[
                {
                    "type": "function",
                    "function": {
                        "name": "search",
                        "description": "Search for information",
                        "input_schema": {
                            "type": "object",
                            "properties": {"search_query": {"type": "string"}},
                            "required": ["search_query"],
                        },
                    },
                }
            ]
        )
        return client

    @pytest.fixture
    def mock_monologue(self):
        """Create a mock monologue."""
        monologue = MagicMock(spec=Monologue)
        monologue.list_messages.return_value = []
        monologue.reset.return_value = None
        monologue.pop.return_value = None
        return monologue

    @pytest.fixture
    def tmp_file_monologue(self):
        """Create a temporary file monologue"""
        import tempfile
        from gai.messages import FileMonologue

        temp_file_path = None
        with tempfile.NamedTemporaryFile(delete=False, suffix=".log") as temp_file:
            temp_file_path = temp_file.name

        monologue = FileMonologue(file_path=temp_file_path)
        monologue.reset()
        return monologue

    def test_tool_use_agent_initialization(
        self, mock_llm_config, mock_mcp_client, mock_monologue
    ):
        """Test that ToolUseAgent initializes correctly."""
        agent = ToolUseAgent(
            agent_name="TestAgent",
            llm_config=mock_llm_config,
            aggregated_client=mock_mcp_client,
            monologue=mock_monologue,
        )

        # Test that the agent was created successfully
        assert isinstance(agent, ToolUseAgent)
        assert agent.monologue == mock_monologue
        assert agent.fsm is not None
        # The FSM starts in INIT state
        assert hasattr(agent.fsm, "state")

    def test_has_message_predicate_with_no_user_message(
        self, mock_llm_config, mock_mcp_client, mock_monologue
    ):
        """Test has_message predicate when no user message is provided."""
        agent = ToolUseAgent(
            agent_name="TestAgent",
            llm_config=mock_llm_config,
            aggregated_client=mock_mcp_client,
            monologue=mock_monologue,
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

    def test_has_message_predicate_with_user_message(
        self, mock_llm_config, mock_mcp_client, mock_monologue
    ):
        """Test has_message predicate when user message is provided."""
        agent = ToolUseAgent(
            agent_name="TestAgent",
            llm_config=mock_llm_config,
            aggregated_client=mock_mcp_client,
            monologue=mock_monologue,
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

    @pytest.mark.asyncio
    @patch("anthropic.AsyncAnthropic.messages", new_callable=PropertyMock)
    async def test_normal_flow(
        self,
        mock_messages_prop,
        tmp_file_monologue,
        mock_llm_config,
        mock_mcp_client,
        request,
    ):
        count = 0

        async def async_generator(**args):
            nonlocal count

            async def streamer_1():
                datadir = get_local_datadir(request)
                filename = "4c_stream_tool_anthropic.json"
                fullpath = os.path.join(datadir, filename)
                with open(fullpath, "r") as f:
                    chunks = json.load(f)
                    adapter = TypeAdapter(List[MessageStreamEvent])
                    chunks = adapter.validate_python(chunks)
                    for chunk in chunks:
                        yield chunk

            async def streamer_2():
                datadir = get_local_datadir(request)
                filename = "4d_stream_tool_use_2_anthropic.json"
                fullpath = os.path.join(datadir, filename)
                with open(fullpath, "r") as f:
                    chunks = json.load(f)
                    adapter = TypeAdapter(List[MessageStreamEvent])
                    chunks = adapter.validate_python(chunks)
                    for chunk in chunks:
                        yield chunk

            if count == 0:
                count += 1
                return streamer_1()
            elif count == 1:
                count += 1
                return streamer_2()
            else:
                raise StopAsyncIteration

        mock_messages = MagicMock()
        mock_messages.create.side_effect = async_generator
        mock_messages_prop.return_value = mock_messages

        # Start testing

        """Test that the agent has a history file."""
        agent = ToolUseAgent(
            agent_name="TestAgent",
            llm_config=mock_llm_config,
            aggregated_client=mock_mcp_client,
            monologue=tmp_file_monologue,
        )

        assert os.path.exists(tmp_file_monologue.file_path)

        # Check if the monologue file exists
        assert tmp_file_monologue.file_path.endswith(".log")

        # Check if the history file exists
        assert os.path.exists(tmp_file_monologue.file_path.replace(".log", ".history"))

        # ACT: INIT -> CHAT -> IS_TOOL_CALL -> TOOL_USE -> FINAL

        resp = agent.run(user_message="What is the current time in Singapore?")
        last_chunk = None
        text = ""
        async for chunk in resp:
            if isinstance(chunk, str):
                chunk = chunk.rstrip()
                if chunk:
                    if not text:
                        text = chunk
                    else:
                        text += " " + chunk
            else:
                last_chunk = chunk
        assert (
            text
            == "I 'll help you find the current time in Singapore. The current time in Singapore is 3:00 PM.  Singapore follows Singapore Standard Time (SGT), which is UTC +8 and does not observe daylight saving time."
        )
        assert len(last_chunk) == 1
        assert len(agent.fsm.state_history) == 6
        assert agent.fsm.state_history[0]["state"] == "INIT"
        assert agent.fsm.state_history[1]["state"] == "HAS_MESSAGE"
        assert agent.fsm.state_history[2]["state"] == "CHAT"
        assert agent.fsm.state_history[3]["state"] == "IS_TOOL_CALL"
        assert agent.fsm.state_history[4]["state"] == "TOOL_USE"
        assert agent.fsm.state_history[5]["state"] == "FINAL"
