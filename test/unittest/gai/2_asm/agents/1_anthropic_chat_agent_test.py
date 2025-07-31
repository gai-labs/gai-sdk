import os
import json
import uuid
import pytest
from anthropic.types import MessageStreamEvent
from pydantic import TypeAdapter
from typing import List
from unittest.mock import MagicMock, patch, PropertyMock, AsyncMock
from gai.lib.tests import get_local_datadir
from gai.asm.agents.chat_agent import ChatAgent
from gai.lib.config import GaiClientConfig
from gai.messages import Monologue
from gai.lib.logging import getLogger

logger = getLogger(__name__)


class TestChatAgent:
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
    def mock_monologue(self):
        """Create a mock monologue."""
        monologue = MagicMock(spec=Monologue)
        monologue.list_messages.return_value = []
        monologue.reset.return_value = None
        monologue.pop.return_value = None
        return monologue

    @pytest.fixture
    def mock_file_monologue(self):
        """Create a temporary file monologue"""
        from gai.messages import FileMonologue

        temp_file_path = os.path.join("/tmp", str(uuid.uuid4()) + ".log")
        monologue = FileMonologue(file_path=temp_file_path)
        monologue.reset()
        return monologue

    @pytest.mark.asyncio
    @patch("anthropic.AsyncAnthropic.messages", new_callable=PropertyMock)
    async def test_normal_flow(
        self,
        mock_messages_prop,
        mock_file_monologue,
        mock_llm_config,
        request,
    ):
        count = 0

        async def async_generator(**args):
            nonlocal count

            async def streamer_1():
                datadir = get_local_datadir(request)
                filename = "1a_anthropic_agent_chat.json"
                fullpath = os.path.join(datadir, filename)
                with open(fullpath, "r") as f:
                    chunks = json.load(f)
                    adapter = TypeAdapter(List[MessageStreamEvent])
                    chunks = adapter.validate_python(chunks)
                    for chunk in chunks:
                        yield chunk

            return streamer_1()

        mock_messages = MagicMock()
        mock_messages.create.side_effect = async_generator
        mock_messages_prop.return_value = mock_messages

        # Start testing

        """Test that the agent has a history file."""
        agent = ChatAgent(
            agent_name="TestAgent",
            llm_config=mock_llm_config,
            monologue=mock_file_monologue,
        )

        # ACT: INIT -> IS_TOOL_CALL

        await agent._init_async()
        print(f"\ncurrent state: {agent.fsm.state}")
        assert agent.fsm.state == "IS_TOOL_CALL"
        assert agent.fsm.state_bag["predicate_result"] is False
        assert agent.fsm.state_bag["is_tool_call_result"] is False

        # ACT: IS_TOOL_CALL -> CHAT

        resp = await agent._run_async(user_message="Tell me a one paragraph story.")
        last_chunk = []
        text = ""
        async for chunk in resp:
            if isinstance(chunk, str):
                chunk = chunk.rstrip()
                if chunk:
                    text += chunk
            else:
                last_chunk = chunk
        print(f"\ncurrent state: {agent.fsm.state}")
        assert agent.fsm.state == "CHAT"
        assert (
            "Here's a horror story for you:\n\nSarah always felt safe in her grandmother's old Victorian house until she found the diary hidden beneath the floorboards of the attic. The yellowed pages revealed her grandmother's desperate entries about \"the thing that watches from the walls,\" describing how it would scratch and whisper her name each night, growing bolder with every passing day."
            in text
        )
        assert len(last_chunk) == 1
        assert last_chunk[0]["type"] == "text"
        assert last_chunk[0]["text"] == text
        assert agent.final_output() == text
        messages = agent.monologue.list_messages()
        assert len(messages) == 2
        # INIT + IS_TOOL_CALL + CHAT == 3 states
        assert len(agent.fsm.state_history) == 3
