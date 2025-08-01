import os
import json
import pytest
from anthropic.types import MessageStreamEvent
from pydantic import TypeAdapter
from typing import List
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock
from gai.asm.agents.tool_use_agent import AnthropicToolUseState
from gai.asm.asm import AgenticStateMachine
from gai.messages import Monologue
from gai.lib.tests import get_local_datadir


class MockMachine:
    def __init__(self):
        self.user_message = ""

        mcp_client = MagicMock()
        mcp_client.list_tools = AsyncMock(
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
        mcp_client.call_tool = AsyncMock(
            return_value={
                "url": "http://www.time.com",
                "data": "The current time in Singapore is 3:00 PM.",
            }
        )

        self.state_bag = {
            "mcp_client": mcp_client,
            "llm_config": {"client_type": "anthropic", "model": "claude-sonnet-4-0"},
        }
        self.state_history = AgenticStateMachine.StateHistory()
        self.state_history.append(
            {
                "state": "CHAT",
                "input": {
                    "llm_config": {"client_type": "anthropic", "model": "sonnet-4"},
                    "mcp_client": MagicMock(),
                },
                "output": {"streamer": MagicMock(), "get_assistant_message": None},
            }
        )
        self.state = "CHAT"
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
        self.agent_name = "Agent"
        self.monologue = Monologue()
        self.monologue.add_user_message(
            content="What is the current time in Singapore?"
        )
        self.monologue.add_assistant_message(
            content=[
                {
                    "citations": None,
                    "text": "I'll help you find the current time in Singapore.",
                    "type": "text",
                },
                {
                    "id": "toolu_01PsZcVuuQAU62ReyTuKYMyH",
                    "input": {"search_query": "current time in Singapore"},
                    "name": "google",
                    "type": "tool_use",
                },
            ]
        )
        self.monologue.add_user_message(
            content=[
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_01PsZcVuuQAU62ReyTuKYMyH",
                    "content": "The current time in Singapore is 3:00 PM.",
                }
            ]
        )


class TestAnthropicToolUseState:
    """Test suite for AnthropicToolUseState"""

    @pytest.mark.asyncio
    @patch("anthropic.AsyncAnthropic.messages", new_callable=PropertyMock)
    async def test_tool_use_state_with_tool_use(self, mock_messages_prop, request):
        """
        This test is using the same underlying Athropic API mocked response but called via the GAI client.
        """

        # Create a mock with a create() method that returns an iterable

        async def async_generator(**args):
            async def streamer():
                datadir = get_local_datadir(request)
                filename = "2e_anthropic_agent_chat_tooluse.json"
                fullpath = os.path.join(datadir, filename)
                with open(fullpath, "r") as f:
                    chunks = json.load(f)
                    adapter = TypeAdapter(List[MessageStreamEvent])
                    chunks = adapter.validate_python(chunks)
                    for chunk in chunks:
                        yield chunk

            return streamer()

        mock_messages = MagicMock()
        mock_messages.create.side_effect = async_generator
        mock_messages_prop.return_value = mock_messages

        # Init State
        self.mock_machine = MockMachine()
        self.mock_machine.user_message = "What is the current time in Singapore?"
        state = AnthropicToolUseState(self.mock_machine)
        state.input = {
            "llm_config": self.mock_machine.state_bag["llm_config"],
            "mcp_client": self.mock_machine.state_bag["mcp_client"],
            "step": 1,
        }

        # Run the state
        await state.run_async()

        # Verify streamer was created
        assert "streamer" in self.mock_machine.state_bag
        streamer = self.mock_machine.state_bag["streamer"]
        text = ""
        last_chunk = None
        async for chunk in streamer:
            if isinstance(chunk, str):
                chunk = chunk.rstrip()
                if chunk:
                    text += chunk
            else:
                last_chunk = chunk
        assert (
            text
            == "The current time in Singapore is 3:00 PM. Singapore follows Singapore Standard Time (SGT), which is UTC+8 and does not observe daylight saving time."
        )
        assert len(last_chunk) == 1


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])
