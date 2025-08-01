import os
import pytest
import json
from anthropic.types import MessageStreamEvent
from pydantic import TypeAdapter
from typing import List
from unittest.mock import MagicMock, patch, PropertyMock, AsyncMock
from gai.lib.tests import get_local_datadir
from gai.asm.agents.tool_use_agent import AnthropicChatState
from gai.messages import Monologue
from gai.asm.asm import AgenticStateMachine


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


class TestAnthropicChatState:
    """Test suite for AnthropicChatState"""

    @pytest.mark.asyncio
    @patch("anthropic.AsyncAnthropic.messages", new_callable=PropertyMock)
    async def test_chat_state_with_tool_use(self, mock_messages_prop, request):
        """
        This test is using the same underlying Athropic API mocked response but called via the GAI client.
        """

        # Create a mock with a create() method that returns an iterable

        async def async_generator(**args):
            async def streamer():
                datadir = get_local_datadir(request)
                filename = "1a_anthropic_agent_chat.json"
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
        self.mock_machine.user_message = "Tell me a one paragraph story."
        state = AnthropicChatState(self.mock_machine)
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
        last_chunk = []
        async for chunk in streamer:
            if isinstance(chunk, str):
                chunk = chunk.rstrip()
                if chunk:
                    text += chunk
            else:
                last_chunk = chunk
        assert (
            "Here's a horror story for you:\n\nSarah always felt safe in her grandmother's old Victorian house until she found the diary hidden beneath the floorboards of the attic. The yellowed pages revealed her grandmother's desperate entries about \"the thing that watches from the walls,\" describing how it would scratch and whisper her name each night, growing bolder with every passing day."
            in text
        )
        assert len(last_chunk) == 1
        assert last_chunk[0]["type"] == "text"
        assert last_chunk[0]["text"] == text

        monologue_messages = self.mock_machine.monologue.list_messages()
        assert len(monologue_messages) == 2
        assert monologue_messages[0].body.role == "user"
        assert monologue_messages[1].body.role == "assistant"


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])
