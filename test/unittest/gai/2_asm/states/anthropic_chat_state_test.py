import pytest
from unittest.mock import MagicMock, patch, PropertyMock, AsyncMock
from data.mock_openai_patch import chat_completions_streaming_toolcall
from gai.asm.states.AnthropicChatState import AnthropicChatState
from gai.messages import Monologue
from gai.asm.asm import AsyncStateMachine


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
        self.state_history = AsyncStateMachine.StateHistory()
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
    async def test_chat_state_with_tool_use(self, mock_messages_prop):
        """
        This test is using the same underlying Athropic API mocked response but called via the GAI client.
        """

        # Create a mock with a create() method that returns an iterable

        async def async_generator(**args):
            async def streamer():
                for chunk in chat_completions_streaming_toolcall("anthropic"):
                    yield chunk

            return streamer()

        mock_messages = MagicMock()
        mock_messages.create.side_effect = async_generator
        mock_messages_prop.return_value = mock_messages

        # Init State
        self.mock_machine = MockMachine()
        self.mock_machine.user_message = "What is the current time in Singapore?"
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
        content = ""
        last_chunk = []
        async for chunk in streamer:
            if isinstance(chunk, str):
                chunk = chunk.rstrip()
                if chunk:
                    content += chunk
            else:
                last_chunk = chunk
        assert content == "I'll help you find the current time in Singapore."
        assert last_chunk[1]["input"]["search_query"] == "current time in Singapore"
        monologue_messages = self.mock_machine.monologue.list_messages()
        assert len(monologue_messages) == 2
        assert monologue_messages[0].body.role == "user"
        assert monologue_messages[1].body.role == "assistant"


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])
