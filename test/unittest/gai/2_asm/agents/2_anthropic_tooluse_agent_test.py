import os
import json
import uuid
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
    def mock_file_monologue(self):
        """Create a temporary file monologue"""
        from gai.messages import FileMonologue

        temp_file_path = os.path.join("/tmp", str(uuid.uuid4()) + ".log")
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

    def get_history_size(self):
        import json
        from gai.asm.constants import HISTORY_PATH
        from gai.lib.constants import DEFAULT_GUID

        history_path = os.path.expanduser(
            HISTORY_PATH.format(
                caller_id=DEFAULT_GUID, dialogue_id=DEFAULT_GUID, order_no=0
            )
        )
        jsoned = []
        with open(history_path, "r") as f:
            jsoned = json.loads(f.read())
        return len(jsoned)

    @pytest.mark.asyncio
    @patch("anthropic.AsyncAnthropic.messages", new_callable=PropertyMock)
    async def test_normal_flow(
        self,
        mock_messages_prop,
        mock_file_monologue,
        mock_llm_config,
        mock_mcp_client,
        request,
    ):
        count = 0

        async def async_generator(**args):
            nonlocal count

            async def streamer_1():
                datadir = get_local_datadir(request)
                filename = "2d_anthropic_agent_chat_recap.json"
                fullpath = os.path.join(datadir, filename)
                with open(fullpath, "r") as f:
                    chunks = json.load(f)
                    adapter = TypeAdapter(List[MessageStreamEvent])
                    chunks = adapter.validate_python(chunks)
                    for chunk in chunks:
                        yield chunk

            async def streamer_2():
                datadir = get_local_datadir(request)
                filename = "2e_anthropic_agent_chat_tooluse.json"
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

        agent = ToolUseAgent(
            agent_name="TestAgent",
            llm_config=mock_llm_config,
            aggregated_client=mock_mcp_client,
            monologue=mock_file_monologue,
        )

        # ACT: INIT -> IS_TOOL_CALL

        await agent._init_async()
        print(f"\ncurrent state: {agent.fsm.state}")
        assert agent.fsm.state == "IS_TOOL_CALL"
        assert agent.fsm.state_bag["predicate_result"] is False
        assert agent.fsm.state_bag["is_tool_call_result"] is False

        # ACT: IS_TOOL_CALL -> CHAT

        resp = await agent._run_async(
            user_message="What is the current time in Singapore?"
        )
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
            text
            == "I'll help you get the current time in Singapore. Let me check that for you."
        )
        assert len(last_chunk) == 2
        assert last_chunk[0]["type"] == "text"
        assert last_chunk[0]["text"] == text
        assert last_chunk[1]["type"] == "tool_use"
        assert last_chunk[1]["input"]["timezone"] == "Asia/Singapore"
        assert agent.final_output() == text
        messages = agent.monologue.list_messages()
        assert len(messages) == 2

        # ACT: CHAT -> IS_TERMINATE

        resp = await agent._run_async()
        print(f"\ncurrent state: {agent.fsm.state}")
        assert agent.fsm.state == "IS_TERMINATE"
        assert agent.fsm.state_bag["predicate_result"] is False
        assert agent.fsm.state_bag["is_terminate_result"] is False

        # ACT: IS_TERMINATE -> IS_TOOL_CALL

        await agent._run_async()
        print(f"\ncurrent state: {agent.fsm.state}")
        assert agent.fsm.state == "IS_TOOL_CALL"
        assert agent.fsm.state_bag["predicate_result"] is True
        assert agent.fsm.state_bag["is_tool_call_result"] is True

        # ACT: IS_TOOL_CALL -> TOOL_USE

        resp = await agent._run_async()
        last_chunk = []
        text = ""
        async for chunk in resp:
            if isinstance(chunk, str):
                if chunk.rstrip():
                    text += chunk
            else:
                last_chunk = chunk
        print(f"\ncurrent state: {agent.fsm.state}")

        assert agent.fsm.state == "TOOL_USE"
        assert (
            text
            == "The current time in Singapore is 3:00 PM. Singapore follows Singapore Standard Time (SGT), which is UTC+8 and does not observe daylight saving time."
        )
        assert len(last_chunk) == 1
        assert last_chunk[0]["type"] == "text"
        assert last_chunk[0]["text"] == text
        assert agent.final_output() == text
        messages = agent.monologue.list_messages()
        assert len(messages) == 4

    @pytest.mark.asyncio
    @patch("anthropic.AsyncAnthropic.messages", new_callable=PropertyMock)
    async def test_stateless_flow(
        self,
        mock_messages_prop,
        mock_file_monologue,
        mock_llm_config,
        mock_mcp_client,
        request,
    ):
        """
        The stateless flow is similar to the normal flow but we reinitialize the agent object after each step to simulate a stateless interaction.
        The agent should be able to load its state from the history file and continue the conversation seamlessly.
        For this to work, make sure agent._init_async() is not called after the first step otherwise it will reset the history.
        """

        count = 0

        async def async_generator(**args):
            nonlocal count

            async def streamer_1():
                datadir = get_local_datadir(request)
                filename = "2d_anthropic_agent_chat_recap.json"
                fullpath = os.path.join(datadir, filename)
                with open(fullpath, "r") as f:
                    chunks = json.load(f)
                    adapter = TypeAdapter(List[MessageStreamEvent])
                    chunks = adapter.validate_python(chunks)
                    for chunk in chunks:
                        yield chunk

            async def streamer_2():
                datadir = get_local_datadir(request)
                filename = "2e_anthropic_agent_chat_tooluse.json"
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

        agent = ToolUseAgent(
            agent_name="TestAgent",
            llm_config=mock_llm_config,
            aggregated_client=mock_mcp_client,
            monologue=mock_file_monologue,
        )

        # ACT: INIT -> IS_TOOL_CALL

        await agent._init_async()
        print(f"\ncurrent state: {agent.fsm.state}")
        assert agent.fsm.state == "IS_TOOL_CALL"
        assert agent.fsm.state_bag["predicate_result"] is False
        assert agent.fsm.state_bag["is_tool_call_result"] is False

        # History size should be the number of states transitioned so its 2 because INIT + IS_TOOL_CALL
        assert self.get_history_size() == 2

        # ACT: IS_TOOL_CALL -> CHAT

        # Agent is reinitialized with exact states

        agent = ToolUseAgent(
            agent_name="TestAgent",
            llm_config=mock_llm_config,
            aggregated_client=mock_mcp_client,
            monologue=mock_file_monologue,
        )
        assert agent.fsm.state == "IS_TOOL_CALL"
        assert agent.fsm.state_bag["predicate_result"] is False
        assert agent.fsm.state_bag["is_tool_call_result"] is False

        resp = await agent._run_async(
            user_message="What is the current time in Singapore?"
        )
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
            text
            == "I'll help you get the current time in Singapore. Let me check that for you."
        )
        assert len(last_chunk) == 2
        assert last_chunk[0]["type"] == "text"
        assert last_chunk[0]["text"] == text
        assert last_chunk[1]["type"] == "tool_use"
        assert last_chunk[1]["input"]["timezone"] == "Asia/Singapore"
        assert agent.final_output() == text
        messages = agent.monologue.list_messages()
        assert len(messages) == 2

        # ACT: CHAT -> IS_TERMINATE

        # Agent is reinitialized

        agent = ToolUseAgent(
            agent_name="TestAgent",
            llm_config=mock_llm_config,
            aggregated_client=mock_mcp_client,
            monologue=mock_file_monologue,
        )

        resp = await agent._run_async()
        print(f"\ncurrent state: {agent.fsm.state}")
        assert agent.fsm.state == "IS_TERMINATE"
        assert agent.fsm.state_bag["predicate_result"] is False
        assert agent.fsm.state_bag["is_terminate_result"] is False

        # ACT: IS_TERMINATE -> IS_TOOL_CALL

        # Agent is reinitialized

        agent = ToolUseAgent(
            agent_name="TestAgent",
            llm_config=mock_llm_config,
            aggregated_client=mock_mcp_client,
            monologue=mock_file_monologue,
        )

        await agent._run_async()
        print(f"\ncurrent state: {agent.fsm.state}")
        assert agent.fsm.state == "IS_TOOL_CALL"
        assert agent.fsm.state_bag["predicate_result"] is True
        assert agent.fsm.state_bag["is_tool_call_result"] is True

        # ACT: IS_TOOL_CALL -> TOOL_USE

        # Agent is reinitialized

        agent = ToolUseAgent(
            agent_name="TestAgent",
            llm_config=mock_llm_config,
            aggregated_client=mock_mcp_client,
            monologue=mock_file_monologue,
        )

        resp = await agent._run_async()
        last_chunk = []
        text = ""
        async for chunk in resp:
            if isinstance(chunk, str):
                if chunk.rstrip():
                    text += chunk
            else:
                last_chunk = chunk
        print(f"\ncurrent state: {agent.fsm.state}")

        assert agent.fsm.state == "TOOL_USE"
        assert (
            text
            == "The current time in Singapore is 3:00 PM. Singapore follows Singapore Standard Time (SGT), which is UTC+8 and does not observe daylight saving time."
        )
        assert len(last_chunk) == 1
        assert last_chunk[0]["type"] == "text"
        assert last_chunk[0]["text"] == text
        assert agent.final_output() == text
        messages = agent.monologue.list_messages()
        assert len(messages) == 4

    @pytest.mark.asyncio
    @patch("anthropic.AsyncAnthropic.messages", new_callable=PropertyMock)
    async def test_terminate_flow(
        self,
        mock_messages_prop,
        mock_file_monologue,
        mock_llm_config,
        mock_mcp_client,
        request,
    ):
        """
        This test will call resume() until agent has completed its task and unable to resume further.
        Run resume(user_message) to continue.
        """

        count = 0

        async def async_generator(**args):
            nonlocal count

            async def streamer_1():
                datadir = get_local_datadir(request)
                filename = "2d_anthropic_agent_chat_recap.json"
                fullpath = os.path.join(datadir, filename)
                with open(fullpath, "r") as f:
                    chunks = json.load(f)
                    adapter = TypeAdapter(List[MessageStreamEvent])
                    chunks = adapter.validate_python(chunks)
                    for chunk in chunks:
                        yield chunk

            async def streamer_2():
                datadir = get_local_datadir(request)
                filename = "2e_anthropic_agent_chat_tooluse.json"
                fullpath = os.path.join(datadir, filename)
                with open(fullpath, "r") as f:
                    chunks = json.load(f)
                    adapter = TypeAdapter(List[MessageStreamEvent])
                    chunks = adapter.validate_python(chunks)
                    for chunk in chunks:
                        yield chunk

            async def streamer_3():
                datadir = get_local_datadir(request)
                filename = "1a_anthropic_agent_chat.json"
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
            elif count == 2:
                count += 1
                return streamer_3()
            else:
                raise StopAsyncIteration

        mock_messages = MagicMock()
        mock_messages.create.side_effect = async_generator
        mock_messages_prop.return_value = mock_messages

        # Start testing

        agent = ToolUseAgent(
            agent_name="TestAgent",
            llm_config=mock_llm_config,
            aggregated_client=mock_mcp_client,
            monologue=mock_file_monologue,
        )

        # ACT: INIT -> IS_TOOL_CALL

        await agent._init_async()
        print(f"\ncurrent state: {agent.fsm.state}")

        # ACT: IS_TOOL_CALL -> CHAT

        resp = await agent._run_async(
            user_message="What is the current time in Singapore?"
        )
        async for chunk in resp:
            pass
        print(f"\ncurrent state: {agent.fsm.state}")

        # ACT: CHAT -> IS_TERMINATE

        resp = await agent._run_async()
        print(f"\ncurrent state: {agent.fsm.state}")

        # ACT: IS_TERMINATE -> IS_TOOL_CALL

        await agent._run_async()
        print(f"\ncurrent state: {agent.fsm.state}")

        # ACT: IS_TOOL_CALL -> TOOL_USE

        resp = await agent._run_async()
        async for chunk in resp:
            pass
        print(f"\ncurrent state: {agent.fsm.state}")

        # ACT: TOOL_USE -> IS_TERMINATE
        await agent._run_async()
        print(f"\ncurrent state: {agent.fsm.state}")

        # ACT: IS_TERMINATE -> IS_TOOL_CALL
        await agent._run_async()
        print(f"\ncurrent state: {agent.fsm.state}")

        # ACT: IS_TOOL_CALL -> ERROR

        # ┌─────────────────────────────────────────────────────────────────────────────┐
        # │ task is completed and cannot resume()                                       │
        # └─────────────────────────────────────────────────────────────────────────────┘

        try:
            await agent._run_async()
            print(f"\ncurrent state: {agent.fsm.state}")
        except Exception as e:
            # Expecting user_message is missing error
            assert "user_message is missing" in str(e)

        # ACT: ERROR -> IS_TERMINATE

        await agent._run_async()

        # ACT: IS_TERMINATE -> IS_TOOL_CALL

        await agent._run_async()

        # ACT: IS_TOOL_CALL -> CHAT

        # ┌─────────────────────────────────────────────────────────────────────────────┐
        # │ resume(user_message) to continue                                            │
        # └─────────────────────────────────────────────────────────────────────────────┘

        resp = await agent._run_async("Tell me a one paragraph story")
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
            text
            == 'Here\'s a horror story for you:\n\nSarah always felt safe in her grandmother\'s old Victorian house until she found the diary hidden beneath the floorboards of the attic. The yellowed pages revealed her grandmother\'s desperate entries about "the thing that watches from the walls," describing how it would scratch and whisper her name each night, growing bolder with every passing day. As Sarah read the final entry—dated the night her grandmother died—she heard a familiar sound: the soft scraping of fingernails against wood, coming from inside the walls around her. The scratching grew louder, more insistent, and then she heard it—a raspy whisper calling her name, just as the diary had described. When she looked up from the pages, she saw fresh scratches appearing on the wall before her, spelling out a message that made her blood run cold: "Welcome home, Sarah."\n\nWhat did you think of that? Do you enjoy psychological horror, supernatural elements, or are you more drawn to other types of horror stories?'
        )

    @pytest.mark.asyncio
    @patch("anthropic.AsyncAnthropic.messages", new_callable=PropertyMock)
    async def test_llm_interrupt_and_run_async_flow(
        self,
        mock_messages_prop,
        mock_file_monologue,
        mock_llm_config,
        mock_mcp_client,
        request,
    ):
        count = 0

        async def async_generator(**args):
            nonlocal count

            async def streamer_1():
                datadir = get_local_datadir(request)
                filename = "2a_anthropic_agent_chat.json"
                fullpath = os.path.join(datadir, filename)
                with open(fullpath, "r") as f:
                    chunks = json.load(f)
                    adapter = TypeAdapter(List[MessageStreamEvent])
                    chunks = adapter.validate_python(chunks)
                    for chunk in chunks:
                        yield chunk

            async def streamer_2():
                datadir = get_local_datadir(request)
                filename = "2b_anthropic_agent_tooluse.json"
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
        )

        # ACT: INIT -> IS_TOOL_CALL

        await agent._init_async()
        print(f"\ncurrent state: {agent.fsm.state}")
        assert agent.fsm.state == "IS_TOOL_CALL"
        assert agent.fsm.state_bag["predicate_result"] is False
        assert agent.fsm.state_bag["is_tool_call_result"] is False

        # ACT: IS_TOOL_CALL -> CHAT

        resp = await agent._run_async(
            user_message="When is the next public holiday? Please ask if you need more information."
        )
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
            text
            == "I need some additional information to help you find the next public holiday:"
        )
        assert len(last_chunk) == 2
        assert last_chunk[0]["type"] == "text"
        assert last_chunk[0]["text"] == text
        assert last_chunk[1]["type"] == "tool_use"
        assert last_chunk[1]["name"] == "user_input"
        assert agent.fsm.state_bag["is_user_input"] == True
        assert agent.final_output() == text
        messages = agent.monologue.list_messages()
        assert len(messages) == 2

        # ACT: CHAT -> IS_TERMINATE

        resp = await agent._run_async()
        print(f"\ncurrent state: {agent.fsm.state}")
        assert agent.fsm.state == "IS_TERMINATE"
        assert agent.fsm.state_bag["predicate_result"] is False
        assert agent.fsm.state_bag["is_terminate_result"] is False

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
        assert not text

        # ACT: IS_TERMINATE -> IS_TOOL_CALL

        resp = await agent._run_async()
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
        assert agent.fsm.state == "IS_TOOL_CALL"
        assert agent.fsm.state_bag["predicate_result"] is True
        assert agent.fsm.state_bag["is_tool_call_result"] is True
        assert agent.fsm.state_bag["is_user_input"] is True
        assert not text

        # ┌─────────────────────────────────────────────────────────────────────────────┐
        # │ Resume pending message from AI                                              │
        # └─────────────────────────────────────────────────────────────────────────────┘

        resp = await agent._run_async("Use SGT")
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
        print("text:", text)
        assert (
            text
            == "Thank you! I'll search for the next public holiday in Singapore (SGT timezone)."
        )
        assert len(last_chunk) == 2
        assert last_chunk[0]["type"] == "text"
        assert last_chunk[0]["text"] == text
        assert last_chunk[1]["type"] == "tool_use"
        assert last_chunk[1]["name"] == "current_time"
        assert agent.fsm.state_bag["is_user_input"] is False
        assert agent.final_output() == text
        messages = agent.monologue.list_messages()
        assert len(messages) == 4

    @pytest.mark.asyncio
    @patch("anthropic.AsyncAnthropic.messages", new_callable=PropertyMock)
    async def test_llm_interrupt_and_pending_flow(
        self,
        mock_messages_prop,
        mock_file_monologue,
        mock_llm_config,
        mock_mcp_client,
        request,
    ):
        count = 0

        async def async_generator(**args):
            nonlocal count

            async def streamer_1():
                datadir = get_local_datadir(request)
                filename = "2a_anthropic_agent_chat.json"
                fullpath = os.path.join(datadir, filename)
                with open(fullpath, "r") as f:
                    chunks = json.load(f)
                    adapter = TypeAdapter(List[MessageStreamEvent])
                    chunks = adapter.validate_python(chunks)
                    for chunk in chunks:
                        yield chunk

            async def streamer_2():
                datadir = get_local_datadir(request)
                filename = "2b_anthropic_agent_tooluse.json"
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
            monologue=mock_file_monologue,
        )

        # ACT: INIT -> IS_TOOL_CALL

        await agent._init_async()
        print(f"\ncurrent state: {agent.fsm.state}")
        assert agent.fsm.state == "IS_TOOL_CALL"
        assert agent.fsm.state_bag["predicate_result"] is False
        assert agent.fsm.state_bag["is_tool_call_result"] is False

        # ACT: IS_TOOL_CALL -> CHAT

        resp = await agent._run_async(
            user_message="When is the next public holiday? Please ask if you need more information."
        )
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
            text
            == "I need some additional information to help you find the next public holiday:"
        )
        assert len(last_chunk) == 2
        assert last_chunk[0]["type"] == "text"
        assert last_chunk[0]["text"] == text
        assert last_chunk[1]["type"] == "tool_use"
        assert last_chunk[1]["name"] == "user_input"
        assert agent.final_output() == text
        messages = agent.monologue.list_messages()
        assert len(messages) == 2

        # ACT: CHAT -> IS_TERMINATE

        resp = await agent._run_async()
        print(f"\ncurrent state: {agent.fsm.state}")
        assert agent.fsm.state == "IS_TERMINATE"
        assert agent.fsm.state_bag["predicate_result"] is False
        assert agent.fsm.state_bag["is_terminate_result"] is False

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
        assert not text

        # ACT: IS_TERMINATE -> IS_TOOL_CALL

        resp = await agent._run_async()
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
        assert agent.fsm.state == "IS_TOOL_CALL"
        assert agent.fsm.state_bag["predicate_result"] is True
        assert agent.fsm.state_bag["is_tool_call_result"] is True
        assert agent.fsm.state_bag["is_user_input"] is True
        assert not text

        # ACT: IS_TOOL_CALL -> TOOL_USE
        # Because resume() is called without user input, TOOL_USE state will throw an exception

        try:
            await agent._run_async()
        except Exception as e:
            print(f"Exception caught: {e}")
            assert "pending user input" in str(e)

    @pytest.mark.asyncio
    @patch("anthropic.AsyncAnthropic.messages", new_callable=PropertyMock)
    async def test_user_interrupt_flow(
        self,
        mock_messages_prop,
        mock_file_monologue,
        mock_llm_config,
        mock_mcp_client,
        request,
    ):
        count = 0

        async def async_generator(**args):
            nonlocal count

            async def streamer_1():
                datadir = get_local_datadir(request)
                filename = "2a_anthropic_agent_chat.json"
                fullpath = os.path.join(datadir, filename)
                with open(fullpath, "r") as f:
                    chunks = json.load(f)
                    adapter = TypeAdapter(List[MessageStreamEvent])
                    chunks = adapter.validate_python(chunks)
                    for chunk in chunks:
                        yield chunk

            async def streamer_2():
                datadir = get_local_datadir(request)
                filename = "2c_anthropic_agent_user_interrupt.json"
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
            monologue=mock_file_monologue,
        )

        # ACT: INIT -> IS_TOOL_CALL

        await agent._init_async()
        print(f"\ncurrent state: {agent.fsm.state}")
        assert agent.fsm.state == "IS_TOOL_CALL"
        assert agent.fsm.state_bag["predicate_result"] is False
        assert agent.fsm.state_bag["is_tool_call_result"] is False

        # ACT: IS_TOOL_CALL -> CHAT

        resp = await agent._run_async(
            user_message="When is the next public holiday? Please ask if you need more information."
        )
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
            text
            == "I need some additional information to help you find the next public holiday:"
        )
        assert len(last_chunk) == 2
        assert last_chunk[0]["type"] == "text"
        assert last_chunk[0]["text"] == text
        assert last_chunk[1]["type"] == "tool_use"
        assert last_chunk[1]["name"] == "user_input"
        assert agent.fsm.state_bag["is_user_input"] == True
        assert agent.final_output() == text
        messages = agent.monologue.list_messages()
        assert len(messages) == 2

        # ACT: CHAT -> IS_TERMINATE

        resp = await agent._run_async()
        print(f"\ncurrent state: {agent.fsm.state}")
        assert agent.fsm.state == "IS_TERMINATE"
        assert agent.fsm.state_bag["predicate_result"] is False
        assert agent.fsm.state_bag["is_terminate_result"] is False

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
        assert not text

        # ACT: IS_TERMINATE -> IS_TOOL_CALL

        resp = await agent._run_async()
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
        assert agent.fsm.state == "IS_TOOL_CALL"
        assert agent.fsm.state_bag["predicate_result"] is True
        assert agent.fsm.state_bag["is_tool_call_result"] is True
        assert agent.fsm.state_bag["is_user_input"] is True
        assert not text

        # ACT: IS_TOOL_CALL -> TOOL_USE

        # ┌───────────────────────────────────────────────────────────────────┐
        # │ Instead of answering the question, interrupt with something else  │
        # └───────────────────────────────────────────────────────────────────┘

        resp = await agent._run_async("Tell me a one paragraph joke.")
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
        print("text:", text)
        assert (
            text
            == "I understand you'd like to hear a joke, but to properly answer your original question about the next public holiday, I still need some information:"
        )
        assert len(last_chunk) == 2
        assert last_chunk[0]["type"] == "text"
        assert last_chunk[0]["text"] == text

    @pytest.mark.asyncio
    @patch("anthropic.AsyncAnthropic.messages", new_callable=PropertyMock)
    async def test_user_interrupt_undo_flow(
        self,
        mock_messages_prop,
        mock_file_monologue,
        mock_llm_config,
        mock_mcp_client,
        request,
    ):
        count = 0

        async def async_generator(**args):
            nonlocal count

            async def streamer_1():
                datadir = get_local_datadir(request)
                filename = "2a_anthropic_agent_chat.json"
                fullpath = os.path.join(datadir, filename)
                with open(fullpath, "r") as f:
                    chunks = json.load(f)
                    adapter = TypeAdapter(List[MessageStreamEvent])
                    chunks = adapter.validate_python(chunks)
                    for chunk in chunks:
                        yield chunk

            async def streamer_2():
                datadir = get_local_datadir(request)
                filename = "2c_anthropic_agent_user_interrupt.json"
                fullpath = os.path.join(datadir, filename)
                with open(fullpath, "r") as f:
                    chunks = json.load(f)
                    adapter = TypeAdapter(List[MessageStreamEvent])
                    chunks = adapter.validate_python(chunks)
                    for chunk in chunks:
                        yield chunk

            async def streamer_3():
                datadir = get_local_datadir(request)
                filename = "2b_anthropic_agent_tooluse.json"
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
            elif count == 2:
                count += 1
                return streamer_3()
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
            monologue=mock_file_monologue,
        )

        # ACT: INIT -> IS_TOOL_CALL

        await agent._init_async()
        print(f"\ncurrent state: {agent.fsm.state}")
        assert agent.fsm.state == "IS_TOOL_CALL"
        assert agent.fsm.state_bag["predicate_result"] is False
        assert agent.fsm.state_bag["is_tool_call_result"] is False

        # ACT: IS_TOOL_CALL -> CHAT

        resp = await agent._run_async(
            user_message="When is the next public holiday? Please ask if you need more information."
        )
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
            text
            == "I need some additional information to help you find the next public holiday:"
        )
        assert len(last_chunk) == 2
        assert last_chunk[0]["type"] == "text"
        assert last_chunk[0]["text"] == text
        assert last_chunk[1]["type"] == "tool_use"
        assert last_chunk[1]["name"] == "user_input"
        assert agent.fsm.state_bag["is_user_input"] == True
        assert agent.final_output() == text
        messages = agent.monologue.list_messages()
        assert len(messages) == 2

        # ACT: CHAT -> IS_TERMINATE

        resp = await agent._run_async()
        print(f"\ncurrent state: {agent.fsm.state}")
        assert agent.fsm.state == "IS_TERMINATE"
        assert agent.fsm.state_bag["predicate_result"] is False
        assert agent.fsm.state_bag["is_terminate_result"] is False

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
        assert not text

        # ACT: IS_TERMINATE -> IS_TOOL_CALL

        resp = await agent._run_async()
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
        assert agent.fsm.state == "IS_TOOL_CALL"
        assert agent.fsm.state_bag["predicate_result"] is True
        assert agent.fsm.state_bag["is_tool_call_result"] is True
        assert agent.fsm.state_bag["is_user_input"] is True
        assert not text

        # ACT: IS_TOOL_CALL -> TOOL_USE

        # ┌───────────────────────────────────────────────────────────────────┐
        # │ Instead of answering the question, interrupt with something else  │
        # └───────────────────────────────────────────────────────────────────┘

        resp = await agent._run_async("Tell me a one paragraph joke.")
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
        print("text:", text)
        assert (
            text
            == "I understand you'd like to hear a joke, but to properly answer your original question about the next public holiday, I still need some information:"
        )
        assert len(last_chunk) == 2
        assert last_chunk[0]["type"] == "text"
        assert last_chunk[0]["text"] == text
        messages = agent.monologue.list_messages()
        assert len(messages) == 4
        assert self.get_history_size() == 6

        # ACT: TOOL_USE -> IS_TOOL_CALL

        # ┌─────────────────────────────────────────────────────────────────────────────┐
        # │ This time we will undo the previous user message and send the correct one   │
        # └─────────────────────────────────────────────────────────────────────────────┘

        agent._undo()
        print(f"\ncurrent state: {agent.fsm.state}")
        assert agent.fsm.state == "IS_TOOL_CALL"

        # should undo previous user and assistant messages
        messages = agent.monologue.list_messages()
        assert len(messages) == 2

        # should undo state history
        assert self.get_history_size() == 5

        # ┌─────────────────────────────────────────────────────────────────────────────┐
        # │ Resume pending message from AI                                              │
        # └─────────────────────────────────────────────────────────────────────────────┘

        resp = await agent._run_async("Use SGT")
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
        print("text:", text)
        assert (
            text
            == "Thank you! I'll search for the next public holiday in Singapore (SGT timezone)."
        )
        assert len(last_chunk) == 2
        assert last_chunk[0]["type"] == "text"
        assert last_chunk[0]["text"] == text
        assert last_chunk[1]["type"] == "tool_use"
        assert last_chunk[1]["name"] == "current_time"
        assert agent.fsm.state_bag["is_user_input"] is False
        assert agent.final_output() == text
        messages = agent.monologue.list_messages()
        assert len(messages) == 4

    @pytest.mark.asyncio
    @patch("anthropic.AsyncAnthropic.messages", new_callable=PropertyMock)
    async def test_user_undo_history(
        self,
        mock_messages_prop,
        mock_file_monologue,
        mock_llm_config,
        mock_mcp_client,
        request,
    ):
        """
        Compare the history size before and after undo.
        """

        # Start testing

        """Test that the agent has a history file."""
        agent = ToolUseAgent(
            agent_name="TestAgent",
            llm_config=mock_llm_config,
            aggregated_client=mock_mcp_client,
            monologue=mock_file_monologue,
        )
        # '/home/vscode/.gai/data/00000000-0000-0000-0000-000000000000/User/dialogue/00000000-0000-0000-0000-000000000000/history/0.json'
        history_size_before = len(agent.fsm.state_history.history)
        agent.undo()
        history_size_after = len(agent.fsm.state_history.history)
        assert history_size_after < history_size_before
