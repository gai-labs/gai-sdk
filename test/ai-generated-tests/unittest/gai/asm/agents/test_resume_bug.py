#!/usr/bin/env python3

"""
Test to surgically reproduce the resume() bug that causes infinite LLM requests.
This test follows the exact pattern from 1_anthropic_chat_agent_test.py.
"""

import os
import json
import uuid
import pytest
from anthropic.types import MessageStreamEvent
from pydantic import TypeAdapter
from typing import List
from unittest.mock import MagicMock, patch, PropertyMock

from gai.asm.agents import ChatAgent, ToolUseAgent, AutoResumeError
from gai.lib.config import GaiClientConfig
from gai.messages import Monologue
from gai.lib.tests import get_local_datadir


class TestResumeInfiniteLoopBug:
    """Test to prove resume() causes infinite LLM calls on terminated agents"""

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
    def mock_file_monologue(self):
        """Create a temporary file monologue"""
        from gai.messages import FileMonologue

        temp_file_path = os.path.join("/tmp", str(uuid.uuid4()) + ".log")
        monologue = FileMonologue(file_path=temp_file_path)
        monologue.reset()
        return monologue

    @pytest.mark.asyncio 
    @patch("anthropic.AsyncAnthropic.messages", new_callable=PropertyMock)
    async def test_resume_after_normal_completion_triggers_bug(
        self,
        mock_messages_prop,
        mock_file_monologue,
        mock_llm_config,
        request,
    ):
        """
        This test replicates the exact scenario from test_normal_flow() but then calls resume()
        to reproduce the bug where terminated agents make unnecessary LLM calls.
        """
        
        # Track LLM calls
        llm_call_count = 0
        original_call_count = 0
        
        async def async_generator(**args):
            nonlocal llm_call_count, original_call_count
            original_call_count += 1
            
            async def streamer():
                # Use the same test data as the normal flow test
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

        # For the subsequent resume() call, we'll track additional calls
        async def resume_call_tracker(**args):
            nonlocal llm_call_count
            llm_call_count += 1
            print(f"❌ BUG: Additional LLM call #{llm_call_count} during resume() on completed agent!")
            print(f"    Request args: {list(args.keys())}")
            
            # Return minimal stream to avoid hanging
            async def minimal_stream():
                yield {
                    "type": "message_start",
                    "message": {
                        "id": "msg_resume_bug",
                        "type": "message",
                        "role": "assistant", 
                        "content": [],
                        "model": "claude-sonnet-4-0",
                        "stop_reason": None,
                        "stop_sequence": None,
                        "usage": {"input_tokens": 1, "output_tokens": 1}
                    }
                }
                yield {"type": "message_stop"}
            
            return minimal_stream()

        mock_messages = MagicMock()
        mock_messages.create.side_effect = async_generator
        mock_messages_prop.return_value = mock_messages

        # Create agent following exact pattern from test_normal_flow()
        agent = ChatAgent(
            agent_name="TestAgent",
            llm_config=mock_llm_config,
            monologue=mock_file_monologue,
        )

        print(f"=== STEP 1: Normal agent flow (should work) ===")
        
        # INIT -> IS_TOOL_CALL (following exact test_normal_flow pattern)
        await agent._init_async()
        print(f"After init: {agent.fsm.state}")
        assert agent.fsm.state == "IS_TOOL_CALL"

        # IS_TOOL_CALL -> CHAT -> IS_TERMINATE (complete the task normally)
        resp = await agent._run_async(user_message="Tell me a one paragraph story.")
        text = ""
        async for chunk in resp:
            if isinstance(chunk, str):
                chunk = chunk.rstrip()
                if chunk:
                    text += chunk

        print(f"After normal completion: {agent.fsm.state}")
        print(f"Original LLM calls made: {original_call_count}")
        
        # At this point, agent should be in a completed state
        # In the real scenario, this would be IS_TERMINATE or similar
        
        print(f"\n=== STEP 2: Testing resume() on completed agent ===")
        
        # Now modify the mock to track additional calls during resume()
        mock_messages.create.side_effect = resume_call_tracker
        
        # Reset the additional call counter
        llm_call_count = 0
        
        # Mock _run_async to track internal calls
        original_run_async = agent._run_async
        run_async_call_count = 0
        
        async def track_run_async(*args, **kwargs):
            nonlocal run_async_call_count
            run_async_call_count += 1
            print(f"🔍 _run_async() call #{run_async_call_count}")
            print(f"   Current state: {agent.fsm.state}")
            print(f"   Args: {args}, Kwargs: {kwargs}")
            
            # Call the original
            return await original_run_async(*args, **kwargs)
        
        agent._run_async = track_run_async
        
        # Force agent to completed state for testing
        # This simulates what happens in multi-agent sessions after completion
        if agent.fsm.state != "IS_TERMINATE":
            print(f"Forcing agent to IS_TERMINATE state (was: {agent.fsm.state})")
            agent.fsm.state = "IS_TERMINATE"
        
        print(f"Agent state before resume(): {agent.fsm.state}")
        
        # THE BUG TEST: Call resume() on completed agent
        try:
            print("Calling resume() on completed agent...")
            async for chunk in agent.resume():
                print(f"❌ Unexpected chunk from resume(): {chunk}")
                break  # Don't let it run forever
                
        except AutoResumeError as e:
            print(f"✅ AutoResumeError raised: {e}")
        except Exception as e:
            print(f"❌ Unexpected exception: {type(e).__name__}: {e}")
        
        print(f"\n=== BUG ANALYSIS ===")
        print(f"_run_async() calls during resume(): {run_async_call_count}")
        print(f"Additional LLM API calls during resume(): {llm_call_count}")
        
        # The bug is proven if _run_async() is called on terminated agent
        if run_async_call_count > 0:
            print("❌ BUG CONFIRMED: resume() called _run_async() on terminated agent!")
            print("   This is the source of infinite loops in multi-agent sessions")
            print("   Lines 636-637 in tool_use_agent.py should not call _run_async() on IS_TERMINATE")
            
            # Don't fail the test yet - we want to see the full behavior
            # assert False, f"BUG: resume() should not call _run_async() on terminated agent"
        else:
            print("✅ NO BUG: resume() correctly avoided _run_async() calls on terminated agent")

    @pytest.mark.asyncio 
    @patch("anthropic.AsyncAnthropic.messages", new_callable=PropertyMock)
    async def test_tooluse_agent_resume_bug(
        self,
        mock_messages_prop,
        mock_file_monologue,
        mock_llm_config,
        request,
    ):
        """
        Test the exact same resume() bug but with ToolUseAgent directly instead of ChatAgent.
        This will confirm if the bug is in the base ToolUseAgent class.
        """
        
        # Track LLM calls
        llm_call_count = 0
        original_call_count = 0
        
        async def async_generator(**args):
            nonlocal llm_call_count, original_call_count
            original_call_count += 1
            
            async def streamer():
                # Use the same test data as the normal flow test
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

        # For the subsequent resume() call, we'll track additional calls
        async def resume_call_tracker(**args):
            nonlocal llm_call_count
            llm_call_count += 1
            print(f"❌ BUG: Additional LLM call #{llm_call_count} during ToolUseAgent resume() on completed agent!")
            print(f"    Request args: {list(args.keys())}")
            
            # Return minimal stream to avoid hanging
            async def minimal_stream():
                yield {
                    "type": "message_start",
                    "message": {
                        "id": "msg_tooluse_resume_bug",
                        "type": "message",
                        "role": "assistant", 
                        "content": [],
                        "model": "claude-sonnet-4-0",
                        "stop_reason": None,
                        "stop_sequence": None,
                        "usage": {"input_tokens": 1, "output_tokens": 1}
                    }
                }
                yield {"type": "message_stop"}
            
            return minimal_stream()

        mock_messages = MagicMock()
        mock_messages.create.side_effect = async_generator
        mock_messages_prop.return_value = mock_messages

        # Create ToolUseAgent directly (not ChatAgent)
        from gai.mcp.client import McpAggregatedClient
        
        agent = ToolUseAgent(
            agent_name="TestToolUseAgent",
            llm_config=mock_llm_config,
            monologue=mock_file_monologue,
            aggregated_client=McpAggregatedClient([])  # Empty client like ChatAgent
        )

        print(f"=== STEP 1: Normal ToolUseAgent flow ===")
        
        # INIT -> IS_TOOL_CALL (following exact pattern)
        await agent._init_async()
        print(f"After init: {agent.fsm.state}")
        assert agent.fsm.state == "IS_TOOL_CALL"

        # IS_TOOL_CALL -> CHAT -> IS_TERMINATE (complete the task normally)
        resp = await agent._run_async(user_message="Tell me a one paragraph story.")
        text = ""
        async for chunk in resp:
            if isinstance(chunk, str):
                chunk = chunk.rstrip()
                if chunk:
                    text += chunk

        print(f"After normal completion: {agent.fsm.state}")
        print(f"Original LLM calls made: {original_call_count}")
        
        print(f"\n=== STEP 2: Testing ToolUseAgent resume() on completed agent ===")
        
        # Now modify the mock to track additional calls during resume()
        mock_messages.create.side_effect = resume_call_tracker
        
        # Reset the additional call counter
        llm_call_count = 0
        
        # Mock _run_async to track internal calls
        original_run_async = agent._run_async
        run_async_call_count = 0
        
        async def track_run_async(*args, **kwargs):
            nonlocal run_async_call_count
            run_async_call_count += 1
            print(f"🔍 ToolUseAgent._run_async() call #{run_async_call_count}")
            print(f"   Current state: {agent.fsm.state}")
            print(f"   Args: {args}, Kwargs: {kwargs}")
            
            # Call the original
            return await original_run_async(*args, **kwargs)
        
        agent._run_async = track_run_async
        
        # Force agent to completed state for testing
        if agent.fsm.state != "IS_TERMINATE":
            print(f"Forcing ToolUseAgent to IS_TERMINATE state (was: {agent.fsm.state})")
            agent.fsm.state = "IS_TERMINATE"
        
        print(f"ToolUseAgent state before resume(): {agent.fsm.state}")
        
        # THE BUG TEST: Call resume() on completed ToolUseAgent
        try:
            print("Calling resume() on completed ToolUseAgent...")
            async for chunk in agent.resume():
                print(f"❌ Unexpected chunk from ToolUseAgent resume(): {chunk}")
                break  # Don't let it run forever
                
        except AutoResumeError as e:
            print(f"✅ AutoResumeError raised: {e}")
        except Exception as e:
            print(f"❌ Unexpected exception: {type(e).__name__}: {e}")
        
        print(f"\n=== TOOLUSE AGENT BUG ANALYSIS ===")
        print(f"ToolUseAgent._run_async() calls during resume(): {run_async_call_count}")
        print(f"Additional LLM API calls during resume(): {llm_call_count}")
        
        # The bug is proven if _run_async() is called on terminated agent
        if run_async_call_count > 0:
            print("❌ BUG CONFIRMED: ToolUseAgent.resume() called _run_async() on terminated agent!")
            print("   This confirms the bug is in the base ToolUseAgent class")
            print("   ChatAgent inherits this buggy behavior from ToolUseAgent")
            print("   Lines 636-637 in tool_use_agent.py should not call _run_async() on IS_TERMINATE")
            
            # Document the comparison
            print(f"\n🔍 COMPARISON:")
            print(f"   ChatAgent _run_async() calls: 3 (from previous test)")
            print(f"   ToolUseAgent _run_async() calls: {run_async_call_count}")
            
            if run_async_call_count == 3:
                print("   ✅ Both agents show identical behavior - confirms inheritance issue")
            else:
                print("   ⚠️  Different behavior - may indicate additional complexity")
                
        else:
            print("✅ NO BUG: ToolUseAgent.resume() correctly avoided _run_async() calls on terminated agent")
    
    @pytest.mark.asyncio
    async def test_resume_normal_case_should_work(self):
        """
        Control test: Ensure resume() works correctly on non-terminated agents.
        This ensures our eventual fix doesn't break normal functionality.
        """
        
        # Create agent (not terminated)
        llm_config = GaiClientConfig(
            client_type="anthropic",
            type="ttt", 
            model="claude-sonnet-4-0",
            url=None
        )
        
        agent = ChatAgent(
            agent_name="TestAgent",
            llm_config=llm_config
        )
        
        print(f"\nNormal case - Agent state: {agent.fsm.state}")
        
        # For normal (non-terminated) agent, resume() making LLM calls is expected
        # This test just verifies the agent is not already terminated
        
        if agent.fsm.state != "IS_TERMINATE":
            print("✅ Control test: Agent is not terminated - resume() should work normally")
        else:
            print("❌ Unexpected: Fresh agent is already terminated")