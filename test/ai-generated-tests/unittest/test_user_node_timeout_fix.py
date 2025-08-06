"""
Unit test to verify the infinite loop fix in UserNode.stream_response().
This test simulates the conditions that would cause an infinite loop.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock
from gai.nodes.user_node import UserNode
from gai.sessions import SessionManager
from gai.messages.typing import MessagePydantic


class TestUserNodeTimeoutFix:
    """Test the timeout fix for infinite loop in stream_response."""

    @pytest.mark.asyncio
    async def test_stream_response_timeout_prevents_infinite_loop(self):
        """Test that stream_response handles timeout gracefully without infinite loop."""
        
        # Create a UserNode with mocked dependencies
        session_mgr = AsyncMock(spec=SessionManager)
        user_node = UserNode(node_name="TestUser", session_mgr=session_mgr, timeout=1)
        
        # Create a mock plan with steps
        mock_plan = MagicMock()
        mock_plan.steps = ["step1", "step2"]  # 2 steps total
        mock_plan.curr_step_no = 0
        user_node.sender.plan = mock_plan
        
        # Put a response_queue that never sends <eom> to simulate the bug condition
        user_node.response_queue = asyncio.Queue()
        
        # Create a mock message chunk that will never be <eom>
        mock_chunk = MagicMock()
        mock_chunk.header.sender = "TestAgent"
        mock_chunk.body.chunk = "Hello world"  # Never "<eom>"
        mock_chunk.body.chunk_no = 0
        
        # Put the mock chunk in the queue
        await user_node.response_queue.put(mock_chunk)
        
        # This should timeout during chunk processing instead of hanging forever
        start_time = asyncio.get_event_loop().time()
        
        # Should complete with timeout protection instead of hanging
        has_more, sender, content = await user_node.stream_response()
        
        end_time = asyncio.get_event_loop().time()
        elapsed = end_time - start_time
        
        # Should timeout in approximately 5 seconds (our configured chunk timeout)
        assert elapsed < 8, f"Test took too long ({elapsed}s), timeout mechanism might not be working"
        assert elapsed > 3, f"Test completed too quickly ({elapsed}s), timeout might not be engaged"
        assert sender == "TestAgent"
        assert content == "Hello world"  # Should get the content before timeout

    @pytest.mark.asyncio
    async def test_stream_response_max_chunks_prevents_infinite_loop(self):
        """Test that stream_response prevents infinite loop with max_chunks limit."""
        
        # Create a UserNode with mocked dependencies
        session_mgr = AsyncMock(spec=SessionManager)
        user_node = UserNode(node_name="TestUser", session_mgr=session_mgr, timeout=1)
        
        # Create a mock plan
        mock_plan = MagicMock()
        mock_plan.steps = ["step1", "step2"]
        mock_plan.curr_step_no = 0
        user_node.sender.plan = mock_plan
        
        user_node.response_queue = asyncio.Queue()
        
        # Create a function to continuously feed non-<eom> chunks
        async def feed_chunks():
            for i in range(1200):  # More than max_chunks (1000)
                mock_chunk = MagicMock()
                mock_chunk.header.sender = "TestAgent"
                mock_chunk.body.chunk = f"chunk_{i}"  # Never "<eom>"
                mock_chunk.body.chunk_no = i
                await user_node.response_queue.put(mock_chunk)
                await asyncio.sleep(0.001)  # Small delay to prevent tight loop
        
        # Start feeding chunks in the background
        feed_task = asyncio.create_task(feed_chunks())
        
        try:
            # This should stop after max_chunks, not run forever
            start_time = asyncio.get_event_loop().time()
            
            has_more, sender, content = await user_node.stream_response()
            
            end_time = asyncio.get_event_loop().time()
            elapsed = end_time - start_time
            
            # Should complete quickly due to max_chunks limit
            assert elapsed < 10, f"Test took too long ({elapsed}s), max_chunks limit might not be working"
            assert "chunk_999" in content, "Should have processed up to max_chunks"
            assert len(content.split("chunk_")) > 900, "Should have processed many chunks before stopping"
            
        finally:
            feed_task.cancel()
            try:
                await feed_task
            except asyncio.CancelledError:
                pass

    @pytest.mark.asyncio
    async def test_stream_response_normal_operation_with_eom(self):
        """Test that normal operation still works with proper <eom> marker."""
        
        # Create a UserNode with mocked dependencies
        session_mgr = AsyncMock(spec=SessionManager)
        user_node = UserNode(node_name="TestUser", session_mgr=session_mgr, timeout=1)
        
        # Create a mock plan
        mock_plan = MagicMock()
        mock_plan.steps = ["step1", "step2"]
        mock_plan.curr_step_no = 0
        user_node.sender.plan = mock_plan
        
        user_node.response_queue = asyncio.Queue()
        
        # Create normal message chunks followed by <eom>
        chunks_data = [
            ("TestAgent", "Hello ", 0),
            ("TestAgent", "world!", 1),
            ("TestAgent", "<eom>", 2),
        ]
        
        for sender, chunk_text, chunk_no in chunks_data:
            mock_chunk = MagicMock()
            mock_chunk.header.sender = sender
            mock_chunk.body.chunk = chunk_text
            mock_chunk.body.chunk_no = chunk_no
            await user_node.response_queue.put(mock_chunk)
        
        # This should work normally
        has_more, sender, content = await user_node.stream_response()
        
        assert sender == "TestAgent"
        assert content == "Hello world!"
        assert has_more is True  # More steps remain (curr_step_no=0, total_steps=2)


if __name__ == "__main__":
    # Run a quick test
    asyncio.run(TestUserNodeTimeoutFix().test_stream_response_normal_operation_with_eom())
    print("Quick test passed!")