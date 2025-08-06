"""
Integration test for multi-agent session based on quick-start notebook.
"""

import pytest
import asyncio
from gai.sessions import SessionManager
from gai.messages.dialogue import Dialogue
from gai.lib.config import config_helper
from gai.nodes.agent_node import AgentNode
from gai.nodes.user_node import UserNode
from dotenv import load_dotenv

load_dotenv()


class TestMultiAgentSession:
    """Integration test for multi-agent session functionality."""

    @pytest.mark.asyncio
    async def test_haiku_writer_reviewer_session_sonnet4(self):
        """Test multi-agent haiku writing and reviewing with sonnet-4."""
        await self._run_haiku_session("sonnet-4", "sonnet-4")

    @pytest.mark.asyncio
    async def test_haiku_writer_reviewer_session_ttt(self):
        """Test multi-agent haiku writing and reviewing with ttt for writer."""
        await self._run_haiku_session("ttt", "sonnet-4")

    async def _run_haiku_session(self, writer_model: str, reviewer_model: str):
        """
        Run a multi-agent session with Writer and Reviewer agents.

        Args:
            writer_model: Model to use for Writer agent
            reviewer_model: Model to use for Reviewer agent
        """
        # Setup multi-agent session
        session_mgr = SessionManager()
        await session_mgr.start()

        dialogue = Dialogue(agent_name="User")

        try:
            # Create the user node
            user = UserNode(node_name="User", session_mgr=session_mgr)
            await user.subscribe()

            # Define the conversation plan
            flow_plan = """
                User ->> HaikuWriter
                HaikuWriter ->> Reviewer
                """

            # Create agent nodes
            writer = AgentNode(
                agent_name="HaikuWriter",
                session_mgr=session_mgr,
                llm_config=config_helper.get_client_config(
                    writer_model,
                    file_path="/workspaces/gai-sdk/lib/test/integrationtest/data/gai.yml",
                ),
                dialogue=dialogue,
            )
            await writer.subscribe(flow_plan=flow_plan)

            reviewer = AgentNode(
                agent_name="Reviewer",
                session_mgr=session_mgr,
                llm_config=config_helper.get_client_config(
                    reviewer_model,
                    file_path="/workspaces/gai-sdk/lib/test/integrationtest/data/gai.yml",
                ),
                dialogue=dialogue,
            )
            await reviewer.subscribe(flow_plan=flow_plan)

            # Collect outputs for verification
            collected_outputs = []

            def collect_output(sender_name: str, text: str):
                collected_outputs.append({"sender": sender_name, "text": text})
                if sender_name:  # Only print sender name if provided
                    print(f"{sender_name}: {text}", end="", flush=True)
                else:  # Just the text for content chunks
                    print(text, end="", flush=True)

            # Run the complete conversation
            user_message = (
                "Write a short haiku about coding, then review and improve it."
            )

            responses = await user.run_full_conversation(
                user_message=user_message,
                flow_plan=flow_plan,
                display_callback=collect_output,
            )

            # Verify the conversation completed successfully
            assert len(responses) >= 2, (
                f"Expected at least 2 responses, got {len(responses)}"
            )

            # Verify we got outputs from both agents
            sender_names = {output["sender"] for output in collected_outputs}
            assert "HaikuWriter" in sender_names, (
                "HaikuWriter should have provided output"
            )
            assert "Reviewer" in sender_names, "Reviewer should have provided output"

            # Verify the content contains haiku-related terms
            all_text = " ".join([output["text"] for output in collected_outputs])
            assert "haiku" in all_text.lower(), "Output should mention haiku"

            # Check for coding-related content
            coding_terms = ["cod", "debug", "compile", "syntax", "program", "semicolon"]
            has_coding_term = any(term in all_text.lower() for term in coding_terms)
            assert has_coding_term, (
                f"Output should contain coding-related terms. Got: {all_text[:500]}..."
            )

            print(
                f"\\n=== Test completed successfully with {writer_model} (Writer) and {reviewer_model} (Reviewer) ==="
            )
            print(f"Total responses: {len(responses)}")
            print(f"Total output segments: {len(collected_outputs)}")

        except Exception as e:
            print(
                f"\\n=== Test failed with {writer_model} (Writer) and {reviewer_model} (Reviewer) ==="
            )
            print(f"Error: {str(e)}")
            print(f"Error type: {type(e).__name__}")
            raise
        finally:
            # Clean up
            try:
                await session_mgr.stop()
            except Exception as cleanup_error:
                print(f"Cleanup error: {cleanup_error}")


if __name__ == "__main__":
    # Run tests individually for debugging
    import sys

    async def run_single_test():
        test_instance = TestMultiAgentSession()

        if len(sys.argv) > 1 and sys.argv[1] == "ttt":
            print("Running test with ttt for writer...")
            await test_instance.test_haiku_writer_reviewer_session_ttt()
        else:
            print("Running test with sonnet-4 for both agents...")
            await test_instance.test_haiku_writer_reviewer_session_sonnet4()

    if __name__ == "__main__":
        asyncio.run(run_single_test())
