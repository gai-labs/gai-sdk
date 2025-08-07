import os

os.environ["LOG_LEVEL"] = "Warning"
import json
import asyncio
from gai.sessions import SessionManager
from rich.console import Console
from gai.nodes.agent_node import AgentNode
from gai.nodes.user_node import UserNode
from gai.lib.constants import DEFAULT_GUID
from dotenv import load_dotenv

load_dotenv()
console = Console(force_terminal=True)


async def main(session_mgr: SessionManager):
    # Initialize the dialogue
    session_mgr.reset()

    await session_mgr.start()

    ## Create plan

    flow_plan = """
        User ->> Sara
        Sara ->> Diana
        """

    ## Register Sara and wait for User

    sara = AgentNode(
        agent_name="Sara", model_name="llama3.2:3b", session_mgr=session_mgr
    )
    await sara.subscribe(flow_plan)

    ## Register Diana and wait for User

    diana = AgentNode(
        agent_name="Diana", model_name="sonnet-4", session_mgr=session_mgr
    )
    await diana.subscribe(flow_plan)

    user = UserNode(session_mgr=session_mgr)

    ## START Chain Response
    resp = await user.start(
        user_message="Tell me a one paragraph story about a dragon and a knight.",
        flow_plan=flow_plan,
    )
    content = ""
    async for chunk in resp:
        if not content:
            console.print(f"[bright_green]{chunk}[/bright_green] :")
            content = chunk
        else:
            print(chunk, end="", flush=True)
            content += chunk

    ## RESUME Chain Response
    resp = await user.resume()
    content = ""
    async for chunk in resp:
        if not content:
            console.print(f"[bright_green]{chunk}[/bright_green] :")
            content = chunk
        else:
            print(chunk, end="", flush=True)
            content += chunk


if __name__ == "__main__":
    session_mgr = SessionManager(
        dialogue_id=DEFAULT_GUID, file_path=os.path.join("tmp", "dialogue.json")
    )
    try:
        asyncio.run(main(session_mgr))
    except KeyboardInterrupt:
        print("\nProcess interrupted by user.")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # Stop the session manager
        asyncio.run(session_mgr.stop())
