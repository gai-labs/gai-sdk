import os
import asyncio
import nats
import sys
from nats.errors import TimeoutError
from rich.console import Console
console=Console()
import aioconsole

SUBJECT_ALL="dialogue.*"
SUBJECT_user=f"dialogue.user"

async def main():
    servers = os.environ.get("NATS_URL", "nats://localhost:4222").split(",")
    nc = await nats.connect(servers=servers)
    console.print(f"[yellow italic] Publishing to servers:{servers}[/]")

    prompt = await aioconsole.ainput("Please enter a message:")
    while prompt.lower() != "bye":
    
        
        await nc.publish(SUBJECT_user,prompt.encode("utf-8"))
        sub = await nc.subscribe(SUBJECT_ALL)
        
        # Listen from_all agents
        is_streaming = False
        while True:

            msg = await sub.next_msg(timeout=100)
            content = msg.data.decode('utf-8')
            if content == "<s>":
                is_streaming = True
                console.print(f"[bright_white]{msg.subject}[/]:>")
                continue

            if content == "</s>":
                is_streaming = False
                break

            if is_streaming:
                print(f"[green]{content}[/]", end="", flush=True)

        print()
        prompt = await aioconsole.ainput("Please enter a message:")


if __name__ == '__main__':
    asyncio.run(main())    