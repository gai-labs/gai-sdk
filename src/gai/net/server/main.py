import os
import asyncio
import nats
from nats.errors import TimeoutError
from rich.console import Console
console=Console()

id="sara"
SUBJECT_ALL="dialogue.*"
SUBJECT_agent=f"dialogue.{id}"

import os
if os.environ.get("OPENAI_API_KEY") is None:
    os.environ["OPENAI_API_KEY"] = ""

from openai import OpenAI
client = OpenAI()
from gai.ttt.client.completions import Completions
client = Completions.PatchOpenAI(client)

async def send_message(nc, subject, msg):
    payload = msg.encode("utf-8")
    await nc.publish(subject,payload)

async def main():
    servers = os.environ.get("NATS_URL", "nats://localhost:4222").split(",")
    nc = None
    try:
        nc = await nats.connect(servers=servers)
        console.print(f"[green italic] Successfully connected to servers:{servers}[/]")
    except Exception as e:
        console.print(f"[red] Failed to connect to servers. {e}[/]")
        exit(1)

    try:
        # Listen from_all users
        sub = await nc.subscribe(SUBJECT_ALL)
        while True:
            try:
                msg = await sub.next_msg(timeout=60)

                # ignore message published by self
                if msg.subject == SUBJECT_agent:
                    continue

                subject = msg.subject
                content = msg.data.decode('utf-8')        
                console.print(f"[bright_white]{subject}[/]:>[green italic]{content}[/]")

                if content == "bye":
                    break

                # Publish from_agent
                response=client.chat.completions.create(model="exllamav2-mistral7b",messages=[{"role":"user","content":content}],stream=True)
                await send_message(nc, SUBJECT_agent, "<s>")
                for chunk in response:
                    if chunk:
                        chunk = chunk.extract()
                        if  type(chunk) is str:
                            print(chunk,end="",flush=True)
                            await send_message(nc, SUBJECT_agent, chunk)
                await send_message(nc, SUBJECT_agent, "</s>")
                
            except TimeoutError:
                pass   

    except Exception as e:
        console.print(f"[red]{e}[/]")




if __name__ == '__main__':
    asyncio.run(main())