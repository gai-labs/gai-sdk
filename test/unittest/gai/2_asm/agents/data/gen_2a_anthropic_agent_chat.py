# Test Data Generator

import json
from anthropic import AsyncAnthropic
from gai.mcp.client.mcp_client import McpAggregatedClient
from gai.llm.openai.patch_common import openai_to_anthropic_tools
from dotenv import load_dotenv

load_dotenv()


async def main():
    # Build tools list
    aggregated_client = McpAggregatedClient(["mcp-pseudo", "mcp-time", "mcp-web"])
    tools = await aggregated_client.list_tools()
    tools = openai_to_anthropic_tools(tools)

    client = AsyncAnthropic()
    resp = await client.messages.create(
        model="claude-sonnet-4-0",
        max_tokens=32000,
        messages=[
            {
                "role": "user",
                "content": "When is the next public holiday? Please ask if you need more information.",
            }
        ],
        tools=tools,
        stream=True,
    )
    chunks = []
    async for chunk in resp:
        chunks.append(chunk.model_dump())

    with open("2a_anthropic_agent_chat.json", "w") as f:
        f.write(json.dumps(chunks, indent=4))


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
