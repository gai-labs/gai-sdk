# Test Data Generator

import json
from anthropic import AsyncAnthropic
from gai.mcp.client.mcp_client import McpAggregatedClient
from gai.llm.openai.patch_common import openai_to_anthropic_tools
from dotenv import load_dotenv

load_dotenv()


async def main():
    # Build tools list
    aggregated_client = McpAggregatedClient(
        ["mcp-pseudo", "mcp-time", "mcp-web"])
    tools = await aggregated_client.list_tools()
    tools = openai_to_anthropic_tools(tools)

    client = AsyncAnthropic()
    resp = await client.messages.create(
        model="claude-sonnet-4-0",
        max_tokens=32000,
        messages=[
            {
                "role": "user",
                "content": "I love horror stories, are you familiar with them?"
            },
            {
                "role": "assistant",
                "content": "Yes, I am familiar with horror stories. They are a fascinating genre that can evoke strong emotions and create a sense of suspense and fear. Do you have any specific horror stories in mind that you would like to discuss?"
            },
            {
                "role": "user",
                "content": "Tell me a one paragraph story.",
            }
        ],
        tools=tools,
        stream=True,
    )
    chunks = []
    async for chunk in resp:
        chunks.append(chunk.model_dump())

    with open("1a_anthropic_agent_chat.json", "w") as f:
        f.write(json.dumps(chunks, indent=4))


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
