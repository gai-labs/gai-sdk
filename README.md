# GAI SDK - Generative Agent Infrastructure SDK

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Version](https://img.shields.io/badge/version-1.0.249-green.svg)](https://pypi.org/project/gai-sdk/)

A comprehensive Python SDK for building intelligent multi-agent AI systems using Agentic State Machines (ASM), LLM integrations, and tool support via Model Context Protocol (MCP).

## Quick Start

### a) Installation

Download gai-init script and run it to initialize the gai application directory at `~/.gai`.

```bash
pip install --upgrade gai-init
uvx gai-init@latest
```

### b) Setup

GAI-SDK supports local LLMs(EXL2,GGUF) and cloud-based LLMs(OpenAI,Claude).

**Configure for OpenAI**

Create a .env file

```bash
OPENAI_API_KEY=your_openai_api_key
```

**Configure for Anthropic**

Create a .env file

```bash
ANTHROPIC_API_KEY=your_anthropic_api_key
```

**Configure for Ollama**

Download the Ollama model

```bash
ollama pull llama3.2:3b
```

**Configure for Exllama-v2**

Download the Exllama-v2 model

```bash
uvx gai-pull@latest llama3.1-exl2
```

### c) Basic Usage

#### Create an LLM Configuration

In the example below, we use a local Ollama model.

```python
from gai.lib.config import config_helper
from gai.llm.openai import AsyncOpenAI

# "llama3.2:3b" for Ollama with CPU
# or "ttt" for Gai/Exl2 running llama3.1-exl2 with GPU
# or "sonnet-4" for Claude
# or "gpt-4o" for OpenAI

model_name = "llama3.2:3b"
llm_config = config_helper.get_client_config(model_name)
```

#### Create a Haiku Writer Agent

TBD

#### Create a Haiku Reviewer Agent

TBD

#### Create a Multi-Agent Session

TBD

---

## Package Structure

The GAI SDK is organized into several interconnected modules:

### Core Components

-   **`gai.asm`** - Agentic State Machine framework
-   **`gai.messages`** - Message handling and conversation memory
-   **`gai.sessions`** - Session management and orchestration
-   **`gai.mcp`** - Model Context Protocol integration
-   **`gai.lib`** - Core utilities and configuration
-   **`gai.llm`** - LLM client integrations

### Sub-packages

-   **`init/`** - Project initialization and template management
-   **`lib/`** - Core GAI library with shared utilities
-   **`llm/`** - LLM clients and OpenAI compatibility layer

## Architecture & Key Features

The GAI SDK implements a unique **Agentic State Machine** pattern that provides fault tolerance, observability, flexibility, and scalability - making it particularly suitable for production environments requiring reliable, observable, and maintainable AI agent systems.

### Agentic State Machines (ASM)

-   **Declarative State Definition**: Define agent behavior using mermaid-like state diagrams and JSON manifests
-   **State Persistence**: Built-in backup and recovery for fault-tolerant agents
-   **Dependency Injection**: Three types of dependencies (`getter`, `state_bag`, `prev_state`)
-   **Rollback Support**: Undo functionality for error recovery with `fsm.undo()`
-   **Execution Traceability**: Complete state history for debugging and monitoring
-   **Conditional Logic**: `PurePredicateState` for branching based on LLM decisions or custom logic
-   **Streaming Support**: Real-time output streaming from state actions

### Memory & Context Management

-   **Monologue**: Intra-agent conversation memory
-   **Dialogue**: Inter-agent conversation storage (in-memory and persistent)
-   **Message Types**: Structured message formats with automatic type resolution
-   **Overflow Policies**: Configurable memory management strategies

### Tool Integration (MCP)

-   **Multi-Server Support**: Aggregate tools from multiple MCP servers
-   **Auto-Discovery**: Tools automatically discovered at startup
-   **Security**: Directory access controls and tool blocking
-   **Pseudo Tools**: Special tools like `user_input` for interaction flows

### Multi-Agent Collaboration

-   **Session Management**: High-level orchestration for multi-agent conversations
-   **Message Bus Architecture**: Supports local and distributed messaging
-   **Conversation Flows**: Poll (parallel) and chain (sequential) interaction patterns
-   **Network Distribution**: Scale across multiple hosts with NATS

## Core Concepts

### State Machine Pattern

```python
# State diagram using mermaid-like syntax
state_diagram = """
    INIT --> IS_TOOL_CALL
    IS_TOOL_CALL --> CHAT: condition_false
    IS_TOOL_CALL --> TOOL_USE: condition_true
    CHAT --> IS_TERMINATE
    TOOL_USE --> IS_TERMINATE
    IS_TERMINATE --> IS_TOOL_CALL: condition_false
    IS_TERMINATE --> FINAL: condition_true
"""

# Predicate state for conditional logic
def needs_tools(state) -> bool:
    # Use LLM to determine if tools are needed
    from gai.llm.openai import OpenAI, boolean_schema
    client = OpenAI()
    response = client.beta.chat.completions.parse(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Does this require tools?"}],
        response_format=boolean_schema
    )
    return json.loads(response.choices[0].message.content)["result"]

# State manifest defining behavior
state_manifest = {
    "IS_TOOL_CALL": {
        "module_path": "gai.asm.states",
        "class_name": "PurePredicateState",
        "predicate": "needs_tools",
        "conditions": ["condition_true", "condition_false"],
        "output_data": ["predicate_result"]
    },
    "TOOL_USE": {
        "module_path": "gai.asm.states",
        "class_name": "PureActionState",
        "action": "execute_tools",
        "output_data": ["tool_results"]
    }
}
```

### Message Flow

```python
from gai.messages import Monologue, Dialogue

# Intra-agent memory
monologue = Monologue(agent_name="Assistant")
monologue.add_user_message("Hello, world!")
monologue.add_assistant_message("Hi there!")

# Inter-agent communication
dialogue = Dialogue()
dialogue.add_user_message(recipient="Agent1", content="Start analysis")
```

### MCP Tool Integration

```python
from gai.mcp.client import McpClient

# Configure MCP client
mcp_config = {
    "filesystem": {
        "command": "npx",
        "args": ["-y", "@modelcontextprotocol/server-filesystem", "/path/to/files"]
    }
}

client = McpClient(mcp_config)
await client.initialize()

# Tools are auto-discovered and available
tools = await client.list_tools()
```

## Development

### Prerequisites

-   Python 3.10+
-   [uv](https://github.com/astral-sh/uv) package manager

### Setup Development Environment

```bash
# Clone and setup
git clone <repository-url>
cd gai-sdk

# Install in development mode
make install

# Run tests
make test

# Build package
make build
```

### Project Structure

```
gai-sdk/
├── src/gai/                 # Main SDK code
│   ├── asm/                 # Agentic State Machine
│   ├── messages/            # Message handling
│   ├── sessions/            # Session management
│   └── mcp/                 # MCP integration
├── lib/                     # Core library
├── llm/                     # LLM integrations
├── init/                    # Project templates
└── test/                    # Test suites
```

## Key Features Demonstrated

### State Machine Rollback and Recovery

```python
# Undo last state transition
state = agent.undo()
print(f"Rolled back to state: {state}")

# Resume with different input
response = agent.resume("New message after rollback")
```

### Error Handling and User Interruption

```python
from gai.asm.agents.tool_use_agent import PendingUserInputError, AutoResumeError

try:
    response = agent.resume()
    async for chunk in response:
        print(chunk, end="")
except PendingUserInputError:
    # Agent is waiting for user input via user_input tool
    response = agent.resume("User's response")
except AutoResumeError:
    # Conversation is complete
    print("Agent has finished its task")
```

### Memory and Context Management

```python
from gai.messages import FileMonologue, FileDialogue

# Persistent conversation memory
monologue = FileMonologue(agent_name="Assistant", file_path="memory.json")

# Multi-agent dialogue storage
dialogue = FileDialogue(file_path="conversation.json")
recap = dialogue.extract_recap()  # Get conversation summary

# Use recap to provide context to agents
agent.start(user_message="Continue our discussion", recap=recap)
```

## Examples

### Simple Chat Agent

```python
from gai.asm.agents import ToolUseAgent
from gai.lib.config import config_helper

# Configure LLM
llm_config = config_helper.get_client_config("sonnet-4")

agent = ToolUseAgent(
    agent_name="Assistant",
    llm_config=llm_config
)

# Start conversation with streaming
response = agent.start(user_message="Tell me a joke")
async for chunk in response:
    if isinstance(chunk, str):
        print(chunk, end="", flush=True)

# Continue conversation
response = agent.resume("Tell me another one")
async for chunk in response:
    if isinstance(chunk, str):
        print(chunk, end="", flush=True)
```

### Tool-Using Agent with MCP

```python
from gai.asm.agents import ToolUseAgent
from gai.mcp.client.mcp_client import McpAggregatedClient
from gai.lib.config import config_helper

# Configure MCP clients
aggregated_client = McpAggregatedClient(["mcp-time", "mcp-filesystem"])
tools = await aggregated_client.list_tools()

# Configure LLM
llm_config = config_helper.get_client_config("claude-sonnet-4")

agent = ToolUseAgent(
    agent_name="FileHelper",
    llm_config=llm_config,
    aggregated_client=aggregated_client
)

# Agent can use MCP tools automatically
response = agent.start(user_message="What time is it in Singapore?")
async for chunk in response:
    if isinstance(chunk, str):
        print(chunk, end="", flush=True)
    elif isinstance(chunk, list):  # Tool calls
        for tool in chunk:
            print(f'Tool: "{tool["name"]}"')

# Handle user input requests
try:
    response = agent.resume()
    async for chunk in response:
        print(chunk, end="", flush=True)
except PendingUserInputError:
    # Agent is waiting for user input
    response = agent.resume("Use SGT timezone")
    async for chunk in response:
        print(chunk, end="", flush=True)
```

### Multi-Agent Session

```python
from gai.sessions import SessionManager
from gai.sessions.operations.chat import ChatSender, ChatResponder
from gai.sessions.operations.handshake import HandshakeSender
import asyncio

# Initialize session manager
session_mgr = SessionManager(file_path="dialogue.json")
await session_mgr.start()

# Create conversation plan
plan = HandshakeSender.create_plan("""
    User ->> Sara
    Sara ->> Diana
""")

# Create agent nodes
async def create_agent_node(name, session_mgr, plan):
    from gai.asm.agents import ToolUseAgent
    from gai.lib.config import config_helper

    async def input_handler(message):
        agent = ToolUseAgent(
            agent_name=name,
            llm_config=config_helper.get_client_config("sonnet-4")
        )
        response = agent.start(user_message=message.body.content)

        async def streamer():
            async for chunk in response:
                if isinstance(chunk, str) and chunk:
                    yield chunk
        return streamer()

    node = ChatResponder(node_name=name, session_mgr=session_mgr)
    await node.subscribe(input_chunks_callback=input_handler)
    node.plans[plan.dialogue_id] = plan.model_copy()
    return node

# Register agents
sara = await create_agent_node("Sara", session_mgr, plan)
diana = await create_agent_node("Diana", session_mgr, plan)

# Create user sender
user = ChatSender(node_name="User", session_mgr=session_mgr)
output_queue = asyncio.Queue()

await user.subscribe(output_chunks_callback=lambda msg: output_queue.put_nowait(msg))

# Start chain conversation
await user.chat_send(
    user_message="Tell me a story about a dragon and knight",
    plan=plan
)

# Stream responses from each agent
while True:
    chunk = await output_queue.get()
    if chunk.body.chunk != "<eom>":
        print(chunk.body.chunk, end="", flush=True)
    else:
        print(f"\n--- {chunk.header.sender} finished ---\n")
        if not await user.next():  # Move to next agent
            break
```

## Testing

### Unit Tests

```bash
pytest test/unittest/
```

### Integration Tests

```bash
pytest test/integrationtest/
```

### Smoke Test

```bash
python test/gai_sdk_smoke_test.py
```

## Documentation

-   **State Machine Guide**: See `test/integrationtest/1_agentic_state_machine.ipynb`
-   **Multi-Agent Examples**: Check `test/integrationtest/3_multi_agent_session.ipynb`
-   **API Reference**: Generated from docstrings in source code

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Run the test suite
5. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Links

-   **Documentation**: https://kakkoii1337.github.io/gai
-   **PyPI Package**: https://pypi.org/project/gai-sdk/
-   **Issues**: Submit bug reports and feature requests via GitHub issues
