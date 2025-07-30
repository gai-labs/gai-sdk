# ToolUseAgent Test Documentation

## Purpose

The `ToolUseAgent` is an Anthropic-powered agent that implements a state machine for handling tool-based interactions. It orchestrates conversations with an LLM (Large Language Model) that can call tools to accomplish tasks, providing a sophisticated flow for handling user queries that require external tool execution.

## Core Functionality

### State Machine Flow

The agent implements the following state transitions:

```
INIT --> HAS_MESSAGE
HAS_MESSAGE --> CHAT: condition_true (when user message exists)
HAS_MESSAGE --> TOOL_USE: condition_false (when no user message, resuming tool execution)

CHAT --> IS_TOOL_CALL

IS_TOOL_CALL --> TOOL_USE: condition_true (when LLM wants to call a tool)
IS_TOOL_CALL --> FINAL: condition_false (when no tool call needed)

TOOL_USE --> FINAL
```

### Key Components

1. **Monologue**: Manages conversation history and message persistence
2. **MCP Client**: Aggregated client for tool execution via MCP (Model Context Protocol)
3. **LLM Config**: Configuration for the Anthropic Claude model
4. **State Machine**: Orchestrates the conversation flow

### Primary Methods

1. **`start(user_message, recap=None)`**: Initiates a new conversation
2. **`resume(user_message=None)`**: Continues an existing conversation
3. **`interrupt(user_message)`**: Handles user interruptions during tool execution
4. **`final_output()`**: Returns the final assistant message

### Usage Scenarios

#### Scenario 1: Direct Tool Use with Context
- User provides sufficient context for the agent to infer tool usage
- Agent directly proceeds to use appropriate tools
- No user input required during execution

#### Scenario 2: Agent Requests User Input
- User query lacks sufficient context
- Agent asks for clarification using `user_input` tool
- User must provide input before agent can continue

#### Scenario 3: User Interruption
- User interrupts agent during tool execution
- Agent handles the interruption while maintaining original task context
- Agent returns to original task after handling interruption

## Testing Strategy

The tests should verify:

1. **State transitions**: Proper flow through the state machine
2. **Tool calling**: Correct invocation of MCP tools
3. **Streaming responses**: Proper handling of streamed LLM responses
4. **Message persistence**: Correct monologue management
5. **Error handling**: Graceful handling of various error conditions
6. **User interactions**: Proper handling of start/resume/interrupt flows

## Mock Strategy

Tests use mocked Anthropic API responses to avoid actual API calls while testing the agent's logic and state management. The mocks simulate:

- Text generation responses
- Tool calling sequences
- Streaming chunks
- Error conditions

This approach ensures tests are reliable, fast, and don't depend on external services.