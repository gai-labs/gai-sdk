# AnthropicToolUseState Documentation

## Purpose

The `AnthropicToolUseState` is a crucial component in the GAI (General AI) framework that handles tool use interactions with Anthropic's Claude models. This state manages the process of executing tools through MCP (Message Control Protocol) clients and streaming the LLM's responses back to users.

## Key Responsibilities

1. **Tool Execution**: Execute MCP tools based on the last tool calls from the LLM
2. **User Input Handling**: Handle special "user_input" pseudo tool calls for interactive sessions  
3. **Response Streaming**: Create async generators that stream LLM responses in real-time
4. **Session Flow Control**: Determine when to continue, terminate, or wait for user input

## Usage Flow

The state follows a decision tree pattern based on the conversation context:

### Case 1: Terminated Session
- **Condition**: `monologue.is_terminated()` returns `True`
- **Action**: Sets `streamer` to `None` and exits
- **Purpose**: Prevents unnecessary API calls when conversation is complete

### Case 2a: Pending User Input  
- **Condition**: Last tool call is "user_input" but no `user_message` in state_bag
- **Action**: Sets `streamer` to `None` and waits
- **Purpose**: Handles interactive sessions where LLM needs user clarification

### Case 2b: User Input Provided
- **Condition**: Last tool call is "user_input" and `user_message` exists in state_bag
- **Action**: Creates pseudo tool result from user message and streams LLM response
- **Purpose**: Continues conversation with user-provided information

### Case 3: Normal Tool Execution
- **Condition**: Last tool calls are for real MCP tools
- **Action**: Executes tools via MCP client and streams LLM response  
- **Purpose**: Standard tool use workflow

## Key Methods

### `_use_tool(last_tool_calls)`
- Executes actual MCP tools
- Handles different result content formats
- Stores results in `machine.state_bag["tool_results"]`
- Returns formatted tool results for the LLM

### `_make_user_input_tool_result(last_tool_calls)`
- Creates pseudo tool results for user input
- Extracts user message from state_bag
- Returns formatted tool result without actual tool execution

### `run_async()`
- Main entry point for the state
- Determines execution path based on conversation state
- Creates streaming generators for real-time response delivery
- Manages monologue updates and state transitions

## Dependencies

- **LLM Client**: Uses `AsyncOpenAI` for Anthropic API communication
- **MCP Client**: Interfaces with Message Control Protocol for tool execution
- **Monologue**: Manages conversation history and message flow
- **State Machine**: Coordinates with parent state machine for flow control

## Testing Strategy

The test suite covers all execution paths and edge cases:

1. **State Decision Logic**: Tests all branching conditions
2. **Tool Execution**: Validates MCP tool calling and result formatting
3. **User Input Handling**: Tests pseudo tool result creation
4. **Error Handling**: Verifies proper exception propagation
5. **Content Format Handling**: Tests various MCP result formats
6. **Streaming Behavior**: Validates async generator creation and flow

## Integration Points

- **Input Requirements**: `llm_config`, `mcp_client`
- **State Bag Dependencies**: `user_message` (optional)
- **Output Products**: `streamer`, `tool_results`, `get_assistant_message` (conditional)
- **Monologue Updates**: Adds tool results and assistant messages to conversation

## Error Handling

- MCP client errors are propagated upward
- Empty message lists raise `ValueError` with descriptive message
- Tool execution failures are logged and re-raised
- Streaming errors are handled by retry policies

This state is essential for creating interactive AI assistants that can use tools while maintaining smooth, real-time communication with users.