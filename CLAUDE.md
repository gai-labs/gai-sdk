# Claude AI Assistant Context

## Project Overview

This project is the **GAI SDK - Generative Agent Infrastructure SDK**, a comprehensive Python SDK for building intelligent multi-agent AI systems. For complete project details, architecture, and usage examples, please refer to the [README.md](README.md).

## Key Information for AI Assistance

### Project Type

-   Python SDK with multi-agent AI capabilities
-   Uses Agentic State Machines (ASM) pattern
-   Integrates with LLMs and Model Context Protocol (MCP)

### Development Environment

-   **Python Version**: 3.10+
-   **Package Manager**: [uv](https://github.com/astral-sh/uv)
-   **Build System**: Uses Makefile

### Common Commands

```bash
# Development setup
make install

# Run tests
make test              # All tests
pytest test/unittest/ # Unit tests only
pytest test/integrationtest/ # Integration tests only

# Build package
make build

# Smoke test
python test/gai_sdk_smoke_test.py
```

### Testing Instructions

-   Use `pytest` for running tests
-   Tests should be placed in `test/ai-generated-tests` under integrationtest or unittest as appropriate following the same structure as the source to be tested.

### Debugging Instructions

-   Always show the full path to the file where the error occurs.
-   Show the code snippet (include line numbers) where the error is raised.
-   Provide root cause explanation
-   Show the fix to be applied to fix the error in the code snippet.
-   Replicate the error in a minimal test case in `test/ai-generated-tests` and verify that it works before suggesting it.

### Project Structure

-   `src/gai/` - Main SDK code
-   `lib/` - Core library
-   `llm/` - LLM integrations
-   `init/` - Project templates
-   `test/` - Test suites

### Architecture Notes

-   Uses **Agentic State Machines** for agent behavior
-   Supports **multi-agent collaboration** via sessions
-   Integrates **MCP tools** for enhanced capabilities
-   Provides **memory management** (Monologue/Dialogue)

For detailed information about features, examples, and API reference, see the main [README.md](README.md).
