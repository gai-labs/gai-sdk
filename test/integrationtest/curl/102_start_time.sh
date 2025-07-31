curl -X POST http://localhost:8000/asm/start \
    -H "Content-Type: application/json" \
    --no-buffer \
    -d '{
        "agent_name": "AgentX",
        "model_name": "sonnet-4",
        "mcp_names": ["mcp-pseudo", "mcp-filesystem", "mcp-time"],
        "user_message": "What is the current time here?"
    }' \
    -w "\n"