curl -X POST http://localhost:8000/asm/start \
    -H "Content-Type: application/json" \
    --no-buffer \
    -d '{
        "agent_name": "AgentX",
        "model_name": "sonnet-4",
        "mcp_names": ["mcp-pseudo", "mcp-filesystem", "mcp-web"],
        "user_message": "It is a very nice weather in Singapore right now."
    }' \
    -w "\n"