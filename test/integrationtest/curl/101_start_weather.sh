curl -X POST http://localhost:8000/asm/start \
    -H "Content-Type: application/json" \
    --no-buffer \
    -d '{
        "agent_name": "AgentX",
        "model_name": "sonnet-4",
        "mcp_names": ["mcp-pseudo", "mcp-filesystem", "mcp-web"],
        "user_message": "I just want to say that I am in Singapore right now and the weather is great."
    }' \
    -w "\n"