curl -X POST http://localhost:8000/asm/start \
    -H "Content-Type: application/json" \
    --no-buffer \
    -d '{
        "agent_name": "AgentX",
        "model_name": "sonnet-4",
        "mcp_names": ["mcp-pseudo", "mcp-filesystem", "mcp-time"],
        "user_message": "What is the current time? Let me know if you need anything from me to help you answer."
    }' \
    -w "\n"