curl -X POST http://localhost:8000/asm/undo \
    -H "Content-Type: application/json" \
    -d '{
        "agent_name": "AgentX"
    }' \
    -w "\n"