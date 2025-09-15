curl -X GET http://localhost:8000/asm/monologue \
    -H "Content-Type: application/json" \
    -d '{
        "agent_name": "AgentX"
    }' \
    -w "\n"