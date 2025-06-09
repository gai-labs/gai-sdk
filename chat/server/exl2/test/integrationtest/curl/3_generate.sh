curl -X POST http://localhost:12031/gen/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "ttt",
    "messages": [
      {"role": "user", "content": "Tell me a one sentence story."}
    ],
    "stream": false
  }'