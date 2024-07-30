echo STREAMING
curl -X POST \
    http://localhost:12031/gen/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -s \
    -N \
    -d "{\"model\":\"exllamav2-mistral7b\", \
        \"messages\": [ \
            {\"role\": \"user\",\"content\": \"Tell me a story.\"}, \
            {\"role\": \"assistant\",\"content\": \"\"} \
        ],\
        \"stream\":true}"
echo
echo NON-STREAMING
curl -X POST \
    http://localhost:12031/gen/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -s \
    -N \
    -d "{\"model\":\"exllamav2-mistral7b\", \
        \"messages\": [ \
            {\"role\": \"user\",\"content\": \"Tell me a story.\"}, \
            {\"role\": \"assistant\",\"content\": \"\"} \
        ],\
        \"tool_choice\": \"none\"}"

