curl -X POST \
    http://localhost:12031/gen/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -s \
    -N \
    -d "{\"model\":\"exllamav2-mistral7b\", \
        \"messages\": [ \
            {\"role\": \"user\",\"content\": \"What is the current time in Singapore\"}, \
            {\"role\": \"assistant\",\"content\": \"\"} \
        ],\
        \"tools\": [\
            {\
                \"type\": \"function\",\
                \"function\": {\
                    \"name\": \"google\",\
                    \"description\": \"The 'google' function is a powerful tool that allows the AI to gather external information from the internet using Google search. It can be invoked when the AI needs to answer a question or provide information that requires up-to-date, comprehensive, and diverse sources which are not inherently known by the AI. For instance, it can be used to find current date, current news, weather updates, latest sports scores, trending topics, specific facts, or even the current date and time. The usage of this tool should be considered when the user's query implies or explicitly requests recent or wide-ranging data, or when the AI's inherent knowledge base may not have the required or most current information. The 'search_query' parameter should be a concise and accurate representation of the information needed.\",\
                    \"parameters\": {\
                        \"type\": \"object\",\
                        \"properties\": {\
                            \"search_query\": {\
                                \"type\": \"string\",\
                                \"description\": \"The search query to search google with. For example, to find the current date or time, use 'current date' or 'current time' respectively.\"\
                            }\
                        },\
                        \"required\": [\"search_query\"]\
                    }\
                }\
            }\
        ],\
        \"tool_choice\": \"required\"}"