curl -X POST \
    http://localhost:12031/gen/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -s \
    -N \
    -d "{\"model\":\"exllamav2-mistral7b\", \
        \"messages\": [ \
            {\"role\": \"user\",\"content\": \"Foundation is a science fiction novel by American writer \
            Isaac Asimov. It is the first published in his Foundation Trilogy (later \
            expanded into the Foundation series). Foundation is a cycle of five \
            interrelated short stories, first published as a single book by Gnome Press \
            in 1951. Collectively they tell the early story of the Foundation, \
            an institute founded by psychohistorian Hari Seldon to preserve the best \
            of galactic civilization after the collapse of the Galactic Empire.\"}, \
            {\"role\": \"assistant\",\"content\": \"\"} \
        ],\
        \"response_model\": {\"properties\": \
            {\"title\": \
                {\"title\": \"Title\", \"type\": \"string\"}, \
                    \"summary\": {\"title\": \"Summary\", \"type\": \"string\"}, \
                    \"author\": {\"title\": \"Author\", \
                    \"type\": \"string\"\
                }, \
                \"published_year\": {\
                    \"title\": \"Published Year\", \
                    \"type\": \"integer\"}}, \
                \"required\": [\
                    \"title\", \
                    \"summary\", \
                    \"author\", \
                    \"published_year\"\
                ], \
                \"title\": \"Book\", \
                \"type\": \"object\"\
            },\
        \"tool_choice\": \"none\"}"