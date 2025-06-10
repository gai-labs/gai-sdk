boolean_schema={
        "type": "json_schema",
        "json_schema": {
            "name": "predicate_result",             # ← required by the API
            "strict": True,                   # ← optional, enforces no extra fields
            "schema": {
                "type": "object",
                "properties": {
                    "result": {
                        "type": "boolean"
                    }
                },
                "required": ["result"],
                "additionalProperties": False
            }
        }
    }