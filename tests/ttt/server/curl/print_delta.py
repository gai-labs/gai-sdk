import sys
import json

for line in sys.stdin:
    result = json.loads(line)
    try:
        content = result["choices"][0]["delta"]["content"]
        if content:
            print(content, end="", flush=True)
    except Exception as e:
        print("Error: ", e,". Cannot parse ", result, file=sys.stderr, flush=True)
