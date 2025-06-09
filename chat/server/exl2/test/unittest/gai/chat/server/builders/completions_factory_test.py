import json

def test_convert_message_content_with_finish_reason_stop():
    # arrange
    job='{"job": "ExLlamaV2DynamicJob #4", "stage": "streaming", "eos": true, "serial": 4, "eos_reason": "stop_string", "full_completion": " Once upon a time, in a small village nestled between the mountains, lived a young girl named Mia. She was known for her kind heart and her love for animals. One day, a wounded bird fell from the sky into her garden. Mia took it in, nursed it back to health, and released it back into the wild. This act of kindness sparked a movement in her village, inspiring others to help animals in need, and Mia became a beacon of compassion and care for all creatures great and small.", "new_tokens": 112, "prompt_tokens": 15, "time_enqueued": 7.772445678710938e-05, "time_prefill": 0.0003190040588378906, "time_generate": 3.033987045288086, "cached_pages": 0, "cached_tokens": 13}'
    result=json.loads(job)

    # act
    from gai.chat.server.builders import CompletionsFactory
    response = CompletionsFactory().message.build_content(result)

    # assert
    assert response.choices[0].finish_reason == "stop"
    assert response.choices[0].message.content == " Once upon a time, in a small village nestled between the mountains, lived a young girl named Mia. She was known for her kind heart and her love for animals. One day, a wounded bird fell from the sky into her garden. Mia took it in, nursed it back to health, and released it back into the wild. This act of kindness sparked a movement in her village, inspiring others to help animals in need, and Mia became a beacon of compassion and care for all creatures great and small."

def test_convert_message_content_with_finish_reason_length():
    job='{"job": "ExLlamaV2DynamicJob #6", "stage": "streaming", "eos": true, "serial": 6, "eos_reason": "max_new_tokens", "text": " nest", "full_completion": " Once upon a time, in a small village nest", "new_tokens": 10, "prompt_tokens": 15, "time_enqueued": 6.866455078125e-05, "time_prefill": 0.0002491474151611328, "time_generate": 0.4717264175415039, "cached_pages": 0, "cached_tokens": 13}'
    result=json.loads(job)

    # act
    from gai.chat.server.builders import CompletionsFactory
    response = CompletionsFactory().message.build_content(result)

    # assert
    assert response.choices[0].finish_reason == "length"
    assert response.choices[0].message.content == " Once upon a time, in a small village nest"

def test_convert_chunk_content_with_finish_reason_length():
    chunks=[' Once',
        ' upon',
        ' a',
        ' time',
        ',',
        ' in',
        ' a',
        ' small',
        ' village',
        ' nest',
        {"job": "ExLlamaV2DynamicJob #9", "stage": "streaming", "eos": True, "serial": 9, "eos_reason": "max_new_tokens", "text": " nest", "full_completion": " Once upon a time, in a small village nest", "new_tokens": 10, "prompt_tokens": 15, "time_enqueued": 0.00022840499877929688, "time_prefill": 0.00025272369384765625, "time_generate": 0.35219478607177734, "cached_pages": 0, "cached_tokens": 13}
        ]
    
    # act
    from gai.chat.server.builders import CompletionsFactory
    first=True
    for i,chunk in enumerate(CompletionsFactory().chunk.build_stream(chunks)):
        if first:
            # first chunk
            assert chunk.choices[0].delta.role == "assistant"
            assert chunk.choices[0].delta.content == ""
            assert chunk.choices[0].finish_reason is None
        elif not first and type(chunk) is str:
            assert chunk.choices[0].delta.role is None
            assert chunk.choices[0].delta.content
            assert chunk.choices[0].finish_reason is None
        elif i == len(chunks):
            # last chunk
            assert chunk.choices[0].delta.role is None
            assert chunk.choices[0].delta.content is None
            assert chunk.choices[0].finish_reason == "length"            
        print(chunk)
        first=False

def test_convert_message_toolcall():
    #job='{"job": "ExLlamaV2DynamicJob #2", "stage": "streaming", "eos": true, "serial": 2, "eos_reason": "stop_token", "full_completion": " {\\"name\\": \\"google\\", \\"arguments\\": {\\"search_query\\": \\"current time Singapore\\"}}", "new_tokens": 21, "prompt_tokens": 336, "time_enqueued": 6.699562072753906e-05, "time_prefill": 0.0002429485321044922, "time_generate": 0.7014470100402832, "cached_pages": 0, "cached_tokens": 334}'
    job="""{
            "job": "ExLlamaV2DynamicJob #2", 
            "stage": "streaming", 
            "eos": true, 
            "serial": 2, 
            "eos_reason": "stop_token", 
            "full_completion": 
                "{ \\"function\\": {\\"name\\": \\"google\\", \\"arguments\\": {\\"search_query\\": \\"current time Singapore\\"}} }", 
            "new_tokens": 21, 
            "prompt_tokens": 336, 
            "time_enqueued": 6.699562072753906e-05, 
            "time_prefill": 0.0002429485321044922, 
            "time_generate": 0.7014470100402832, 
            "cached_pages": 0, 
            "cached_tokens": 334
        }"""
    result=json.loads(job)

    # Convert to OpenAI Completion API response
    from gai.chat.server.builders import CompletionsFactory
    response = CompletionsFactory().message.build_toolcall(result)
    print(response)

    assert response.choices[0].finish_reason == "tool_calls"
    assert response.choices[0].message.tool_calls[0].type == "function"
    assert response.choices[0].message.tool_calls[0].function.name == "google"
    assert response.choices[0].message.tool_calls[0].function.arguments == json.dumps({"search_query": "current time Singapore"})

def test_convert_schema():
    job='{"job": "ExLlamaV2DynamicJob #1", "stage": "streaming", "eos": true, "serial": 1, "eos_reason": "stop_token", "full_completion": " {\\n    \\"title\\": \\"Foundation\\",\\n    \\"summary\\": \\"Foundation is a science fiction novel by Isaac Asimov. It is the first published in his Foundation Trilogy. The book tells the early story of the Foundation, an institute founded by psychohistorian Hari Seldon to preserve the best of galactic civilization after the collapse of the Galactic Empire.\\",\\n    \\"author\\": \\"Isaac Asimov\\",\\n    \\"published_year\\": 1951\\n}", "new_tokens": 109, "prompt_tokens": 259, "time_enqueued": 3.1598808765411377, "time_prefill": 0.432448148727417, "time_generate": 3.0512359142303467, "cached_pages": 0, "cached_tokens": 0}'
    result=json.loads(job)

    # Convert to OpenAI Completion API response
    from gai.chat.server.builders import CompletionsFactory
    response = CompletionsFactory().message.build_content(result)
    print(response)
    assert response.choices[0].finish_reason == "stop"
    assert response.choices[0].message.content == ' {\n    "title": "Foundation",\n    "summary": "Foundation is a science fiction novel by Isaac Asimov. It is the first published in his Foundation Trilogy. The book tells the early story of the Foundation, an institute founded by psychohistorian Hari Seldon to preserve the best of galactic civilization after the collapse of the Galactic Empire.",\n    "author": "Isaac Asimov",\n    "published_year": 1951\n}'