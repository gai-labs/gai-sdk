from mock.mock_openai_client import MockOpenAI
openai = MockOpenAI()

def test_extract_generate_text_from_openai():
    messages=[
        {"role":"user","content":"Tell me a one paragraph story"}
    ]
    response = openai.chat.completions.create(model="gpt-4",messages=messages,stream=False)
    assert response.choices[0].message.content == "\"Despite being lost in the dense, mystifying forest for hours, the brave little puppy finally managed to find his way back home, surprising his family who welcomed him with more love than ever before.\""

def test_extract_stream_text_from_openai():
    messages=[
        {"role":"user","content":"Tell me a one paragraph story"}
    ]
    response = openai.chat.completions.create(model="gpt-4",messages=messages,stream=True)
    print("\n===")
    count=0
    for chunk in response:
        if chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)
        count+=1
    print("\n===")
    assert count == 42

def test_extract_generate_text_from_gai():
    messages=[
        {"role":"user","content":"Tell me a one paragraph story"}
    ]
    response = openai.chat.completions.create(model="exllamav2-mistral7b",messages=messages,stream=False)
    assert response.choices[0].message.content == "Under a tree, a little boy shared his last bread with a hungry crow. They became friends, teaching him that kindness can feed more than just Hunger."

def test_extract_stream_text_from_gai():
    messages=[
        {"role":"user","content":"Tell me a one paragraph story"}
    ]
    response = openai.chat.completions.create(model="exllamav2-mistral7b",messages=messages,stream=True)
    print("\n===")
    count=0
    for chunk in response:
        if chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)
        count+=1
    print("\n===")
    assert count == 18

