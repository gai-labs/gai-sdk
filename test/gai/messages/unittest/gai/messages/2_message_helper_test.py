import json
from gai.messages.messages import _message_registry
from gai.messages import MessagePydantic, MessageHeaderPydantic, message_helper, MessageCounter

def test_extract_recap_with_last_n(monkeypatch):

    # This is to prevent the test from failing due to missing registry
    monkeypatch.setitem(_message_registry, "chat.send", MessagePydantic)

    # Scenario 1: Dialogue with 1 user message has no recap.
    # This is because if the last message is a user message, it is excluded as recap as it is not a reply.


    recap = message_helper.extract_recap([
        MessagePydantic.from_dict({
            "id":"001",
            "type":"chat.send",
            "header":{
                "sender": "User",
                "recipient": "Sara",
            },
            "body": {
                "content":"a" * 10
            }
        })
    ], last_n=1)
    assert recap == '[]'

    # Scenario 2: Dialogue ends with a non-user message has recap.
    # This is because if the last message is a non-user message, it is considered a reply from previous user message.

    ## if last_n = 1:

    recap = message_helper.extract_recap([
        MessagePydantic.from_dict({
            "id":"001",
            "type":"chat.send",
            "header":{
                "sender": "User",
                "recipient": "Sara",
            },
            "body": {
                "content":"a" * 10
            }
        }),
        MessagePydantic.from_dict({
            "id":"001",
            "type":"chat.send",
            "header":{
                "sender": "Sara",
                "recipient": "User",
            },
            "body": {
                "content":"b" * 20
            }
        })
    ], last_n=1)

    #[{"Sara": "bbbbbbbbbbbbbbbbbbbb"}]

    assert len(recap) == 34
    
    ## if last_n = 2:
    
    recap = message_helper.extract_recap([
        MessagePydantic.from_dict({
            "id":"001",
            "type":"chat.send",
            "header":{
                "sender": "User",
                "recipient": "Sara",
            },
            "body": {
                "content":"a" * 10
            }
        }),
        MessagePydantic.from_dict({
            "id":"001",
            "type":"chat.send",
            "header":{
                "sender": "Sara",
                "recipient": "User",
            },
            "body": {
                "content":"b" * 20
            }
        })
    ], last_n=2)

    #[{"User": "aaaaaaaaaa"}, {"Sara": "bbbbbbbbbbbbbbbbbbbb"}]

    assert len(recap) == 58
    
    # Therefore the length of recap can be calculated as:
    
    """
    size_of_outer_brackers = 2
    message 1:
        size_of_inner_braces = 2 +
        size_of_inner_quotes = 4 +
        size_of_sender_name = 4 +
        size_of_colon_space = 2 +
        size_of_message_content = 10
    size_of_message_1 = 2 + 4 + 4 + 2 + 10 = 22 +
    comma = 1
    space = 1
    message 2:
        size_of_inner_braces = 2 +
        size_of_inner_quotes = 4 +
        size_of_sender_name = 4 +
        size_of_colon_space = 2 +
        size_of_message_content = 20
    size_of_message_2 = 2 + 4 + 4 + 2 + 20 = 32
    
    Therefore,
        Length of recap = 2 + 22 + 1 + 1 + 32 = 58
    """

def test_extract_recap_with_recap_size():
    # Simulate a dialogue with 4 messages with total size : 37 + 34 + 37 + 19 + 2 + 6 = 135 characters
    
    messages = [
        MessagePydantic(
            id="001",
            type="test",
            header=MessageHeaderPydantic(sender="User", recipient="A", timestamp=None),
            # {"User": "..."} is 25+12 = 37 char
            body={"content":"a" * 25}
        ),
        MessagePydantic(
            id="002",
            type="test",
            header=MessageHeaderPydantic(sender="A", recipient="User", timestamp=None),
            # {"A": "..."} is 25+9 = 34 char
            body={"content":"b" * 25}
        ),
        MessagePydantic(
            id="003",
            type="test",
            header=MessageHeaderPydantic(sender="User", recipient="B", timestamp=None),
            # {"User": "..."} is 25+12 = 37 char
            body={"content":"c" * 25}
        ),
        MessagePydantic(
            id="004",
            type="test",
            header=MessageHeaderPydantic(sender="B", recipient="User", timestamp=None),
            # {"B": "..."} is 10+9 = 19 char
            body={"content":"d" * 10}
        )
    ]
    
    # Scenario 1: Default max_recap_size = 400 and current size = 135 so no truncation
    
    result = message_helper.extract_recap(messages=messages)
    assert len(result) == 135

    # Scenario 2: max_recap matches current size = 135 so no truncation
    
    result = message_helper.extract_recap(messages=messages, max_recap_size=135)
    assert len(result) == 135
    
    # Scenario 3: max_recap = 134 and current size = 135, so oldest message is excluded
    
    result = message_helper.extract_recap(messages=messages, max_recap_size=134)
    assert len(result) == 96

def test_create_user_message():
    mc = MessageCounter()
    mc.initialize(1)
    user_message = message_helper.create_user_send_message(content="Hello, how are you?")
    assert user_message.id is not None
    assert user_message.body.content == "Hello, how are you?"
    assert user_message.body.content_type == "text"
    assert user_message.body.dialogue_id == "00000000-0000-0000-0000-000000000000"
    assert user_message.body.message_no == 2
    
def test_create_assistant_message():
    MessageCounter().initialize(2)
    assistant_message = message_helper.create_assistant_reply_chunk(
        sender="Sara",
        chunk_no=0,
        chunk="<eom>",
        content="Hello, I am fine. How can I help you?",
    )
    assert assistant_message.id is not None
    assert assistant_message.body.content == "Hello, I am fine. How can I help you?"
    assert assistant_message.body.chunk == "<eom>"
    assert assistant_message.body.content_type == "text"
    assert assistant_message.body.dialogue_id == "00000000-0000-0000-0000-000000000000"
    assert assistant_message.body.message_no == 3
    