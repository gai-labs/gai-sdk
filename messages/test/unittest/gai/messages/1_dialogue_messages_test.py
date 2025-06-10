import time
import uuid
from gai.messages import MessagePydantic, MessageHeaderPydantic, MessageCounter, SendMessagePydantic, ReplyMessagePydantic    


def test_message_header_pydantic():
    MessageHeaderPydantic(sender="sender",recipient="recipient", timestamp=time.time())

def test_message_pydantic():
    MessagePydantic(
        id = str(uuid.uuid4()), 
        type="test", 
        header=MessageHeaderPydantic(sender="sender",recipient="recipient", timestamp=time.time()), 
        body="This is a test message"
        )

def test_create_non_standard_message_pydantic():
    message = MessagePydantic.from_dict({
        "type":"test",
        "header":{
            "sender": "User",
        },
        "body":{
            "dialogue_id": "12345",
            "content_type": "text",
            "content": "Hello, how are you?"
        }
    })
    print(message)
    assert message.type == "test"
    assert message.header.sender == "User"
    assert message.header.recipient == ""
    assert message.body["dialogue_id"] == "12345"
    assert message.body["content_type"] == "text"
    assert message.body["content"] == "Hello, how are you?"
   
def test_create_send_message_pydantic():
    
    MessageCounter().initialize(1)
    
    message = MessagePydantic.from_dict({
        "type":"send",
        "header":{
            "sender": "User",
        },
        "body":{
            "dialogue_id": "12345",
            "content_type": "text",
            "content": "Hello, how are you?"
        }
    })
    
    assert isinstance(message, SendMessagePydantic)
    assert message.body.message_no == 2
    assert message.body.message_id == "12345.2"
    
def test_create_reply_message_pydantic():
    
    MessageCounter().initialize(2)
    
    message = MessagePydantic.from_dict({
        "type":"reply",
        "header":{
            "sender": "Sara",
        },
        "body":{
            "dialogue_id": "12345",
            "content_type": "text",
            "content": "I'm fine, thank you!"
        }
    })
    assert isinstance(message, ReplyMessagePydantic)
    assert message.body.message_no == 3
    assert message.body.message_id == "12345.3"
    assert message.body.content == "I'm fine, thank you!"
    assert message.body.content_type == "text"
    assert message.body.chunk == "<eom>"