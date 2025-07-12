import time
import uuid
from gai.lib.constants import DEFAULT_GUID
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

def test_create_custom_message_pydantic():
    from typing import Literal
    from gai.messages.typing import get_message_cls,register_body
    @register_body
    class TestBodyType:
        type: Literal["test"]="test"
        dialogue_id: str
        content_type: str
        content: str
    MessagePydantic = get_message_cls()
    message = MessagePydantic(**{
        "id": DEFAULT_GUID,
        "header":{
            "sender": "User",
        },
        "body":{
            "type":"test",
            "dialogue_id": "12345",
            "content_type": "text",
            "content": "Hello, how are you?"
        }
    })    
    print(message)
    assert isinstance(message.body,TestBodyType)
    assert message.header.sender == "User"
    assert message.header.recipient == ""
    assert message.body.type == "test"
    assert message.body.dialogue_id == "12345"
    assert message.body.content_type == "text"
    assert message.body.content == "Hello, how are you?"
   
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