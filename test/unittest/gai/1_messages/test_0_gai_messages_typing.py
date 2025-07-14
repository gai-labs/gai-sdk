from gai.messages.typing import MessagePydantic, DefaultBodyPydantic
from pydantic import BaseModel
from typing import Literal
from gai.messages.typing import register_body, get_message_cls
from gai.messages import message_helper

def test_default_body_as_message_body():
    # Can wrap DefaultBodyPydantic directly in MessagePydantic
    msg = MessagePydantic(body=DefaultBodyPydantic(content="hello"))
    assert isinstance(msg.body, DefaultBodyPydantic)
    assert msg.body.content == "hello"
    assert msg.body.type == "default"

def test_default_body_serialization_deserialization():
    # Prepare JSON with a default body
    state_json = {
        'id': 'a0e5f98c-f6eb-47de-a6e2-387510d970f9',
        'header': {
            'sender': 'User',
            'recipient': 'Assistant',
            'timestamp': 1752213128.0972457,
            'order': 0
        },
        'body': {
            'type': 'default',
            'content': 'Tell me a one paragraph story'
        }
    }
    # Deserialize
    msg = MessagePydantic(**state_json)
    # Type and content checks
    assert isinstance(msg.body, DefaultBodyPydantic)
    assert msg.id == state_json['id']
    assert msg.header.sender == state_json['header']['sender']
    assert msg.body.content == state_json['body']['content']

def test_custom_ping_body_registration_and_serialization():
    # Dynamically register a new PingBodyPydantic
    @register_body
    class PingBodyPydantic(BaseModel):
        type: Literal["Ping"] = "Ping"
        content: str

    # Rebuild MessagePydantic so it picks up our new body type
    MessagePydantic = get_message_cls()

    # 1) Can instantiate via the model directly
    msg = MessagePydantic(body=PingBodyPydantic(content="Ping"))
    assert isinstance(msg.body, PingBodyPydantic)
    assert msg.body.content == "Ping"
    assert msg.body.type == "Ping"

    # 2) Can serialize → deserialize via the dict form
    jsoned = {
        'id': 'a0e5f98c-f6eb-47de-a6e2-387510d970f9',
        'header': {
            'sender': 'User',
            'recipient': 'Assistant',
            'timestamp': 1752213128.0972457,
            'order': 0
        },
        'body': {
            'type': 'Ping',
            'content': 'Ping'
        }
    }
    msg2 = MessagePydantic(**jsoned)
    assert isinstance(msg2.body, PingBodyPydantic)
    assert msg2.body.content == "Ping"
    assert msg2.body.type == "Ping"
    
def test_message_helper_json_unjson_roundtrip():
    # Dynamically register send and reply body types
    @register_body
    class SendBodyPydantic(BaseModel):
        type: Literal["send"] = "send"
        recipient: str
        content: str

    @register_body
    class ReplyBodyPydantic(BaseModel):
        type: Literal["reply"] = "reply"
        sender: str
        recipient: str
        chunk_no: int
        chunk: str

    # Rebuild MessagePydantic so it picks up our new body types
    MessagePydantic = get_message_cls()

    # Create a couple of messages
    original_messages = [
        MessagePydantic(
            header={
                "sender": "User",
                "recipient": "Sara",
                "timestamp": 1752213128.0972457,
                "order": 0
            },
            body=SendBodyPydantic(recipient="Sara", content="Tell me a story about a brave knight.")
        ),
        MessagePydantic(
            header={
                "sender": "Sara",
                "recipient": "User",
                "timestamp": 1752213128.0972457,
                "order": 1
            },
            body=ReplyBodyPydantic(
                sender="Sara",
                recipient="User",
                chunk_no=0,
                chunk="<eom>"
            )
        )
    ]

    # 1) Serialize list to JSON string
    jsoned = message_helper.json(original_messages)
    assert isinstance(jsoned, str)

    # 2) Deserialize back to model instances
    restored = message_helper.unjson(jsoned, MessagePydantic)
    assert isinstance(restored, list)
    assert len(restored) == len(original_messages)

    # 3) Each item round-trips exactly
    for orig, new in zip(original_messages, restored):
        assert isinstance(new, MessagePydantic)
        # Compare dict dumps for full fidelity
        assert new.model_dump() == orig.model_dump()