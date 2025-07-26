import json
from gai.messages.typing import MessagePydantic, DefaultBodyPydantic, ChatSendBodyPydantic, MonologueBodyPydantic
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
            body=SendBodyPydantic(
                recipient="Sara", content="Tell me a story about a brave knight.")
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


def test_message_helper_get_message_length():

    # a) Find length of message with content = `None`

    messages = [
        MessagePydantic(
            body=DefaultBodyPydantic()
        )
    ]
    chat_messages = message_helper.convert_to_chat_messages(messages)
    assert message_helper.get_messages_length(chat_messages) == 0

    # b) Find length of message where body has no 'content' field

    from gai.messages.typing import register_body, get_message_cls

    @register_body
    class NonContentBodyPydantic(BaseModel):
        type: Literal["non_content"] = "non_content"
        description: str

    NewMessagePydanticType = get_message_cls()

    messages = [
        NewMessagePydanticType(
            body=NonContentBodyPydantic(
                description="This message has no content field.")
        )
    ]

    chat_messages = message_helper.convert_to_chat_messages(messages)
    assert message_helper.get_messages_length(chat_messages) == 0

    # c) Find length of message with text content

    messages = [
        MessagePydantic(
            body=ChatSendBodyPydantic(
                dialogue_id="12345",
                round_no=0,
                step_no=0,
                role="user",
                content="This is a message with content length = 76."
            )
        )
    ]

    chat_messages = message_helper.convert_to_chat_messages(messages)
    assert message_helper.get_messages_length(chat_messages) == 76


def test_shrink_messages():

    # a) Shrink messages with no content

    messages = []
    shrunk = message_helper.shrink_messages(messages)
    assert shrunk == []

    # b) Shrink messages begin with assistant message should throw error

    messages = [
        {
            "role": "assistant",
            "content": "This is an assistant message.",
        }
    ]
    try:
        shrunk = message_helper.shrink_messages(messages)
    except ValueError as e:
        assert "First message is not a user message" in str(e)

    # c) Shrink messages containing one user message should return the same message

    messages = [
        {
            "role": "user",
            "content": "You are a unit test generating agent. You are tasked to generate unit tests for the project.",
        }
    ]
    shrunk = message_helper.shrink_messages(messages)
    assert json.dumps(shrunk) == json.dumps(messages)

    # d) If shrink messages ends with an assistant message, the assistant message should be removed

    messages = [
        {
            "role": "user",
            "content": "You are a unit test generating agent. You are tasked to generate unit tests for the project.",
        },
        {
            "role": "assistant",
            "content": "Let me check the project structure and files.",
        }
    ]
    shrunk = message_helper.shrink_messages(messages)
    assert len(shrunk) == 1
    assert shrunk[0]["role"] == "user"

    # e) If messages are too long, they will be shrunk to fit within the limit

    messages = [
        {
            "role": "user",
            "content": "You are a unit test generating agent. You are tasked to generate unit tests for the project.",
        },
        {
            "role": "assistant",
            "content": [
                {
                    "text": "Let me check the project structure and files.",
                    "type": "text"
                },
                {
                    "id": "toolu_01XM8an6Qn1rVhRxzhJmgpdn",
                    "input": {},
                    "name": "list_allowed_directories",
                    "type": "tool_use"
                }
            ]
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_01XM8an6Qn1rVhRxzhJmgpdn",
                    "content": "Allowed directories:\n/workspace/projects\n/workspace/tests"
                }
            ]

        },
        {
            "role": "assistant",
            "content": [
                {
                    "id": "toolu_01EMHCds7Gpxgujwe9nSpYLW",
                    "input": {
                        "path": "/workspace"
                    },
                    "name": "list_directory",
                    "type": "tool_use"
                }
            ]
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_01EMHCds7Gpxgujwe9nSpYLW",
                    "content": "[FILE] .gitignore\n"
                }
            ]

        }
    ]
    message_helper.get_messages_length(messages)
    print(
        f"Total length before shrinking: {message_helper.get_messages_length(messages)}")

    """
    This test comprise of 5 messages with a total length of 822 characters so it won't fit into a limit_len of 800 characters.
    It will preserve the earliest user message, so it need to remove message[1] which is the assistant message.
    But in order for the messages to alternate between user and assistant, it will also have to remove message[2] which is the user message.
    So the final result should be 3 messages = {user[0],assistant[3],user[4]}
    """

    shrunk = message_helper.shrink_messages(messages, limit_len=800)
    assert len(shrunk) == 3
    assert shrunk[0]["role"] == "user"
    assert shrunk[1]["role"] == "assistant"
    assert shrunk[1]["content"][0]["type"] == "tool_use"
    assert shrunk[1]["content"][0]["id"] == "toolu_01EMHCds7Gpxgujwe9nSpYLW"
    assert shrunk[2]["role"] == "user"
    assert shrunk[2]["content"][0]["type"] == "tool_result"
    assert shrunk[2]["content"][0]["tool_use_id"] == "toolu_01EMHCds7Gpxgujwe9nSpYLW"


def test_more_shrink_messages_from_file_monologue_1(request):
    import os
    from gai.lib.tests import get_local_datadir
    from gai.messages import MessagePydantic
    file_path = os.path.join(get_local_datadir(request), "monologue_1.log")
    with open(file_path, "r") as f:
        monologue_file = json.load(f)
        messages = [MessagePydantic(**msg)
                    for msg in monologue_file["messages"]]
    chat_messages = message_helper.convert_to_chat_messages(messages)
    message_helper.shrink_messages(chat_messages, limit_len=300000)


def test_more_shrink_messages_from_file_monologue_2(request):
    import os
    from gai.lib.tests import get_local_datadir
    from gai.messages import MessagePydantic
    file_path = os.path.join(get_local_datadir(request), "monologue_2.log")
    with open(file_path, "r") as f:
        monologue_file = json.load(f)
        messages = [MessagePydantic(**msg)
                    for msg in monologue_file["messages"]]
    chat_messages = message_helper.convert_to_chat_messages(messages)
    message_helper.shrink_messages(chat_messages, limit_len=300000)


def test_more_shrink_messages_from_file_monologue_3(request):
    import os
    from gai.lib.tests import get_local_datadir
    from gai.messages import MessagePydantic
    file_path = os.path.join(get_local_datadir(request), "monologue_3.log")
    with open(file_path, "r") as f:
        monologue_file = json.load(f)
        messages = [MessagePydantic(**msg)
                    for msg in monologue_file["messages"]]
    chat_messages = message_helper.convert_to_chat_messages(messages)
    message_helper.shrink_messages(chat_messages, limit_len=300000)
