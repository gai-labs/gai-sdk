import os
import uuid
import pytest

from gai.lib.tests import make_local_tmp
from gai.messages.dialogue import FileDialogue
from gai.lib.constants import DEFAULT_GUID
from gai.messages.typing import MessagePydantic, DefaultBodyPydantic


def test_file_dialogue_creates_file_for_custom_id(request):
    # Arrange: per-test temp directory
    tmp_dir = make_local_tmp(request)
    dialogue_id = str(uuid.uuid4())
    file_path = os.path.join(tmp_dir, f"{dialogue_id}.json")

    # Act: initialize with a custom dialogue_id
    dialogue = FileDialogue(dialogue_id=dialogue_id, file_path=file_path)

    # Assert: the backing file was created
    assert os.path.exists(file_path), f"Expected {file_path} to exist"

    # And that its list_messages() is empty by default
    messages = dialogue.list_messages()
    assert isinstance(messages, list)
    assert len(messages) == 0


def test_file_dialogue_creates_default_file_and_is_empty(request):
    # Arrange: fresh temp dir for this test
    tmp_dir = make_local_tmp(request)
    file_path = os.path.join(tmp_dir, f"{DEFAULT_GUID}.json")

    # Act: initialize without passing dialogue_id; uses DEFAULT_GUID
    dialogue = FileDialogue(file_path=file_path)

    # Assert: the file was created at the expected path
    assert os.path.exists(file_path), f"Expected dialogue file at {file_path}"

    # And that listing messages yields an empty list
    messages = dialogue.list_messages()
    assert isinstance(messages, list)
    assert len(messages) == 0


USER_STORY = (
    "Once upon a time, in a small village, there lived a kind-hearted girl named Sara. "
    "She loved helping others and often spent her days tending to the village garden, "
    "where she grew beautiful flowers and vegetables. One day, a traveler passed through "
    "the village and was captivated by the vibrant colors of Sara's garden. He asked her "
    "for directions to the nearest town, and in return, he gifted her a magical seed that "
    "would grow into a tree that bore fruit of wisdom. From that day on, Sara not only "
    "tended to her garden but also shared the wisdom from the magical tree with everyone "
    "in the village."
)

DIANA_STORY = (
    "One day, while tending to her garden, Sara met another AI assistant named Diana. "
    "Diana was curious about the magical seed and the wisdom it bore. Sara shared her "
    "story and the wisdom she had learned from the tree. Diana was fascinated and offered "
    "to help Sara spread the wisdom even further, using her own abilities to assist with "
    "various tasks and provide information on a wide range of topics. Together, they became "
    "a powerful team, helping the villagers and travelers alike with their knowledge and kindness."
)


def test_file_dialogue_add_and_list_messages(request):
    # Arrange
    tmp_dir = make_local_tmp(request)
    file_path = os.path.join(tmp_dir, f"{DEFAULT_GUID}.json")
    dialogue = FileDialogue(file_path=file_path)

    # Ensure starting fresh
    dialogue.reset()

    # Act & Assert: first user message
    umsg = dialogue.add_user_message(
        recipient="Sara", content="Please tell me a one paragraph story."
    )
    assert umsg.id is not None
    assert umsg.header.sender == "User"
    assert umsg.header.recipient == "Sara"
    assert umsg.header.timestamp > 0
    assert umsg.header.order == 0
    assert umsg.body.type == "chat.send"
    assert umsg.body.dialogue_id == DEFAULT_GUID
    assert umsg.body.round_no == 0
    assert umsg.body.step_no == 0
    assert umsg.body.content == "Sara, Please tell me a one paragraph story."

    # Act & Assert: first assistant message
    amsg = dialogue.add_assistant_message(
        sender="Sara", chunk="<eom>", content=USER_STORY
    )
    assert amsg.id is not None
    assert amsg.header.sender == "Sara"
    assert amsg.header.recipient == "User"
    assert amsg.header.timestamp > 0
    assert amsg.header.order == 1
    assert amsg.body.type == "chat.reply"
    assert amsg.body.dialogue_id == DEFAULT_GUID
    assert amsg.body.round_no == 0
    assert amsg.body.step_no == 1
    assert amsg.body.content == USER_STORY

    # Act & Assert: second user message
    umsg2 = dialogue.add_user_message(recipient="Diana", content="Please continue.")
    assert umsg2.id is not None
    assert umsg2.header.sender == "User"
    assert umsg2.header.recipient == "Diana"
    assert umsg2.header.timestamp > 0
    assert umsg2.header.order == 2
    assert umsg2.body.type == "chat.send"
    assert umsg2.body.dialogue_id == DEFAULT_GUID
    assert umsg2.body.round_no == 1
    assert umsg2.body.step_no == 0
    assert umsg2.body.content == "Diana, Please continue."

    # Act & Assert: second assistant message
    amsg2 = dialogue.add_assistant_message(
        sender="Diana", chunk="<eom>", content=DIANA_STORY
    )
    assert amsg2.id is not None
    assert amsg2.header.sender == "Diana"
    assert amsg2.header.recipient == "User"
    assert amsg2.header.timestamp > 0
    assert amsg2.header.order == 3
    assert amsg2.body.type == "chat.reply"
    assert amsg2.body.dialogue_id == DEFAULT_GUID
    assert amsg2.body.round_no == 1
    assert amsg2.body.step_no == 1
    assert amsg2.body.content == DIANA_STORY

    # Finally, check that list_messages returns all four
    all_msgs = dialogue.list_messages()
    assert isinstance(all_msgs, list)
    assert len(all_msgs) == 4


def test_file_dialogue_delete_message(request):
    # Arrange: make a fresh tmp dir and seed with 3 messages
    tmp_dir = make_local_tmp(request)
    file_path = os.path.join(tmp_dir, "dialogue.json")

    initial_msgs = [
        MessagePydantic(body=DefaultBodyPydantic(content="first")),
        MessagePydantic(body=DefaultBodyPydantic(content="second")),
        MessagePydantic(body=DefaultBodyPydantic(content="third")),
    ]
    dialogue = FileDialogue(file_path=file_path, messages=initial_msgs)

    # Wait a second to ensure the message store file is created
    import time

    time.sleep(1)

    # Sanity check
    msgs = dialogue.list_messages()
    assert len(msgs) == 3

    # Act: delete the first message
    to_delete = msgs[0].id
    dialogue.delete_message(to_delete)

    # Assert: one fewer, and none have the deleted ID
    remaining = dialogue.list_messages()
    assert len(remaining) == 2
    assert not any(m.id == to_delete for m in remaining)


def test_file_dialogue_batch_messages(request):
    # Arrange: per-test temp directory
    tmp_dir = make_local_tmp(request)
    file_path = os.path.join(tmp_dir, f"{DEFAULT_GUID}.json")

    # Prepare a batch of 4 messages (note: two share the same id on purpose)
    messages = [
        MessagePydantic(
            **{
                "id": "b1e5f98c-f6eb-47de-a6e2-387510d970f9",
                "header": {
                    "sender": "User",
                    "recipient": "Sara",
                    "timestamp": 1751308157.270983,
                    "order": 0,
                },
                "body": {
                    "type": "chat.send",
                    "dialogue_id": DEFAULT_GUID,
                    "round_no": 0,
                    "step_no": 0,
                    "role": "user",
                    "content": "Please introduce yourselves. Sara, it is your turn.",
                },
            }
        ),
        MessagePydantic(
            **{
                "id": "abbc7961-45dc-4973-aaf4-a6224ed35d37",
                "header": {
                    "sender": "Sara",
                    "recipient": "User",
                    "timestamp": 1751308167.3488164,
                    "order": 1,
                },
                "body": {
                    "type": "chat.reply",
                    "dialogue_id": DEFAULT_GUID,
                    "round_no": 0,
                    "step_no": 1,
                    "chunk_no": 10,
                    "chunk": "<eom>",
                    "role": "assistant",
                    "content": "Hi there! I am Sara, your AI assistant. I am here to help you with any questions or tasks you have.",
                },
            }
        ),
        MessagePydantic(
            **{
                "id": "b1e5f98c-f6eb-47de-a6e2-387510d970f9",
                "header": {
                    "sender": "User",
                    "recipient": "Diana",
                    "timestamp": 1751308157.270983,
                    "order": 2,
                },
                "body": {
                    "type": "chat.send",
                    "dialogue_id": DEFAULT_GUID,
                    "round_no": 0,
                    "step_no": 2,
                    "role": "user",
                    "content": "Please introduce yourselves. Diana, it is your turn.",
                },
            }
        ),
        MessagePydantic(
            **{
                "id": "c2d3e4f5-6a7b-8c9d-a0b1-c2d3e4f5a6b7",
                "header": {
                    "sender": "Diana",
                    "recipient": "User",
                    "timestamp": 1751308177.456789,
                    "order": 3,
                },
                "body": {
                    "type": "chat.reply",
                    "dialogue_id": DEFAULT_GUID,
                    "round_no": 0,
                    "step_no": 3,
                    "chunk_no": 12,
                    "chunk": "<eom>",
                    "role": "assistant",
                    "content": "Hello! I am Diana, another AI assistant. I can assist you with various tasks and provide information on a wide range of topics.",
                },
            }
        ),
    ]

    # Act: create a FileDialogue from those messages
    dialogue = FileDialogue(agent_name="User", file_path=file_path, messages=messages)
    loaded = dialogue.list_messages()

    # Assert: all 4 messages are present and in the correct order
    assert len(loaded) == 4

    assert (
        loaded[0].body.content == "Please introduce yourselves. Sara, it is your turn."
    )
    assert (
        loaded[1].body.content
        == "Hi there! I am Sara, your AI assistant. I am here to help you with any questions or tasks you have."
    )
    assert (
        loaded[2].body.content == "Please introduce yourselves. Diana, it is your turn."
    )
    assert (
        loaded[3].body.content
        == "Hello! I am Diana, another AI assistant. I can assist you with various tasks and provide information on a wide range of topics."
    )
