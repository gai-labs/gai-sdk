from gai.messages.monologue import Monologue
import os
from types import SimpleNamespace
from gai.lib.tests import make_local_tmp
from gai.messages.monologue import FileMonologue
from gai.messages.typing import MessagePydantic


def test_file_monologue_creates_and_loads_empty(request):
    # Arrange
    tmp_dir = make_local_tmp(request)
    agent_name = "TestBot"
    file_path = os.path.join(tmp_dir, f"{agent_name}.json")

    # Act: initialize—should create a new monologue file
    monologue = FileMonologue(agent_name=agent_name, file_path=file_path)

    # Assert: file is created
    assert os.path.exists(file_path), f"Expected monologue file at {file_path}"

    # And listing messages returns an empty list
    messages = monologue.list_messages()
    assert isinstance(messages, list)
    assert len(messages) == 0


def test_add_user_message(request):
    # Arrange: make a clean temp folder for this test
    tmp_dir = make_local_tmp(request)
    agent_name = "TestBot"
    file_path = os.path.join(tmp_dir, f"{agent_name}.json")
    monologue = FileMonologue(agent_name=agent_name, file_path=file_path)

    # Act: add a user message
    mock_state = SimpleNamespace(state="GENERATE", input={
                                 "step": 5}, title="GENERATE")
    monologue.add_user_message("Hello, how are you?", mock_state)
    messages = monologue.list_messages()

    # Assert: exactly one message, with the expected role and content
    assert len(messages) == 1
    body = messages[0].body
    assert body.role == "user"
    assert body.content == "Hello, how are you?"


def test_add_assistant_message(request):
    # Arrange: fresh monologue with one user message
    tmp_dir = make_local_tmp(request)
    agent_name = "TestBot"
    file_path = os.path.join(tmp_dir, f"{agent_name}.json")
    monologue = FileMonologue(agent_name=agent_name, file_path=file_path)

    # seed a user message so assistant will be second
    user_state = SimpleNamespace(state="GENERATE", input={
                                 "step": 5}, title="GENERATE")
    monologue.add_user_message("Hello, how are you?", user_state)

    # Act: add an assistant reply
    assistant_state = SimpleNamespace(
        state="GENERATE", input={"step": 6}, title="GENERATE")
    monologue.add_assistant_message(
        "I'm doing well, thank you!", assistant_state)

    # Assert: we now have exactly two messages, and the second is the assistant’s
    messages = monologue.list_messages()
    assert len(messages) == 2
    assert messages[1].body.role == "assistant"
    assert messages[1].body.content == "I'm doing well, thank you!"


def test_monologue_pop_removes_last_message(request):
    # Arrange: fresh temp dir and monologue
    tmp_dir = make_local_tmp(request)
    agent_name = "TestBot"
    file_path = os.path.join(tmp_dir, f"{agent_name}.json")
    monologue = FileMonologue(agent_name=agent_name, file_path=file_path)

    # Seed with two messages
    user_state = SimpleNamespace(
        state="S1", input={"step": 0}, title="UserState")
    monologue.add_user_message("First message", user_state)
    assistant_state = SimpleNamespace(
        state="S2", input={"step": 1}, title="AssistantState")
    monologue.add_assistant_message("Second message", assistant_state)

    # Confirm both are present
    messages = monologue.list_messages()
    assert len(messages) == 2
    last_message_id = messages[-1].id
    assert any(m.id == last_message_id for m in messages)

    # Act: pop the last message
    monologue.pop()

    # Assert: last message gone
    messages_after = monologue.list_messages()
    assert len(messages_after) == 1
    assert not any(m.id == last_message_id for m in messages_after)


def test_create_monologue_from_messages(request):

    # Arrange: a clean temp dir for this test
    tmp_dir = make_local_tmp(request)
    file_path = os.path.join(tmp_dir, "monologue.json")

    # Prepare a batch of 4 MessagePydantic objects with mixed user/assistant content
    messages = [
        MessagePydantic(**{
            'id': 'b1e5f98c-f6eb-47de-a6e2-387510d970f9',
            'header': {
                'sender': 'User',
                'recipient': 'Assistant',
                'timestamp': 1751308157.270983,
                'order': 0
            },
            'body': {
                'type': 'monologue',
                'state_name': 'CHAT',
                'step_no': 1,
                'content_type': 'text',
                'role': 'user',
                'content': 'Tell me a one paragragh story.',
            }
        }),
        MessagePydantic(**{
            'id': 'abbc7961-45dc-4973-aaf4-a6224ed35d37',
            'header': {
                'sender': 'Assistant',
                'recipient': 'User',
                'timestamp': 1751308167.3488164,
                'order': 1
            },
            'body': {
                'type': 'monologue',
                'state_name': 'CHAT',
                'step_no': 2,
                'content_type': 'text',
                'role': 'assistant',
                'content': [
                    {
                        'citations': None,
                        'text': 'I will search for a story online.',
                    },
                    {
                        'id': 'toolu_013epBSqLbV61tRCLd1aMT86',
                        'input': {'url': 'https://www.story.com'},
                        'name': 'scrape',
                        'type': 'tool_use'
                    }
                ]
            }
        }),
        MessagePydantic(**{
            'id': 'f22fd2b5-dc7e-4389-a079-703d28dd365a',
            'header': {
                'sender': 'User',
                'recipient': 'Assistant',
                'timestamp': 1751308180.9790845,
                'order': 2
            },
            'body': {
                'type': 'monologue',
                'state_name': 'TOOL_USE',
                'step_no': 3,
                'content_type': 'text',
                'role': 'user',
                'content': [
                    {
                        'type': 'tool_result',
                        'tool_use_id': 'toolu_013epBSqLbV61tRCLd1aMT86',
                        'content': 'Once upon a time, in a small village, there lived a kind...'
                    }
                ]
            }
        }),
        MessagePydantic(**{
            'id': '787ef108-16f9-489a-a6f1-141aff5f0e49',
            'header': {
                'sender': 'Assistant',
                'recipient': 'User',
                'timestamp': 1751308181.9790845,
                'order': 3
            },
            'body': {
                'type': 'monologue',
                'state_name': 'TOOL_USE',
                'step_no': 4,
                'content_type': 'text',
                'role': 'assistant',
                'content': 'Once upon a time, in a small village, there lived a kind...'
            }
        }),
    ]

    # Act: create a FileMonologue from those messages
    monologue = FileMonologue(
        agent_name="TestBot",
        file_path=file_path,
        messages=messages
    )
    loaded = monologue.list_messages()

    # Assert: all 4 round-trip correctly
    assert len(loaded) == 4

    # 1st message
    assert loaded[0].body.role == "user"
    assert loaded[0].body.content == "Tell me a one paragragh story."

    # 2nd message
    assert loaded[1].body.role == "assistant"
    # first content item text
    assert loaded[1].body.content[0]["text"] == "I will search for a story online."
    # second content item type
    assert loaded[1].body.content[1]["type"] == "tool_use"

    # 3rd message
    assert loaded[2].body.role == "user"
    assert loaded[2].body.content[0]["type"] == "tool_result"

    # 4th message
    assert loaded[3].body.role == "assistant"
    assert loaded[3].body.content == "Once upon a time, in a small village, there lived a kind..."
