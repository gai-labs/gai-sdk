import io
import uuid
import json
from gai.messages import MessagePydantic, MessageHeaderPydantic
from gai.dialogue import FileDialogueBus
import time
from unittest.mock import mock_open, patch, call

# add a decorator here to use mock_open
@patch("os.makedirs")
@patch("os.path.exists", return_value=True)
@patch("builtins.open", new_callable=mock_open, read_data='{"last_message_order":1,"messages":[{"id": "123", "type": "test", "header": {"sender": "sender", "recipient": "recipient", "timestamp": 1234567890}, "body": "This is a test message"}]}\n')
def test_message_read(mock_file_open, mock_exists, mock_makedirs):
    temp_memory = FileDialogueBus.DialogueFileStorage(logger_name="TestNode", app_dir="/fake", dialogue_id="test")    
    m = temp_memory.get_message(id="123")
    assert m.body == "This is a test message"

@patch("os.makedirs")
@patch("os.path.exists", return_value=True)
@patch("builtins.open", new_callable=mock_open, read_data=json.dumps({"last_message_order": 0, "messages": []}))
def test_message_insert(mock_file_open, mock_exists, mock_makedirs):
    dialogue = FileDialogueBus.DialogueFileStorage(logger_name="TestNode", app_dir="/fake", dialogue_id="test")

    msg = MessagePydantic.from_dict({
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

    dialogue.insert_message(msg)
    expected_path = dialogue.get_dialogue_path()
    
    # Filter top-level open() calls only
    top_level_calls = [call_args for call_args in mock_file_open.call_args_list if call_args[0][0] == expected_path]

    assert top_level_calls == [
        call(expected_path, "r"),
        call(expected_path, "w"),
    ]

@patch("os.makedirs")
@patch("os.path.exists", return_value=True)
@patch("builtins.open")
def test_message_update(mock_file_open, mock_exists, mock_makedirs):
    # Original messages
    existing_messages = [
        {
            "id": "123",
            "type": "test",
            "header": {"sender": "sender", "recipient": "recipient", "timestamp": 1234567890},
            "body": "Original message"
        },
        {
            "id": "456",
            "type": "test",
            "header": {"sender": "sender", "recipient": "recipient", "timestamp": 1234567891},
            "body": "Another message"
        }
    ]
    
    # Full internal structure from file
    initial_data = {
        "last_message_order": 1,
        "messages": existing_messages
    }

    # Prepare mocked file handles
    mock_read_handle = mock_open(read_data=json.dumps(initial_data)).return_value
    mock_write_handle = mock_open().return_value
    mock_file_open.side_effect = [mock_read_handle, mock_write_handle]

    # Instantiate storage
    storage = FileDialogueBus.DialogueFileStorage(
        logger_name="TestNode", app_dir="/fake", dialogue_id="test"
    )

    # Updated message (only first message should be changed)
    updated = MessagePydantic(
        id="123",
        type="test",
        header=MessageHeaderPydantic(sender="sender", recipient="recipient", timestamp=time.time()),
        body="Updated message"
    )

    # Perform update
    storage.update_message(updated)

    # --- Now verify what was written ---
    # Get the actual data written
    written_json = "".join(call_arg.args[0] for call_arg in mock_write_handle.write.call_args_list)
    written_obj = json.loads(written_json)

    # Verify both messages exist, and first one was replaced
    assert written_obj["messages"][0]["id"] == updated.id
    assert written_obj["messages"][0]["body"] == "Updated message"
    assert written_obj["messages"][1] == existing_messages[1]
    assert written_obj["last_message_order"] == 1  # unchanged



@patch("os.makedirs")
@patch("os.path.exists", return_value=True)
@patch("builtins.open", new_callable=mock_open)
def test_message_delete(mock_file_open, mock_exists, mock_makedirs):
    # Prepare message list
    messages = [
        {"id": "123", "type": "test", "header": {"sender": "sender", "recipient": "recipient", "timestamp": 1234567890}, "body": "Original message"},
        {"id": "456", "type": "test", "header": {"sender": "sender", "recipient": "recipient", "timestamp": 1234567891}, "body": "Another message"}
    ]
    lines = [json.dumps(msg) + "\n" for msg in messages]

    # Mock the read handle to return these lines via readlines
    mock_read_handle = mock_open(read_data="").return_value
    mock_read_handle.readlines.return_value = lines

    # Mock the write handle
    mock_write_handle = mock_open().return_value

    # Set side_effect to alternate between read and write
    mock_file_open.side_effect = [mock_read_handle, mock_write_handle]

    dialogue = FileDialogueBus.DialogueFileStorage(logger_name="TestNode", app_dir="/fake", dialogue_id="test")
    dialogue.delete_message("123")
    
    # Check open was called for read and then for write
    assert mock_file_open.call_count == 2
    assert mock_file_open.call_args_list[0][0][1] == "r"
    assert mock_file_open.call_args_list[1][0][1] == "w"

    # Expect only the second message to be written back
    expected_write = json.dumps(messages[1]) + "\n"
    mock_write_handle.write.assert_called_once_with(expected_write)