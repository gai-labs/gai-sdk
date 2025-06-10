import json
import pytest
import asyncio
from unittest.mock import AsyncMock

from gai.dialogue import DialogueBus
from gai.messages import MessagePydantic

@pytest.mark.asyncio
async def test_add_and_list_messages():
    dialogue = DialogueBus()
    await dialogue.start()
    
    msg = MessagePydantic.from_dict({
        "type": "send",
        "header": {"sender": "User"},
        "body": {"content": "Hello"}
    })
    await dialogue.publish(pydantic=msg)
    
    #  ✅ Assert that only one message is present
    
    assert len(dialogue.list_messages()) == 1
    
    #  ✅ Assert that message is correct
    
    assert dialogue.list_messages()[0].body.content == "Hello"

@pytest.mark.asyncio
async def test_subscribe_and_publish():
    dialogue = DialogueBus()    
    await dialogue.start()
    
    handler = AsyncMock()
    await dialogue.subscribe("send", {"Alice":handler})
    
    msg = MessagePydantic.from_dict({
        "type": "send",
        "header": {"sender": "User"},
        "body": {"content": "Hello"}
    })
    await dialogue.publish(pydantic=msg)
    
    # Allow async handlers time to complete before checking
    await asyncio.sleep(0.2)
    
    #  ✅ Assert that handler was called once
    
    handler.assert_called_once()

    await dialogue.stop()
    

@pytest.mark.asyncio
async def test_reset_and_delete_message():
    dialogue = DialogueBus()
    await dialogue.start()
    
    msg = MessagePydantic.from_dict({
        "type": "send",
        "header": {"sender": "User"},
        "body": {"content": "Test"}
    })
    await dialogue.publish(pydantic=msg)
    mid = msg.id

    dialogue.delete_message(mid)
    
    #  ✅ Assert that one messages is deleted
    
    assert len(dialogue.list_messages()) == 0

    await dialogue.publish(pydantic=msg)
    
    # Allow async handlers to run
    
    dialogue.reset()

    #  ✅ Assert that all messages are deleted
    
    assert len(dialogue.list_messages()) == 0
    
    await dialogue.stop()


@pytest.mark.asyncio
async def test_publish_with_recap():
    dialogue = DialogueBus()
    await dialogue.start()
    
    handler = AsyncMock()
    dialogue.subscribe("send", {"Antonio":handler})
    
    await dialogue.publish(pydantic= MessagePydantic.from_dict({
        "type": "send",
        "header": {"sender": "User"},
        "body": {"content": "Tell me your name"}
    }))
    await dialogue.publish(pydantic= MessagePydantic.from_dict({
        "type": "reply",
        "header": {"sender": "Antonio"},
        "body": {"content": "I am Antonio"}
    }))
    await dialogue.publish(pydantic= MessagePydantic.from_dict({
        "type": "send",
        "header": {"sender": "User"},
        "body": {"content": "What is your age?"}
    }))
    await dialogue.publish(pydantic= MessagePydantic.from_dict({
        "type": "reply",
        "header": {"sender": "Antonio"},
        "body": {"content": "I am 25 years old"}
    }))    
    recap = dialogue.extract_recap()
    print(recap)
    recap = json.loads(recap)
    
    #  ✅ Assert that messages are in the correct order and correct length in recap
    
    assert recap[0]["User"] == "Tell me your name"
    assert recap[1]["Antonio"] == "I am Antonio"
    assert recap[2]["User"] == "What is your age?"
    assert recap[3]["Antonio"] == "I am 25 years old"

    await dialogue.stop()