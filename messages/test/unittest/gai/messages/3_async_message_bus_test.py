import pytest
import asyncio
from unittest.mock import AsyncMock
from gai.messages import AsyncMessageBus

@pytest.mark.asyncio
async def test_messagebus_match():
    bus = AsyncMessageBus()
    assert bus._matches('foo.*', 'foo.bar') == True
    assert bus._matches('foo.*', 'foo.bar.baz') == False
    assert bus._matches('foo.>', 'foo.bar') == True
    assert bus._matches('foo.>', 'foo.bar.baz') == True
    assert bus._matches('foo.>', 'foo') == False
    assert bus._matches('>', 'foo') == True
    assert bus._matches('>', 'foo.bar') == True
    assert bus._matches('foo.bar', 'foo.bar') == True
    assert bus._matches('foo.bar', 'foo') == False

@pytest.mark.asyncio
async def test_messagebus_run_single_subscriber_exact_match():
    bus = AsyncMessageBus()
    await bus.start()
    
    # Add subscriber
    
    received = []
    async def handle_send(msg):
        received.append(msg)
    await bus.subscribe("send", {"sara": handle_send})
    
    # Send messages
    
    await bus.publish({"type": "send", "body": {"content": "match"}})
    await asyncio.sleep(0.2)
    await bus.stop()
    
    # ✅ Assert that the correct message was received

    assert received[0].body.content == "match"

@pytest.mark.asyncio
async def test_messagebus_run_single_subscriber_single_token_match():
    bus = AsyncMessageBus()
    await bus.start()
    
    # Add Subscriber

    received = []
    async def handler(msg):
        received.append(msg)
    await bus.subscribe("alpha.*.charlie", {"Sara":handler})
    
    # Send messages
    
    await bus.publish({"type": "alpha.beta"})
    await bus.publish({"type": "alpha.beta.charlie"})
    await bus.publish({"type": "alpha.beta.charlie.delta"})
    await asyncio.sleep(0.2)
    await bus.stop()
    
    # ✅ Assert that the correct message was received
    
    assert len(received) == 1
    assert received[0].type == "alpha.beta.charlie"

@pytest.mark.asyncio
async def test_messagebus_run_single_subscriber_multiple_tokens_match():
    bus = AsyncMessageBus()
    await bus.start()

    # Add Subscriber

    received = []
    async def handler(msg):
        received.append(msg)
    await bus.subscribe("system.rollcall.>", {"Sara":handler})

    # Send messages
    
    await bus.publish({"type": "system.rollcall"})
    await bus.publish({"type": "system.rollcall.ping"})
    await bus.publish({"type": "system.reset"})
    await asyncio.sleep(0.2)
    await bus.stop()
    
    # ✅ Assert that only one correct message was received

    assert len(received) == 1    
    assert received[0].type == "system.rollcall.ping"

@pytest.mark.asyncio
async def test_messagebus_run_multiple_subscribers_receive_same_message():
    bus = AsyncMessageBus()
    await bus.start()
    
    # Add Subscriber 1
    
    received_1 = []
    async def handler1(msg):
        received_1.append(msg)
    await bus.subscribe("x.y", {"Sara":handler1})

    # Add Subscriber 2

    received_2 = []
    async def handler2(msg):
        received_2.append(msg)
    await bus.subscribe("x.y", {"Diana":handler2})
    
    # Simulate that the bus is started
    
    await bus.publish({"type": "x.y", "body": "shared"})
    await asyncio.sleep(0.2)
    await bus.stop()

    # ✅ Assert that both subscribers received the same message

    assert received_1[0].body == "shared"
    assert received_2[0].body == "shared"

@pytest.mark.asyncio
async def test_subscribe_cannot_add_duplicate_async_handlers():
    bus = AsyncMessageBus()
    await bus.start()

    # Simulate bus started with one subscriber
    handler_async = AsyncMock()
    
    # First subscription
    await bus.subscribe("event", {"Sara":handler_async})

    # Duplicate subscription (should replace the previous one)
    await bus.subscribe("event", {"Sara":handler_async})
    
    # Now publish a message
    await bus.publish({"type": "event", "body": {"content": "hello"}})
    await asyncio.sleep(0.2)
    await bus.stop()
    
    # ✅ Bus is subscribed twice for the same subject by the same name but it should only be called once
    assert handler_async.call_count == 1, f"Expected 1 call, got {handler_async.call_count}"

@pytest.mark.asyncio
async def test_subscribe_callback_with_states():
    """
    Problem:
    - Create a single callback method that contains its own state.
    - The same callback method should be usable for multiple subscribers.
    """

    bus = AsyncMessageBus()
    await bus.start()
    
    send_called = []
    reply_called = []

    # Create a dummy bus client
        
    class Dummy:

        '''
        This class will send a 'reply' message whenever a 'send' message is received.
        '''
    
        def __init__(self, name, bus):
            self.name=name
            self.bus = bus

        async def send_handler(self, msg):
            # Received 'send' message
            send_called.append(self.name)
            # Send 'reply' message
            await self.bus.publish({"type": "reply", "body": {"content": "reply"}})
        
        def reply_handler(self, msg):
            # Received 'reply' message
            reply_called.append(self.name)

    sara = Dummy("Sara",bus)
    diana = Dummy("Diana",bus)
    
    # Subscribe to the bus
    
    await bus.subscribe("reply", {sara.name: sara.reply_handler})
    await bus.subscribe("send", {diana.name: diana.send_handler})

    # Send messages

    await bus.publish({"type": "send","body": {"content":"hello"}})
    await asyncio.sleep(0.2)
    await bus.stop()

    # ✅ Assert that both handlers were called
    
    assert send_called == ["Diana"], f"Expected 'Diana', got {send_called}"
    assert reply_called == ["Sara"], f"Expected 'Sara', got {reply_called}"
    
