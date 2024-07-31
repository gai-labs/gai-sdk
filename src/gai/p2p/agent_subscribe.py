import json
from gai.lib.common.http_utils import http_post
from gai.lib.common.generators_utils import chat_string_to_list
from gai.lib.common.errors import ApiException
from gai.lib.common.logging import getLogger
logger = getLogger(__name__)
from gai.lib.common.utils import get_gai_config

from openai.types.chat.chat_completion import ChatCompletion
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk

from nats.aio.client import Client as NATS
import asyncio

"""
This function is designed for subscribing(listening) for a message from the user.
The agent will publish its response upon receiving a user message.
The dialogue will not expect any system message. 
Any system message will be sent internally to monologue.
"""
async def agent_subscribe(message):
    # Connect to NATS server running on Docker
    nc = NATS()
    await nc.connect("nats://localhost:4222")

    async def message_handler(msg):
        subject = msg.subject
        data = msg.data.decode()
        print(f"Received a message on '{subject}': {data}")

    # Subscribe to subject 'test'
    await nc.subscribe("dialogue", cb=message_handler)

    # Keep the connection alive
    await asyncio.Future()  # Run forever
