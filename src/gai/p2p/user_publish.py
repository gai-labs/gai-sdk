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
This function is designed for publishing a single user's message to all subscribed agents.
How the agent responds (or timeout means not responding) is subjected to the agent's design.
"""
async def user_publish(message):
    # Connect to NATS server running on Docker
    nc = NATS()
    await nc.connect("nats://localhost:4222")

    # Send a message to the 'test' subject
    await nc.publish("dialogue", b'Tell me a one paragraph story')

    # Close the connection
    await nc.close()    

