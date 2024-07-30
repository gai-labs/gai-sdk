from gai.lib.common.logging import getLogger
import json
logger = getLogger(__name__)

'''
A wrapper class for the caller websocket from the API endpoint.
This class is used to public JSON messages to caller.
'''

class StatusPublisher:

    def __init__(self, websocket):
        self.websocket=websocket

    async def send_message(self,message):
        try:
            await self.websocket.send(json.dumps({'message':message}))
        except Exception as e:
            logger.error(f"Error sending message: {e}")

    # update_progress is the same as send_message,
    # but it returns an integer between 0 to 100
    async def update_progress(self, i, max):
        try:
            status = int(i*100/max)
            await self.websocket.send(json.dumps({'progress':status}))
        except Exception as e:
            logger.error(f"Error sending progress: {e}")

    async def update_stop(self):
        try:
            await self.websocket.send("<STOP>")
        except Exception as e:
            logger.error(f"Error sending stop: {e}")


