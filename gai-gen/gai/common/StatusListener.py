import websockets
import os
from gai.common.utils import get_gen_config
from gai.common.logging import getLogger
logger = getLogger(__name__)
logger.setLevel("DEBUG")

'''
This is class is used by the client-side websocket to receive status updates from the server.
'''

class StatusListener:

    def __init__(self, ws_url,agent_id):
        self.ws_url = os.path.join(ws_url,agent_id)
        self.agent_id = agent_id

    async def listen(self, callback):
        raise Exception("Implemented in gai-lib")
        # async with websockets.connect(self.ws_url) as websocket:
        #     logger.info(f"Connected to {self.ws_url}")
        #     while True:
        #         try:
        #             message = await websocket.recv()
        #             await callback(message)
        #         except websockets.ConnectionClosedError:
        #             logger.warn("StatusListener.listen: Server disconnected.")
        #             break
        #         except Exception as e:
        #             logger.error(f"Error: {e}")
        #             raise e
    
        