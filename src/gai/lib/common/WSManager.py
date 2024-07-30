import asyncio
from fastapi import WebSocket
import json

class WSManager:
    def __init__(self):
        self.active_connections = {}

    async def connect(self, websocket: WebSocket, agent_id: str):
        await websocket.accept()
        self.active_connections[agent_id]=websocket

    async def disconnect(self, agent_id: str):
        if agent_id in self.active_connections:
            websocket = self.active_connections[agent_id]
            await websocket.close()
            del self.active_connections[agent_id]

    async def broadcast_message(self, message: str):
        await asyncio.sleep(0)
        for key, connection in self.active_connections.items():
            if key != "0":
                await connection.send_text(json.dumps({"message":message}))

    async def broadcast_progress(self, i, max):
        await asyncio.sleep(0)
        progress = int(i*100/max)
        for key, connection in self.active_connections.items():
            if key != "0":
                await connection.send_text(json.dumps({"progress":progress}))

ws_manager = WSManager()