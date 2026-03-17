from fastapi import WebSocket

class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: str):
        # iterate over a copy so we can remove safely
        for connection in list(self.active_connections):
            try:
                await connection.send_text(message)
            except Exception as e:
                # connection is dead or closing -> drop it
                print(f"Removing dead websocket: {e}")
                self.disconnect(connection)