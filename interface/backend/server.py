from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json
import asyncio
from datetime import datetime
app = FastAPI(title="Capstone API")

app.add_middleware(
CORSMiddleware,
allow_origins=[
"http://localhost:5173", # Vite dev
"http://127.0.0.1:8080", # ML dev
],
allow_credentials=True,
allow_methods=["*"],
allow_headers=["*"],
)

# Alert Data Strcture (TBD: adjust fields as needed)
class Alert(BaseModel):
    asset_id: str
    severity: str
    message: str
    ts: float | None = None



# Server Holds Recent Alerts (cache) -> later may move to DB
RECENT: list[dict] = []
MAX_RECENT = 200
RECEIVED = False



# --------- API ---------
@app.get("/")
def health():
    return {"ok": True, "time": datetime.now().isoformat()}

@app.post("/api/alert")
async def receive_alert(alert: Alert):
    global RECEIVED, RECENT
    payload = alert.model_dump()

    # append to recent alerts
    RECENT.append(payload)
    if len(RECENT) > MAX_RECENT:
        RECENT = RECENT[-MAX_RECENT:]  # Keep only the most recent MAX_RECENT items
    # mark that we have a new alert to push over websockets
    RECEIVED = True

    print("Received:", datetime.now(), payload)
    return {"status": "ok", "received": payload}

@app.get("/api/alerts")
def list_alerts():
    # newest first for convenience
    return RECENT

# --------- WEB SOCEKTS ---------
#websocket manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

manager = ConnectionManager()

@app.websocket("/ws/alerts")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            global RECEIVED
            if RECEIVED and RECENT:
                # send the most recent alert as JSON with count
                last = RECENT[-1]
                message = {
                    **last,
                    "count": len(RECENT)
                }
                await manager.broadcast(json.dumps(message))
                RECEIVED = False
            await asyncio.sleep(0.2)
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        print("WebSocket disconnected")


