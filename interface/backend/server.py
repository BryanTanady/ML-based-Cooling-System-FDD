from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json
import asyncio
import uuid
from datetime import datetime
from database import Database
from model import Alert
from stateManager import StateManager
import os
from dotenv import load_dotenv
from fastapi import HTTPException
from websocket import ConnectionManager
from datetime import timezone
load_dotenv()


app = FastAPI(title="Capstone API")

FRONTEND_URL = os.getenv("FRONTEND_URL")
ML_URL = os.getenv("FAULT_DETECTION_URL")


def _normalize_origin(origin: str | None) -> str | None:
    if not origin:
        return None
    # Browser Origin headers do not include a trailing slash.
    return origin.rstrip("/")


_allowed_origins = sorted({
    o for o in [
        _normalize_origin(FRONTEND_URL),
        _normalize_origin(ML_URL),
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:8080",
        "http://127.0.0.1:8080",
    ] if o
})

# WebSocket connection manager (module-global singleton)
manager = ConnectionManager()

app.add_middleware(
CORSMiddleware,
allow_origins=_allowed_origins,
allow_credentials=True,
allow_methods=["*"],
allow_headers=["*"],
)



# Server Holds Recent Alerts (cache) -> later may move to DB
RECENT: list[dict] = []
MAX_RECENT = 200
RECEIVED = False

# on startup, initialize the database and state manager
@app.on_event("startup")
def startup():
    global db, state_manager
    db = Database()
    state_manager = StateManager()


# --------- API ---------
@app.get("/")
def health():
    return {"ok": True, "time": datetime.now().isoformat()}

@app.post("/api/alert")
async def receive_alert(alert: Alert):
    global RECEIVED, RECENT
    try:    
        payload = alert

        # DUMPING RAW DATA TO DB
        db.insert_alert(payload)

        # Broadcast raw alert to all connected clients (for graph / raw event stream)
        raw_msg = {
            "type": "raw_alert",
            "asset_id": payload.asset_id,
            "condition_id": getattr(payload, "condition_id", None),
            "condition_name": getattr(payload, "condition_name", None),
            "message": payload.message,
            "confidence": getattr(payload, "confidence", None),
            "ts": getattr(payload, "ts", None),
        }
        await manager.broadcast(json.dumps(raw_msg))

        # Update fault state: on start → insert DB + broadcast; on end → update end_ts by id
        def on_fault_period_end(fault_id: str, end_ts: float):
            db.update_fault_period_end(fault_id, end_ts)

        fault_started = state_manager.process_alert(payload, on_fault_period_end)
        print(f"==========FAULT STARTED==========: {fault_started}")
        if fault_started:
            db.insert_fault_period_start(
                fault_started["id"],
                fault_started["asset_id"],
                fault_started["fault_type"],
                fault_started["start_ts"],
            )
            message = {
                "type": "fault_period",
                "id": fault_started["id"],
                "asset_id": fault_started["asset_id"],
                "fault_type": fault_started["fault_type"],
                "start_ts": fault_started["start_ts"].isoformat(),
            }
            await manager.broadcast(json.dumps(message))
            print(f"==========FAULT STARTED MESSAGE BROADCASTED==========: {message}")

        # Keep recent alerts for list endpoint (no broadcast on every alert)
        # RECENT.append(payload)
        # if len(RECENT) > MAX_RECENT:
        #     RECENT[:] = RECENT[-MAX_RECENT:]
        print("================================================ALERT INSERTED INTO DB================================================\n")
        print(f"Alert inserted and broadcasted successfully: {payload}")
        print("================================================================================================================")
        return {"status": "ok", "received": payload}
    except Exception as e:
        print(f"================================================ERROR Loading to Database ================================================\n")
        print(f"Error loading alert to database: {e}")
        print("================================================================================================================")
        raise HTTPException(
        status_code=500,
        detail=str(e)
        )

@app.get("/api/alerts")
def list_alerts():
    # newest first for convenience
    return RECENT

@app.get("/api/fault_state")
def get_fault_state():
    """Current fault state (None if normal)."""
    return state_manager.get_state() or {"state": "normal"}

@app.get("/api/db/fault_history")
def get_fault_history():
    """
    Fault history used by the frontend Fault History / Export pages.

    Returns fault *periods* (not raw alerts) from the DB_FAULT_PERIODS collection.
    """
    coll = db.get_database().get_collection(os.getenv("DB_FAULT_PERIODS"))
    results = list(coll.find().sort("start_ts", -1))

    def to_iso(v):
        if v is None:
            return None
        if isinstance(v, datetime):
            return v.replace(tzinfo=timezone.utc).isoformat().replace("+00:00", "Z")
        return v

    for doc in results:
        if "_id" in doc:
            doc["_id"] = str(doc["_id"])
        doc["asset_id"] = str(doc.get("asset_id"))
        doc["fault_type"] = str(doc.get("fault_type"))
        doc["start_ts"] = to_iso(doc.get("start_ts"))
        doc["end_ts"] = to_iso(doc.get("end_ts"))
        doc["acknowledged_at"] = to_iso(doc.get("acknowledged_at"))

    return results


@app.get("/api/db/raw_alerts")
def get_raw_alerts():
    """
    Raw alert history (every alert message), including confidence score and original fields.
    Pulled from the DB_RAW_DATA collection (Fault_History).
    """
    coll = db.get_database().get_collection(os.getenv("DB_RAW_DATA"))
    results = list(coll.find().sort("timestamp", -1))

    def to_iso(v):
        if v is None:
            return None
        if isinstance(v, datetime):
            return v.replace(tzinfo=timezone.utc).isoformat().replace("+00:00", "Z")
        return v

    for doc in results:
        if "_id" in doc:
            doc["_id"] = str(doc["_id"])
        doc["timestamp"] = to_iso(doc.get("timestamp"))

    return results


class AcknowledgeRequest(BaseModel):
    id: str
    acknowledged_at: str | None = None  # ISO string from frontend


@app.post("/api/fault_periods/ack")
def acknowledge_fault_period(req: AcknowledgeRequest):
    ack_dt = None
    if req.acknowledged_at:
        iso = req.acknowledged_at.replace("Z", "+00:00")
        ack_dt = datetime.fromisoformat(iso)
        if ack_dt.tzinfo is not None:
            ack_dt = ack_dt.astimezone(timezone.utc).replace(tzinfo=None)

    ok = db.acknowledge_fault_period(req.id, ack_dt)
    if not ok:
        raise HTTPException(status_code=404, detail=f"Fault period not found: {req.id}")
    return {"ok": True, "id": req.id, "acknowledged_at": (ack_dt or datetime.utcnow()).isoformat()}



# --------- WEB SOCEKTS ---------

@app.websocket("/ws/alerts")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while websocket.application_state == WebSocketState.CONNECTED:
            global RECEIVED
            if RECEIVED and RECENT:
                last = RECENT[-1]
                message = {
                    **last,
                    "count": len(RECENT)
                }
                await manager.broadcast(json.dumps(message))
                print("Frontend Received:", datetime.now(), message)
                RECEIVED = False
            await asyncio.sleep(0.2)
    finally:
        # finally ensures cleanup even on RuntimeError or other exceptions
        manager.disconnect(websocket)
        print("WebSocket disconnected")
