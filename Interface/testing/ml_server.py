# sender.py
import asyncio
from datetime import datetime
from typing import Optional

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import random

TARGET_URL = "http://127.0.0.1:8000/api/alert"   #backedn endpoint url
INTERVAL_SEC = 10  # send every 30s (adjust as needed)

app = FastAPI()

# Alert json data (can be changed later)
class Alert(BaseModel):
    asset_id: str
    severity: str
    message: str
    ts: Optional[float] = None

severity_list = ["critical", "major", "minor", "info"]
message_list = ["Fan Blocked", "Fan Blade Issue", "Electrical Fault", "Unknown"]
asset_list = [
    "FAN-01", "FAN-02", "FAN-03", "FAN-04", "FAN-05",
    "FAN-21", "FAN-22", "FAN-23", "FAN-24", "FAN-25",
]

stop_event: asyncio.Event = asyncio.Event()
loop_task: Optional[asyncio.Task] = None

async def post_alert(alert: Alert):
    async with httpx.AsyncClient(timeout=5.0) as client:
        r = await client.post(TARGET_URL, json=alert.model_dump())
        r.raise_for_status()
        return r.json()

async def schedule_sender():
    while not stop_event.is_set():
        alert = Alert(
            asset_id=random.choice(asset_list),
            severity=random.choice(severity_list),
            message=random.choice(message_list),
            ts=datetime.now().timestamp(),
        )
        try:
            resp = await post_alert(alert)
            print(f"[{datetime.now().isoformat()}] sent: {resp}")
        except Exception as e:
            print(f"[{datetime.now().isoformat()}] failed: {e}")
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=INTERVAL_SEC)
        except asyncio.TimeoutError:
            pass

@app.on_event("startup")
async def startup():
    global loop_task
    loop_task = asyncio.create_task(schedule_sender())
    print(f"Sender started → target={TARGET_URL} interval={INTERVAL_SEC}s")

@app.on_event("shutdown")
async def shutdown():
    stop_event.set()
    if loop_task:
        await loop_task

@app.get("/")
def health():
    return {"status": "ok", "target": TARGET_URL, "interval_seconds": INTERVAL_SEC}

@app.post("/send-now")
async def send_now(alert: Alert | None = None):
    """Trigger an immediate alert send. Optional JSON overrides."""
    alert = alert or Alert(
        asset_id="FAN-"+str(random.randint(1, 100)),
        severity=random.choice(severity_list),
        message=random.choice(message_list),
        ts=datetime.now().timestamp(),
    )
    try:
        resp = await post_alert(alert)
        return {"status": "sent", "response": resp}
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))
