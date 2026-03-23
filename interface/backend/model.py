from pydantic import BaseModel

"""
Alert Data Structure
Example:
asset_id = "FAN-01"
condition_id = 1
condition_name = "BLOCKED_AIRFLOW"
message = "Blocked Airflow"
confidence = 0.95
ts = 1715769600.0
"""
class Alert(BaseModel):
    asset_id: str
    condition_id: int | None = None
    condition_name: str | None = None
    message: str
    confidence: float | None = None
    ts: float | None = None



__all__ = ["Alert"]