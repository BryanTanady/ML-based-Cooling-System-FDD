"""Alert payload mapping and transport utilities."""

from __future__ import annotations

import json
import logging
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np

from fdd_system.ML.common.config import OperatingCondition

def _condition_to_message(condition: OperatingCondition | None) -> str:
    if condition is None:
        return "Unknown"
    return condition.name.replace("_", " ").title()


class AlertSender:
    """Build and publish alert payloads to the diagnostics backend."""

    def __init__(self, api_url: str, asset_id: str, timeout_sec: float, logger: logging.Logger):
        self.api_url = api_url
        self.asset_id = asset_id
        self.timeout_sec = timeout_sec
        self.logger = logger

    def build_alert(
        self,
        pred_class_id: int,
        confidence: float | None,
        ts: float | None = None,
    ) -> dict[str, object] | None:
        """Map a prediction into the backend alert schema."""
        try:
            condition = OperatingCondition(int(pred_class_id))
        except ValueError:
            condition = None

        if condition == OperatingCondition.NORMAL:
            return None

        message = _condition_to_message(condition)
        condition_id: int | None = None
        condition_name: str | None = None
        if condition is not None:
            condition_id = int(condition.value)
            condition_name = condition.name

        confidence_value: float | None = None
        if confidence is not None and np.isfinite(float(confidence)):
            confidence_value = float(confidence)
            self.logger.debug("Alert candidate class=%s confidence=%.4f", pred_class_id, confidence_value)

        return {
            "asset_id": self.asset_id,
            "condition_id": condition_id,
            "condition_name": condition_name,
            "message": message,
            "confidence": confidence_value,
            "ts": float(ts if ts is not None else time.time()),
        }

    def send_alert(self, payload: dict[str, object]) -> bool:
        """POST an alert payload to the backend API."""
        body = json.dumps(payload).encode("utf-8")
        req = Request(
            self.api_url,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urlopen(req, timeout=self.timeout_sec) as response:
                status = getattr(response, "status", response.getcode())
                if 200 <= status < 300:
                    return True
                self.logger.warning("Alert API returned status=%s payload=%s", status, payload)
        except (HTTPError, URLError, TimeoutError, OSError, ValueError) as exc:
            self.logger.warning("Failed to post alert to %s: %s", self.api_url, exc)

        return False

    def send_prediction(
        self,
        pred_class_id: int,
        confidence: float | None,
        ts: float | None = None,
    ) -> bool:
        """Build and send alert for non-normal predictions."""
        payload = self.build_alert(pred_class_id, confidence, ts=ts)
        if payload is None:
            return False
        return self.send_alert(payload)
