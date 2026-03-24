"""I/O utilities for broker serial ingest, parsing, windowing, and alerts."""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from collections import deque
from typing import Deque, Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
import serial
from serial.tools import list_ports

from fdd_system.ML.common.config import OperatingCondition, RawAccWindow, SensorConfig

_XYZ_LOG_PATTERN = re.compile(
    r"x\s+([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s+"
    r"y\s+([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s+"
    r"z\s+([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
)

_S_LINE_PATTERN = re.compile(
    r"S\s+[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?\s+"
    r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s+"
    r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s+"
    r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
)


def parse_sample(line: str) -> Optional[Tuple[float, float, float]]:
    """Parse a serial text line into accelerometer xyz values."""
    if "Skipping unparsable line:" in line:
        line = line.split("Skipping unparsable line:", 1)[1].strip()

    parts = line.strip().split(",")
    if len(parts) < 3:
        match = _XYZ_LOG_PATTERN.search(line)
        if match:
            try:
                return float(match.group(1)), float(match.group(2)), float(match.group(3))
            except ValueError:
                return None

        match_s = _S_LINE_PATTERN.search(line)
        if match_s:
            try:
                return float(match_s.group(1)), float(match_s.group(2)), float(match_s.group(3))
            except ValueError:
                return None

        return None

    try:
        return float(parts[0]), float(parts[1]), float(parts[2])
    except ValueError:
        return None


class WindowBuilder:
    """Accumulates accelerometer samples into RawAccWindow objects."""

    def __init__(self, window_size: int, *, sampling_rate_hz: float | None = None):
        self.window_size = window_size
        self.sampling_rate_hz = sampling_rate_hz
        self.samples: Deque[Tuple[float, float, float]] = deque()

    def add(self, ax: float, ay: float, az: float) -> Optional[RawAccWindow]:
        """Add one sample and emit a window once enough samples are buffered."""
        self.samples.append((ax, ay, az))
        if len(self.samples) < self.window_size:
            return None

        ax_arr, ay_arr, az_arr = (np.array(vals) for vals in zip(*list(self.samples)[: self.window_size]))

        for _ in range(SensorConfig.STRIDE):
            self.samples.popleft()

        return RawAccWindow(acc_x=ax_arr, acc_y=ay_arr, acc_z=az_arr, sampling_rate_hz=self.sampling_rate_hz)


class SerialReader:
    """Continuously reads bytes from a serial port and emits newline-terminated lines."""

    def __init__(
        self,
        port: str = "COM3",
        baudrate: int = 9600,
        timeout: float = 1.0,
        buffer=None,
    ):
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.buffer = buffer

        self._stop_flag = threading.Event()
        self._thread = None
        self._partial = b""

        self.ser = serial.Serial(self.port, self.baudrate, timeout=self.timeout)

        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

    def _read_loop(self):
        """Poll the serial port and accumulate bytes until stopped."""
        while not self._stop_flag.is_set():
            try:
                chunk = self.ser.read(self.ser.in_waiting or 1)
                if chunk:
                    self._process_chunk(chunk)
                else:
                    print("No data received.")
            except Exception:
                print("Error reading from serial port.")

    def _process_chunk(self, chunk: bytes):
        """Split incoming bytes on newline and append decoded lines to buffer."""
        self._partial += chunk

        while b"\n" in self._partial:
            line, self._partial = self._partial.split(b"\n", 1)
            line = line.rstrip(b"\r")
            text = line.decode("utf-8", errors="ignore")
            self.buffer.append(text)

    def stop(self):
        """Stop the reader thread and close the serial port."""
        self._stop_flag.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)

        if self.ser and self.ser.is_open:
            self.ser.close()

    def __del__(self):
        self.stop()

    @staticmethod
    def list_devices():
        """Return a list of available serial ports with descriptions."""
        devices = []
        baudrates = list_ports.comports()
        for b in baudrates:
            devices.append(f"{b.device} - {b.description}")
        return devices


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
