"""Windowing utilities for accelerometer sample streams."""

from __future__ import annotations

from collections import deque
from typing import Deque, Optional, Tuple

import numpy as np

from fdd_system.ML.common.config.data import RawAccWindow
from fdd_system.ML.common.config.system import SensorConfig


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
