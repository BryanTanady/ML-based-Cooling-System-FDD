"""Prediction logging and alert forwarding helpers."""

from __future__ import annotations

import logging
import time
from collections import Counter

import numpy as np

from fdd_system.ML.common.config import OperatingCondition
from fdd_system.broker.alerts import AlertSender


def record_predictions(
    preds: np.ndarray,
    confs: np.ndarray,
    prediction_counts: Counter[int],
    alert_sender: AlertSender,
    logger: logging.Logger,
) -> None:
    """Log predictions, update counters, and forward non-normal events."""
    preds_arr = np.asarray(preds).ravel()
    conf_arr = np.asarray(confs, dtype=float).ravel()

    conditions = []
    for pred in preds_arr:
        try:
            conditions.append(OperatingCondition(int(pred)).name)
        except ValueError:
            conditions.append(f"Unknown({int(pred)})")

    conf_fmt = [f"{float(c):.3f}" if np.isfinite(c) else "nan" for c in conf_arr]
    logger.info("Prediction: %s, %s | conf=%s", preds_arr.tolist(), conditions, conf_fmt)

    now_ts = time.time()
    for idx, pred in enumerate(preds_arr):
        pred_id = int(pred)
        prediction_counts[pred_id] += 1

        confidence: float | None = None
        if idx < conf_arr.size and np.isfinite(conf_arr[idx]):
            confidence = float(conf_arr[idx])

        if pred_id != OperatingCondition.NORMAL.value:
            alert_sender.send_prediction(pred_id, confidence, ts=now_ts)
