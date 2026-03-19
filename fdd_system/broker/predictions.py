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
    rejection_stage: np.ndarray | None = None,
    rejection_reason: np.ndarray | None = None,
) -> None:
    """Log predictions, update counters, and forward non-normal events."""
    preds_arr = np.asarray(preds).ravel()
    conf_arr = np.asarray(confs, dtype=float).ravel()
    stage_arr = np.asarray(rejection_stage, dtype=object).ravel() if rejection_stage is not None else None
    reason_arr = np.asarray(rejection_reason, dtype=object).ravel() if rejection_reason is not None else None

    conditions = []
    for pred in preds_arr:
        try:
            conditions.append(OperatingCondition(int(pred)).name)
        except ValueError:
            conditions.append(f"Unknown({int(pred)})")

    conf_fmt = [f"{float(c):.3f}" if np.isfinite(c) else "nan" for c in conf_arr]
    unknown_source_fmt = []
    for idx, pred in enumerate(preds_arr):
        if int(pred) != OperatingCondition.UNKNOWN.value:
            unknown_source_fmt.append("-")
            continue

        stage = ""
        if stage_arr is not None and idx < stage_arr.size and stage_arr[idx] is not None:
            stage = str(stage_arr[idx]).strip()
        reason = ""
        if reason_arr is not None and idx < reason_arr.size and reason_arr[idx] is not None:
            reason = str(reason_arr[idx]).strip()

        if stage == "STAGE0" and reason:
            unknown_source_fmt.append(f"{stage}:{reason}")
        elif stage:
            unknown_source_fmt.append(stage)
        else:
            unknown_source_fmt.append("UNKNOWN")

    logger.info(
        "Prediction: %s, %s | conf=%s | unknown_source=%s",
        preds_arr.tolist(),
        conditions,
        conf_fmt,
        unknown_source_fmt,
    )

    now_ts = time.time()
    for idx, pred in enumerate(preds_arr):
        pred_id = int(pred)
        prediction_counts[pred_id] += 1

        confidence: float | None = None
        if idx < conf_arr.size and np.isfinite(conf_arr[idx]):
            confidence = float(conf_arr[idx])

        if pred_id != OperatingCondition.NORMAL.value:
            alert_sender.send_prediction(pred_id, confidence, ts=now_ts)
