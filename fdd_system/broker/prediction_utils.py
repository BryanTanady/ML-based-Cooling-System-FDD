"""Prediction, pipeline construction, and calibration helpers for broker runtime."""

from __future__ import annotations

import json
import logging
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

try:
    import joblib
except ImportError:  # pragma: no cover - exercised by runtime environment
    joblib = None

from fdd_system.ML.common.anomaly_detector import load_anomaly_detector
from fdd_system.ML.common.config import OperatingCondition
from fdd_system.ML.common.embedder import (
    MLEmbedder1,
    MLEmbedder2,
    Raw1DCNNEmbedder,
    Spectrogram2DEmbedder,
)
from fdd_system.ML.common.inferrer import OnnxInferrer, SklearnMLInferrer
from fdd_system.ML.common.preprocessor import (
    CalibrationZNormalizer,
    DummyPreprocessor,
    MedianRemoval,
    RMSNormalization,
    StandardZNormal,
)
from fdd_system.ML.inference.classification_pipeline import ClassificationPipeline
from fdd_system.ML.inference.known_unknown_pipeline import KnownUnknownClassificationPipeline
from fdd_system.broker.io_helpers import AlertSender


def _resolve_model_format(model_path: str, requested_format: str = "auto") -> str:
    if requested_format in {"sklearn", "onnx"}:
        return requested_format

    suffix = Path(model_path).suffix.lower()
    if suffix == ".onnx":
        return "onnx"
    return "sklearn"


def _load_model_metadata(model_path: str) -> dict[str, Any] | None:
    sidecar = Path(model_path).with_suffix(".meta.json")
    if not sidecar.exists():
        return None

    try:
        raw = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        logging.getLogger(__name__).warning("Ignoring invalid ONNX metadata sidecar: %s", sidecar)
        return None

    if not isinstance(raw, dict):
        return None
    return raw


def _build_embedder(embedder_name: str, *, metadata: dict[str, Any] | None = None):
    kwargs: dict[str, Any] = {}
    embedder_meta = metadata.get("embedder", {}) if metadata else {}
    if isinstance(embedder_meta, dict):
        maybe_kwargs = embedder_meta.get("kwargs", {})
        if isinstance(maybe_kwargs, dict):
            kwargs = dict(maybe_kwargs)

    if embedder_name == "ml1":
        return MLEmbedder1()
    if embedder_name == "ml2":
        if "highpass_hz" not in kwargs:
            kwargs["highpass_hz"] = 10
        return MLEmbedder2(**kwargs)

    if embedder_name == "spectrogram2d":
        return Spectrogram2DEmbedder(**kwargs)
    if embedder_name == "raw1dcnn":
        return Raw1DCNNEmbedder(**kwargs)

    raise ValueError(f"Unknown embedder '{embedder_name}'.")


def _build_preprocessor(preprocessor_name: str, *, metadata: dict[str, Any] | None = None):
    kwargs: dict[str, Any] = {}
    pre_meta = metadata.get("preprocessor", {}) if metadata else {}
    if isinstance(pre_meta, dict):
        meta_name = pre_meta.get("name")
        maybe_kwargs = pre_meta.get("kwargs", {})
        if isinstance(meta_name, str) and isinstance(maybe_kwargs, dict):
            compatible_names = {
                "basic": {"basic", "median"},
                "median": {"basic", "median"},
                "calibration": {"calibration", "calibration_z"},
                "calibration_z": {"calibration", "calibration_z"},
                "dummy": {"dummy"},
                "robust": {"robust", "standard"},
                "standard": {"robust", "standard"},
                "rms": {"rms"},
            }
            if meta_name in compatible_names.get(preprocessor_name, {preprocessor_name}):
                kwargs = dict(maybe_kwargs)

    if preprocessor_name in {"basic", "median"}:
        return MedianRemoval()
    if preprocessor_name in {"calibration", "calibration_z"}:
        return CalibrationZNormalizer(**kwargs)
    if preprocessor_name == "dummy":
        return DummyPreprocessor()
    if preprocessor_name in {"robust", "standard"}:
        return StandardZNormal(**kwargs)
    if preprocessor_name == "rms":
        return RMSNormalization(**kwargs)

    raise ValueError(f"Unknown preprocessor '{preprocessor_name}'.")


def load_model(model_path: str, model_format: str):
    """Load a trained model from disk."""
    if model_format == "onnx":
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise ImportError("onnxruntime is required to load ONNX models.") from exc
        return ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])

    if model_format == "sklearn":
        if joblib is None:
            raise ImportError("joblib is required to load sklearn models; install it or adjust load_model.")
        return joblib.load(model_path)

    raise ValueError(f"Unsupported model format '{model_format}'.")


def build_pipeline(
    model_path: str,
    *,
    model_format: str = "auto",
    embedder: str = "auto",
    preprocessor: str = "auto",
    anomaly_detector_path: str | None = None,
) -> ClassificationPipeline | KnownUnknownClassificationPipeline:
    """Construct the end-to-end classification pipeline."""
    resolved_model_format = _resolve_model_format(model_path, model_format)
    metadata = _load_model_metadata(model_path)

    if embedder == "auto":
        embedder_name = "ml2" if resolved_model_format == "sklearn" else "spectrogram2d"
        embedder_meta = metadata.get("embedder", {}) if metadata else {}
        if isinstance(embedder_meta, dict) and isinstance(embedder_meta.get("name"), str):
            embedder_name = embedder_meta["name"]
    else:
        embedder_name = embedder

    if preprocessor == "auto":
        preprocessor_name = "basic"
        pre_meta = metadata.get("preprocessor", {}) if metadata else {}
        if isinstance(pre_meta, dict) and isinstance(pre_meta.get("name"), str):
            preprocessor_name = pre_meta["name"]
    else:
        preprocessor_name = preprocessor

    model = load_model(model_path, resolved_model_format)
    pre = _build_preprocessor(preprocessor_name, metadata=metadata)
    emb = _build_embedder(embedder_name, metadata=metadata)
    inf = OnnxInferrer(model) if resolved_model_format == "onnx" else SklearnMLInferrer(model)
    classifier_pipeline = ClassificationPipeline(pre, emb, inf)

    if anomaly_detector_path is None:
        return classifier_pipeline

    anomaly_detector = load_anomaly_detector(anomaly_detector_path)
    return KnownUnknownClassificationPipeline(classifier_pipeline, anomaly_detector)


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


def log_prediction_counts(prediction_counts: Counter[int], log: logging.Logger) -> None:
    if not prediction_counts:
        return

    log.info("Prediction counts:")
    for cls_id, count in prediction_counts.items():
        try:
            cls_name = OperatingCondition(cls_id).name
        except ValueError:
            cls_name = f"Unknown({cls_id})"
        log.info("  %s: %s", cls_name, count)


def _is_calibration_preprocessor(value: object) -> bool:
    return isinstance(value, CalibrationZNormalizer)


def _calibration_preprocessor_name(value: object) -> str:
    if isinstance(value, CalibrationZNormalizer):
        return "calibration_z"
    raise TypeError(f"Unsupported calibration preprocessor: {type(value)!r}")


def _fit_runtime_calibration_preprocessor(
    template,
    calibration_windows,
):
    if isinstance(template, CalibrationZNormalizer):
        return CalibrationZNormalizer.fit(calibration_windows)
    raise TypeError(f"Unsupported calibration preprocessor template: {type(template)!r}")


def collect_calibration_targets(pipeline) -> list[tuple[object, str, str]]:
    targets: list[tuple[object, str, str]] = []

    direct_pre = getattr(pipeline, "preprocessor", None)
    if _is_calibration_preprocessor(direct_pre):
        targets.append((pipeline, "preprocessor", "preprocessor"))

    classifier_pipeline = getattr(pipeline, "classifier_pipeline", None)
    classifier_pre = getattr(classifier_pipeline, "preprocessor", None)
    if _is_calibration_preprocessor(classifier_pre):
        targets.append((classifier_pipeline, "preprocessor", "preprocessor"))

    anomaly_detector = getattr(pipeline, "anomaly_detector", None)
    anomaly_pre = getattr(anomaly_detector, "preprocessor", None)
    if _is_calibration_preprocessor(anomaly_pre):
        targets.append((anomaly_detector, "preprocessor", "preprocessor"))

    return targets


def apply_runtime_calibration(pipeline, calibration_windows) -> None:
    for owner, attr, target_kind in collect_calibration_targets(pipeline):
        template = getattr(owner, attr)
        if target_kind == "preprocessor":
            calibrated = _fit_runtime_calibration_preprocessor(template, calibration_windows)
            preprocessor_name = _calibration_preprocessor_name(calibrated)
            preprocessor_kwargs = calibrated.export_kwargs()
            setattr(owner, attr, calibrated)
            if hasattr(owner, "preprocessor_name"):
                owner.preprocessor_name = preprocessor_name
            if hasattr(owner, "preprocessor_kwargs"):
                owner.preprocessor_kwargs = preprocessor_kwargs
            if hasattr(owner, "bundle") and isinstance(getattr(owner, "bundle"), dict):
                owner.bundle["preprocessor_name"] = preprocessor_name
                owner.bundle["preprocessor_kwargs"] = preprocessor_kwargs
            continue

        raise ValueError(f"Unknown calibration target kind: {target_kind}")


def apply_runtime_gate_calibration(
    pipeline,
    calibration_windows,
    *,
    enabled: bool,
    distance_quantile: float,
    distance_margin: float,
    ambiguity_quantile: float,
    ambiguity_slack: float,
):
    if not enabled:
        return None

    anomaly_detector = getattr(pipeline, "anomaly_detector", None)
    if anomaly_detector is None:
        return None

    recalibrate = getattr(anomaly_detector, "recalibrate_from_normal_data", None)
    if not callable(recalibrate):
        return None

    return recalibrate(
        calibration_windows,
        distance_quantile=distance_quantile,
        distance_margin=distance_margin,
        ambiguity_quantile=ambiguity_quantile,
        ambiguity_slack=ambiguity_slack,
    )


def format_label_counts(counts: dict[int, int]) -> str:
    if not counts:
        return "none"

    parts = []
    for cls_id, count in sorted(counts.items()):
        try:
            cls_name = OperatingCondition(cls_id).name
        except ValueError:
            cls_name = f"Unknown({cls_id})"
        parts.append(f"{cls_name}={count}")
    return ", ".join(parts)
