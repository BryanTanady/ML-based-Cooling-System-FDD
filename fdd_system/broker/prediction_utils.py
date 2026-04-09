"""Prediction, pipeline construction, and calibration helpers for broker runtime."""

from __future__ import annotations

import json
import logging
import pickle
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

try:
    import joblib
except ImportError:  # pragma: no cover - exercised by runtime environment
    joblib = None

try:
    import torch
except ImportError:  # pragma: no cover - exercised by runtime environment
    torch = None

from fdd_system.ML.common.anomaly_detector import load_anomaly_detector
from fdd_system.ML.common.config import OperatingCondition
from fdd_system.ML.common.embedder import (
    MLEmbedder1,
    MLEmbedder2,
    Raw1DCNNEmbedder,
    Spectrogram2DEmbedder,
)
from fdd_system.ML.common.inferrer import OnnxInferrer, SklearnMLInferrer, TorchInferrer
from fdd_system.ML.common.model import build_classifier_model
from fdd_system.ML.common.preprocessor import (
    CalibrationZNormalizer,
    CenteredRMSNormalization,
    DummyPreprocessor,
    MedianRemoval,
    RMSNormalization,
    StandardZNormal,
)
from fdd_system.ML.inference.classification_pipeline import ClassificationPipeline
from fdd_system.ML.inference.known_unknown_pipeline import KnownUnknownClassificationPipeline
from fdd_system.ML.inference.normality_fault_pipeline import NormalityFaultClassificationPipeline
from fdd_system.broker.io_helpers import AlertSender


def _resolve_model_format(model_path: str, requested_format: str = "auto") -> str:
    if requested_format in {"sklearn", "onnx", "torch"}:
        return requested_format

    suffix = Path(model_path).suffix.lower()
    if suffix == ".onnx":
        return "onnx"
    if suffix in {".pt", ".pth"}:
        return "torch"
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


def _extract_idx_to_label_map(metadata: dict[str, Any] | None) -> dict[int, int] | None:
    labels_meta = metadata.get("labels") if isinstance(metadata, dict) else None
    if not isinstance(labels_meta, dict):
        return None

    idx_to_label: dict[int, int] = {}
    for raw_idx, raw_label in labels_meta.items():
        try:
            idx = int(raw_idx)
            label = int(raw_label)
        except (TypeError, ValueError):
            continue
        idx_to_label[idx] = label

    if not idx_to_label:
        return None
    return dict(sorted(idx_to_label.items()))


def _extract_classifier_label_values(metadata: dict[str, Any] | None) -> list[int] | None:
    idx_to_label = _extract_idx_to_label_map(metadata)
    if idx_to_label is None:
        return None

    label_values = [int(label) for label in idx_to_label.values()]
    if not label_values:
        return None
    return sorted(set(label_values))


def _classifier_can_emit_normal_label(metadata: dict[str, Any] | None) -> bool | None:
    label_values = _extract_classifier_label_values(metadata)
    if label_values is None:
        return None
    return int(OperatingCondition.NORMAL.value) in set(label_values)


def _candidate_normality_detector_paths(
    model_path: str,
    anomaly_detector_path: str | None,
) -> list[Path]:
    model_file = Path(model_path)
    candidates: list[Path] = []
    model_stem = model_file.stem

    model_family_prefixes: list[str] = []
    dataset_style_match = re.match(r"^(end_to_end_data_\d+)(?:_.+)?$", model_stem)
    if dataset_style_match:
        model_family_prefixes.append(dataset_style_match.group(1))

    legacy_dataset_style_match = re.match(r"^(end_to_end__data_\d+)(?:__.+)?$", model_stem)
    if legacy_dataset_style_match:
        model_family_prefixes.append(legacy_dataset_style_match.group(1))

    if anomaly_detector_path:
        anomaly_file = Path(anomaly_detector_path)
        anomaly_stem = anomaly_file.stem
        anomaly_suffix = anomaly_file.suffix

        for family_prefix in model_family_prefixes:
            candidates.append(anomaly_file.with_name(f"{family_prefix}_normality_detector{anomaly_suffix}"))
            if anomaly_suffix.lower() != ".pt":
                candidates.append(anomaly_file.with_name(f"{family_prefix}_normality_detector.pt"))

        if "anomaly_gate" in anomaly_stem:
            candidates.append(
                anomaly_file.with_name(anomaly_stem.replace("anomaly_gate", "normality_detector") + anomaly_suffix)
            )
        if "anomaly" in anomaly_stem:
            candidates.append(
                anomaly_file.with_name(anomaly_stem.replace("anomaly", "normality") + anomaly_suffix)
            )

        candidates.append(anomaly_file.with_name(f"{model_file.stem}_normality_detector{anomaly_suffix}"))
        candidates.append(anomaly_file.with_name(f"end_to_end_normality_detector{anomaly_suffix}"))
        candidates.append(anomaly_file.with_name("end_to_end_normality_detector.pt"))

    for family_prefix in model_family_prefixes:
        candidates.append(model_file.with_name(f"{family_prefix}_normality_detector.pt"))

    candidates.append(model_file.with_name("end_to_end_normality_detector.pt"))

    deduped: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        deduped.append(candidate)
    return deduped


def _autodetect_normality_detector_path(
    model_path: str,
    anomaly_detector_path: str | None,
) -> str | None:
    for candidate in _candidate_normality_detector_paths(model_path, anomaly_detector_path):
        if candidate.exists() and candidate.is_file():
            return candidate.as_posix()
    return None


class _LabelMappedInferrer:
    """Adapter that remaps inferrer class indices to exported label ids."""

    def __init__(self, base_inferrer, idx_to_label: dict[int, int]):
        self._base_inferrer = base_inferrer
        self._idx_to_label = {int(idx): int(label) for idx, label in idx_to_label.items()}
        self.model = getattr(base_inferrer, "model", None)

    def _map_preds(self, preds: np.ndarray) -> np.ndarray:
        preds_arr = np.asarray(preds, dtype=np.int64).reshape(-1)
        mapped = [self._idx_to_label.get(int(idx), int(idx)) for idx in preds_arr.tolist()]
        return np.asarray(mapped, dtype=np.int64)

    def infer(self, embeddings: np.ndarray) -> np.ndarray:
        return self._map_preds(self._base_inferrer.infer(embeddings))

    def infer_with_confidence(self, embeddings: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        preds, conf = self._base_inferrer.infer_with_confidence(embeddings)
        return self._map_preds(preds), np.asarray(conf, dtype=float)

    def __getattr__(self, name: str):
        return getattr(self._base_inferrer, name)


def _build_embedder(embedder_name: str, *, metadata: dict[str, Any] | None = None):
    kwargs: dict[str, Any] = {}
    embedder_meta = metadata.get("embedder", {}) if metadata else {}
    if isinstance(embedder_meta, dict):
        maybe_kwargs = embedder_meta.get("kwargs", {})
        if isinstance(maybe_kwargs, dict):
            kwargs = dict(maybe_kwargs)

    axis_names_meta = metadata.get("model_axis_names") if isinstance(metadata, dict) else None
    if not isinstance(axis_names_meta, list):
        axis_names_meta = metadata.get("axis_names") if isinstance(metadata, dict) else None
    if isinstance(axis_names_meta, list) and axis_names_meta and "axis_names" not in kwargs:
        kwargs["axis_names"] = axis_names_meta
    elif isinstance(metadata, dict) and bool(metadata.get("drop_z_axis")) and "axis_names" not in kwargs:
        kwargs["axis_names"] = ["x", "y"]

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
                "centered_rms": {"centered_rms"},
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
    if preprocessor_name == "centered_rms":
        return CenteredRMSNormalization(**kwargs)

    raise ValueError(f"Unknown preprocessor '{preprocessor_name}'.")


def _infer_num_classes_from_state_dict(state_dict: dict[str, Any]) -> int | None:
    if not isinstance(state_dict, dict) or not state_dict:
        return None

    classifier_candidates: list[tuple[int, str]] = []
    for key, value in state_dict.items():
        if not (key.startswith("classifier.") and key.endswith(".weight")):
            continue
        if not hasattr(value, "shape"):
            continue
        shape = tuple(int(v) for v in value.shape)
        if len(shape) != 2 or shape[0] <= 0:
            continue
        parts = key.split(".")
        layer_idx = -1
        if len(parts) >= 3:
            try:
                layer_idx = int(parts[1])
            except ValueError:
                layer_idx = -1
        classifier_candidates.append((layer_idx, key))

    if classifier_candidates:
        classifier_candidates.sort()
        _, best_key = classifier_candidates[-1]
        return int(state_dict[best_key].shape[0])

    linear_candidates: list[str] = []
    for key, value in state_dict.items():
        if not key.endswith(".weight"):
            continue
        if not hasattr(value, "shape"):
            continue
        shape = tuple(int(v) for v in value.shape)
        if len(shape) == 2 and shape[0] > 0:
            linear_candidates.append(key)

    if linear_candidates:
        linear_candidates.sort()
        return int(state_dict[linear_candidates[-1]].shape[0])
    return None


def _extract_num_classes(
    checkpoint: dict[str, Any],
    *,
    metadata: dict[str, Any] | None = None,
    state_dict: dict[str, Any] | None = None,
) -> int:
    idx_to_label = checkpoint.get("idx_to_label")
    if isinstance(idx_to_label, dict) and idx_to_label:
        return int(len(idx_to_label))

    label_to_idx = checkpoint.get("label_to_idx")
    if isinstance(label_to_idx, dict) and label_to_idx:
        return int(len(label_to_idx))

    labels_meta = metadata.get("labels") if isinstance(metadata, dict) else None
    if isinstance(labels_meta, dict) and labels_meta:
        return int(len(labels_meta))

    inferred = _infer_num_classes_from_state_dict(state_dict or {})
    if inferred is not None:
        return int(inferred)

    raise ValueError("Unable to infer torch classifier class count from checkpoint or metadata.")


def _load_torch_model(model_path: str, *, metadata: dict[str, Any] | None = None):
    if torch is None:
        raise ImportError("torch is required to load .pt/.pth models.")

    checkpoint = _load_torch_checkpoint(model_path)
    if isinstance(checkpoint, torch.nn.Module):
        model = checkpoint
        model.eval()
        return model

    if not isinstance(checkpoint, dict):
        raise TypeError(f"Unsupported torch checkpoint type: {type(checkpoint)!r}")

    state_dict = checkpoint.get("model_state_dict")
    if state_dict is None:
        maybe_state = checkpoint.get("state_dict")
        if isinstance(maybe_state, dict):
            state_dict = maybe_state
        elif checkpoint and all(hasattr(v, "shape") for v in checkpoint.values()):
            state_dict = checkpoint

    if not isinstance(state_dict, dict) or not state_dict:
        raise ValueError("Torch checkpoint must contain model_state_dict/state_dict tensor weights.")

    architecture = checkpoint.get("architecture")
    if not isinstance(architecture, str) or not architecture.strip():
        architecture_meta = metadata.get("architecture") if isinstance(metadata, dict) else None
        if isinstance(architecture_meta, str) and architecture_meta.strip():
            architecture = architecture_meta
        else:
            raise ValueError("Torch checkpoint is missing 'architecture' and metadata has no architecture.")

    n_classes = _extract_num_classes(checkpoint, metadata=metadata, state_dict=state_dict)
    in_channels: int | None = None
    axis_names_meta = metadata.get("model_axis_names") if isinstance(metadata, dict) else None
    if not isinstance(axis_names_meta, list):
        axis_names_meta = checkpoint.get("model_axis_names")
    if isinstance(axis_names_meta, list) and axis_names_meta:
        in_channels = int(len(axis_names_meta))
    elif bool((metadata or {}).get("drop_z_axis")) or bool(checkpoint.get("drop_z_axis")):
        in_channels = 2
    else:
        stem_weight = state_dict.get("stem.0.weight")
        if hasattr(stem_weight, "shape") and len(stem_weight.shape) == 3:
            in_channels = int(stem_weight.shape[1])
        else:
            conv_weight = state_dict.get("features.0.weight")
            if hasattr(conv_weight, "shape") and len(conv_weight.shape) == 3:
                in_channels = int(conv_weight.shape[1])
    if in_channels is None or in_channels <= 0:
        in_channels = 3

    model = build_classifier_model(architecture, n_classes=n_classes, in_channels=in_channels)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def _load_torch_checkpoint(model_path: str):
    # PyTorch 2.6+ defaults torch.load(..., weights_only=True). For legacy
    # trusted checkpoints that contain pickled metadata, retry with False.
    try:
        return torch.load(model_path, map_location="cpu", weights_only=True)
    except pickle.UnpicklingError as exc:
        if "Weights only load failed" not in str(exc):
            raise
        logging.getLogger(__name__).warning(
            "Torch checkpoint %s requires legacy unpickling; retrying with weights_only=False. "
            "Only use trusted model files.",
            model_path,
        )
        return torch.load(model_path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(model_path, map_location="cpu")


def _metadata_from_torch_checkpoint(model_path: str) -> dict[str, Any]:
    if torch is None:
        return {}

    checkpoint = _load_torch_checkpoint(model_path)
    if not isinstance(checkpoint, dict):
        return {}

    merged: dict[str, Any] = {}
    for key in (
        "architecture",
        "drop_z_axis",
        "model_axis_names",
        "axis_names",
        "window_len",
        "preprocessor_name",
        "preprocessor_kwargs",
        "classifier_input_normalization",
    ):
        if key in checkpoint:
            merged[key] = checkpoint[key]

    if isinstance(checkpoint.get("idx_to_label"), dict) and checkpoint["idx_to_label"]:
        merged["labels"] = {str(k): int(v) for k, v in checkpoint["idx_to_label"].items()}

    if "mean" in checkpoint or "std" in checkpoint or "window_len" in checkpoint:
        embedder_kwargs: dict[str, Any] = {}
        if "window_len" in checkpoint:
            try:
                embedder_kwargs["target_len"] = int(checkpoint["window_len"])
            except (TypeError, ValueError):
                pass
        if "mean" in checkpoint:
            embedder_kwargs["mean"] = np.asarray(checkpoint["mean"], dtype=np.float32).reshape(-1).tolist()
        if "std" in checkpoint:
            embedder_kwargs["std"] = np.asarray(checkpoint["std"], dtype=np.float32).reshape(-1).tolist()

        axis_names = checkpoint.get("model_axis_names")
        if not isinstance(axis_names, list):
            axis_names = checkpoint.get("axis_names")
        if isinstance(axis_names, list) and axis_names:
            embedder_kwargs["axis_names"] = axis_names
        elif bool(checkpoint.get("drop_z_axis")):
            embedder_kwargs["axis_names"] = ["x", "y"]

        merged["embedder"] = {
            "name": "raw1dcnn",
            "kwargs": embedder_kwargs,
        }

    pre_name = checkpoint.get("preprocessor_name")
    pre_kwargs = checkpoint.get("preprocessor_kwargs")
    if isinstance(pre_name, str):
        merged["preprocessor"] = {
            "name": pre_name,
            "kwargs": pre_kwargs if isinstance(pre_kwargs, dict) else {},
        }

    return merged


def _merge_model_metadata(
    sidecar_metadata: dict[str, Any] | None,
    *,
    model_path: str,
    resolved_model_format: str,
) -> dict[str, Any] | None:
    merged: dict[str, Any] = dict(sidecar_metadata) if isinstance(sidecar_metadata, dict) else {}
    if resolved_model_format != "torch":
        return merged if merged else None

    checkpoint_meta = _metadata_from_torch_checkpoint(model_path)
    if not checkpoint_meta:
        return merged if merged else None

    if not merged:
        return checkpoint_meta

    for key, value in checkpoint_meta.items():
        if key in {"embedder", "preprocessor"} and isinstance(value, dict):
            existing = merged.get(key)
            if isinstance(existing, dict):
                combined = dict(value)
                combined.update(existing)
                if isinstance(value.get("kwargs"), dict) and isinstance(existing.get("kwargs"), dict):
                    kwargs = dict(value["kwargs"])
                    kwargs.update(existing["kwargs"])
                    combined["kwargs"] = kwargs
                merged[key] = combined
            else:
                merged[key] = value
            continue
        merged.setdefault(key, value)
    return merged


def load_model(model_path: str, model_format: str, *, metadata: dict[str, Any] | None = None):
    """Load a trained model from disk."""
    if model_format == "onnx":
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise ImportError("onnxruntime is required to load ONNX models.") from exc
        return ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])

    if model_format == "torch":
        return _load_torch_model(model_path, metadata=metadata)

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
    normality_detector_path: str | None = None,
) -> ClassificationPipeline | KnownUnknownClassificationPipeline | NormalityFaultClassificationPipeline:
    """Construct the end-to-end classification pipeline."""
    resolved_model_format = _resolve_model_format(model_path, model_format)
    metadata = _merge_model_metadata(
        _load_model_metadata(model_path),
        model_path=model_path,
        resolved_model_format=resolved_model_format,
    )
    classifier_can_emit_normal = _classifier_can_emit_normal_label(metadata)

    if normality_detector_path is None and classifier_can_emit_normal is False:
        inferred_normality_path = _autodetect_normality_detector_path(model_path, anomaly_detector_path)
        if inferred_normality_path is not None:
            logging.getLogger(__name__).warning(
                (
                    "Classifier metadata labels do not include NORMAL (%d). "
                    "Auto-selecting normality detector: %s"
                ),
                OperatingCondition.NORMAL.value,
                inferred_normality_path,
            )
            normality_detector_path = inferred_normality_path
        else:
            raise ValueError(
                (
                    "Classifier metadata labels do not include NORMAL (0), so this model cannot emit NORMAL by itself. "
                    "Provide --normality-detector-path (Stage 2 gate) for broker runtime."
                )
            )

    if embedder == "auto":
        if resolved_model_format == "sklearn":
            embedder_name = "ml2"
        elif resolved_model_format == "torch":
            embedder_name = "raw1dcnn"
        else:
            embedder_name = "spectrogram2d"
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

    model = load_model(model_path, resolved_model_format, metadata=metadata)
    pre = _build_preprocessor(preprocessor_name, metadata=metadata)
    emb = _build_embedder(embedder_name, metadata=metadata)
    if resolved_model_format == "onnx":
        inf = OnnxInferrer(model)
    elif resolved_model_format == "torch":
        inf = TorchInferrer(model)
    else:
        inf = SklearnMLInferrer(model)
    idx_to_label = _extract_idx_to_label_map(metadata)
    if idx_to_label and resolved_model_format in {"onnx", "torch"}:
        inf = _LabelMappedInferrer(inf, idx_to_label)
    classifier_pipeline = ClassificationPipeline(pre, emb, inf)

    if anomaly_detector_path is None and normality_detector_path is None:
        return classifier_pipeline

    if anomaly_detector_path is not None and normality_detector_path is None:
        anomaly_detector = load_anomaly_detector(anomaly_detector_path)
        return KnownUnknownClassificationPipeline(classifier_pipeline, anomaly_detector)

    if anomaly_detector_path is None and normality_detector_path is not None:
        normality_detector = load_anomaly_detector(normality_detector_path)
        return NormalityFaultClassificationPipeline(
            classifier_pipeline=classifier_pipeline,
            normality_detector=normality_detector,
            normal_label=OperatingCondition.NORMAL.value,
            unknown_label=OperatingCondition.UNKNOWN.value,
        )

    # Full 4-stage runtime:
    # Stage 2: normality detector -> Stage 3: known/unknown detector -> Stage 4: fault classifier.
    normality_detector = load_anomaly_detector(normality_detector_path)
    anomaly_detector = load_anomaly_detector(anomaly_detector_path)
    downstream_fault_pipeline = KnownUnknownClassificationPipeline(
        classifier_pipeline=classifier_pipeline,
        anomaly_detector=anomaly_detector,
        unknown_label=OperatingCondition.UNKNOWN.value,
    )
    return NormalityFaultClassificationPipeline(
        classifier_pipeline=downstream_fault_pipeline,
        normality_detector=normality_detector,
        normal_label=OperatingCondition.NORMAL.value,
        unknown_label=OperatingCondition.UNKNOWN.value,
    )


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

    logger.info("")
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


def _label_name(label: object) -> str:
    try:
        return OperatingCondition(int(label)).name
    except (TypeError, ValueError):
        return f"Unknown({label})"


def _fmt_float(value: object, *, precision: int = 4) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "nan"
    if not np.isfinite(number):
        return "nan"
    return f"{number:.{precision}f}"


def _summarize_numeric_array(values: object) -> str:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return "n=0"

    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return f"n={int(arr.size)} finite=0"

    rms = float(np.sqrt(np.mean(finite**2)))
    return (
        f"n={int(arr.size)} "
        f"mu={_fmt_float(np.mean(finite))} "
        f"sd={_fmt_float(np.std(finite))} "
        f"rms={_fmt_float(rms)} "
        f"lo={_fmt_float(np.min(finite))} "
        f"hi={_fmt_float(np.max(finite))}"
    )


def _summarize_window_axes(window: object) -> str:
    parts: list[str] = []
    for axis_name, attr_name in (("x", "acc_x"), ("y", "acc_y"), ("z", "acc_z")):
        values = getattr(window, attr_name, None)
        parts.append(f"{axis_name}[{_summarize_numeric_array(values)}]")
    return " ".join(parts)


def _summarize_feature_sample(sample: object) -> str:
    arr = np.asarray(sample, dtype=np.float32)
    if arr.ndim == 2 and arr.shape[0] == 3:
        parts = []
        for axis_name, axis_values in zip(("x", "y", "z"), arr, strict=False):
            parts.append(f"{axis_name}[{_summarize_numeric_array(axis_values)}]")
        return " ".join(parts)

    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return f"shape={list(arr.shape)} finite=0"

    return (
        f"shape={list(arr.shape)} "
        f"mu={_fmt_float(np.mean(finite))} "
        f"sd={_fmt_float(np.std(finite))} "
        f"lo={_fmt_float(np.min(finite))} "
        f"hi={_fmt_float(np.max(finite))}"
    )


def _collect_anomaly_detectors_for_debug(pipeline) -> list[tuple[str, object]]:
    detectors: list[tuple[str, object]] = []
    current = pipeline
    visited_ids: set[int] = set()
    depth = 0
    while current is not None and id(current) not in visited_ids:
        visited_ids.add(id(current))
        detector = getattr(current, "anomaly_detector", None)
        if detector is not None:
            detectors.append((f"{depth}:{type(current).__name__}", detector))
        current = getattr(current, "classifier_pipeline", None)
        depth += 1
    return detectors


def log_live_debug_stats(
    pipeline,
    raw_inputs,
    logger: logging.Logger,
) -> None:
    samples = list(raw_inputs)
    if not samples:
        return

    try:
        classifier_pipeline = _resolve_classifier_pipeline(pipeline)
        preprocessor = getattr(classifier_pipeline, "preprocessor", None)
        embedder = getattr(classifier_pipeline, "embedder", None)
        inferrer = getattr(classifier_pipeline, "inferrer", None)

        processed_inputs = (
            list(preprocessor.preprocess(samples))
            if preprocessor is not None and hasattr(preprocessor, "preprocess")
            else list(samples)
        )

        feature_batch: np.ndarray | None = None
        if embedder is not None and hasattr(embedder, "embed") and processed_inputs:
            feature_batch = np.asarray(embedder.embed(processed_inputs), dtype=np.float32)

        classifier_preds: np.ndarray | None = None
        classifier_confs: np.ndarray | None = None
        if (
            feature_batch is not None
            and feature_batch.shape[0] > 0
            and inferrer is not None
            and hasattr(inferrer, "infer_with_confidence")
        ):
            classifier_preds, classifier_confs = inferrer.infer_with_confidence(feature_batch)
            classifier_preds = np.asarray(classifier_preds, dtype=np.int64).reshape(-1)
            classifier_confs = np.asarray(classifier_confs, dtype=float).reshape(-1)

        detector_debug_details: list[tuple[str, dict[str, np.ndarray]]] = []
        for detector_name, detector in _collect_anomaly_detectors_for_debug(pipeline):
            predict_details = getattr(detector, "predict_details", None)
            if callable(predict_details):
                detector_debug_details.append((detector_name, predict_details(samples)))

        pre_name = type(preprocessor).__name__ if preprocessor is not None else "none"
        embed_name = type(embedder).__name__ if embedder is not None else "none"
        inferrer_name = type(inferrer).__name__ if inferrer is not None else "none"

        for idx, sample in enumerate(samples):
            lines = [
                f"LiveDebug window={idx}",
                f"  raw   {_summarize_window_axes(sample)}",
            ]
            if idx < len(processed_inputs):
                lines.append(
                    f"  pre[{pre_name}]   {_summarize_window_axes(processed_inputs[idx])}"
                )
            if feature_batch is not None and idx < feature_batch.shape[0]:
                lines.append(
                    f"  emb[{embed_name}]  {_summarize_feature_sample(feature_batch[idx])}"
                )

            for detector_name, details in detector_debug_details:
                stage0_valid = ""
                stage0_reason = ""
                if "stage0_valid" in details and idx < np.asarray(details["stage0_valid"]).size:
                    stage0_valid = str(int(np.asarray(details["stage0_valid"], dtype=np.int64).reshape(-1)[idx]))
                if "stage0_reason" in details and idx < np.asarray(details["stage0_reason"], dtype=object).size:
                    stage0_reason = str(np.asarray(details["stage0_reason"], dtype=object).reshape(-1)[idx])

                nearest_label = -1
                if "nearest_label" in details and idx < np.asarray(details["nearest_label"]).size:
                    nearest_label = int(np.asarray(details["nearest_label"], dtype=np.int64).reshape(-1)[idx])

                second_label = -1
                if "second_label" in details and idx < np.asarray(details["second_label"]).size:
                    second_label = int(np.asarray(details["second_label"], dtype=np.int64).reshape(-1)[idx])

                lines.append(
                    (
                        f"  gate[{detector_name}] "
                        f"s0={stage0_valid or '?'} reason={stage0_reason or '-'} "
                        f"near={_label_name(nearest_label)} d={_fmt_float(np.asarray(details.get('distance', np.empty((0,))), dtype=float).reshape(-1)[idx]) if 'distance' in details and idx < np.asarray(details['distance']).size else 'nan'} "
                        f"thr={_fmt_float(np.asarray(details.get('applied_threshold', np.empty((0,))), dtype=float).reshape(-1)[idx]) if 'applied_threshold' in details and idx < np.asarray(details['applied_threshold']).size else 'nan'} "
                        f"second={_label_name(second_label)} d2={_fmt_float(np.asarray(details.get('second_distance', np.empty((0,))), dtype=float).reshape(-1)[idx]) if 'second_distance' in details and idx < np.asarray(details['second_distance']).size else 'nan'} "
                        f"ratio={_fmt_float(np.asarray(details.get('distance_ratio', np.empty((0,))), dtype=float).reshape(-1)[idx]) if 'distance_ratio' in details and idx < np.asarray(details['distance_ratio']).size else 'nan'} "
                        f"unk={str(int(np.asarray(details.get('is_unknown', np.empty((0,))), dtype=np.int64).reshape(-1)[idx])) if 'is_unknown' in details and idx < np.asarray(details['is_unknown']).size else '?'} "
                        f"conf={_fmt_float(np.asarray(details.get('decision_confidence', np.empty((0,))), dtype=float).reshape(-1)[idx]) if 'decision_confidence' in details and idx < np.asarray(details['decision_confidence']).size else 'nan'}"
                    )
                )

            if classifier_preds is not None and idx < classifier_preds.size:
                lines.append(
                    f"  clf[{inferrer_name}] pred={_label_name(int(classifier_preds[idx]))} "
                    f"conf={_fmt_float(classifier_confs[idx] if classifier_confs is not None and idx < classifier_confs.size else np.nan)}"
                )

            logger.info("")
            logger.info("\n".join(lines))
            logger.info("")
    except Exception as exc:  # pragma: no cover - best-effort debug instrumentation
        logger.warning("Live debug stats failed: %s", exc)


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
    seen_objects: set[int] = set()

    def add_if_calibration_pre(owner: object, attr: str) -> None:
        owner_id = id(owner)
        if owner_id in seen_objects:
            return
        value = getattr(owner, attr, None)
        if _is_calibration_preprocessor(value):
            targets.append((owner, attr, "preprocessor"))
            seen_objects.add(owner_id)

    current = pipeline
    visited_pipeline_ids: set[int] = set()
    while current is not None and id(current) not in visited_pipeline_ids:
        visited_pipeline_ids.add(id(current))
        add_if_calibration_pre(current, "preprocessor")

        detector = getattr(current, "anomaly_detector", None)
        if detector is not None:
            add_if_calibration_pre(detector, "preprocessor")

        normality_detector = getattr(current, "normality_detector", None)
        if normality_detector is not None:
            add_if_calibration_pre(normality_detector, "preprocessor")

        current = getattr(current, "classifier_pipeline", None)

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


def supports_runtime_gate_calibration(pipeline) -> bool:
    anomaly_detector = getattr(pipeline, "anomaly_detector", None)
    recalibrate = getattr(anomaly_detector, "recalibrate_from_normal_data", None)
    return callable(recalibrate)


def _iter_batchnorm_layers(model):
    if torch is None:
        return
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            yield module


def _resolve_classifier_pipeline(pipeline):
    current = pipeline
    visited_ids: set[int] = set()
    while current is not None and id(current) not in visited_ids:
        visited_ids.add(id(current))
        classifier_pipeline = getattr(current, "classifier_pipeline", None)
        if classifier_pipeline is None:
            return current
        current = classifier_pipeline
    return pipeline


def supports_runtime_adabn(pipeline) -> bool:
    if torch is None:
        return False
    classifier_pipeline = _resolve_classifier_pipeline(pipeline)
    inferrer = getattr(classifier_pipeline, "inferrer", None)
    model = getattr(inferrer, "model", None)
    if not isinstance(model, torch.nn.Module):
        return False
    return any(True for _ in _iter_batchnorm_layers(model))


@torch.no_grad() if torch is not None else (lambda fn: fn)
def recalibrate_batchnorm_stats(
    model,
    x_np,
    *,
    batch_size: int | None = None,
    ema_momentum: float = 0.1,
) -> dict[str, Any]:
    if torch is None:
        raise ImportError("AdaBN recalibration requires torch.")
    if not isinstance(model, torch.nn.Module):
        raise TypeError("AdaBN recalibration expects a torch.nn.Module classifier.")

    x_np = np.asarray(x_np, dtype=np.float32)
    if x_np.ndim < 2:
        raise ValueError(f"Expected AdaBN tensor with shape [N, ...], got {x_np.shape}.")
    if x_np.shape[0] <= 0:
        raise ValueError("AdaBN recalibration requires at least one window.")
    ema_momentum = float(ema_momentum)
    if not (0.0 < ema_momentum <= 1.0):
        raise ValueError(f"AdaBN ema_momentum must be in (0, 1], got {ema_momentum}.")

    bn_layers = list(_iter_batchnorm_layers(model))
    if not bn_layers:
        return {
            "bn_layers": 0,
            "adaptation_windows": int(x_np.shape[0]),
            "adaptation_batch_size": int(x_np.shape[0]),
            "adaptation_batches": 1,
            "ema_momentum": ema_momentum,
        }

    old_training_mode = bool(model.training)
    old_momentums: list[float | None] = []
    old_track_running_stats: list[bool] = []
    old_training_modes: list[bool] = []

    for param in model.parameters():
        param.requires_grad_(False)

    model.eval()
    for module in bn_layers:
        old_momentums.append(module.momentum)
        old_track_running_stats.append(bool(module.track_running_stats))
        old_training_modes.append(bool(module.training))
        module.train()
        module.track_running_stats = True
        module.momentum = ema_momentum

    first_param = next(model.parameters(), None)
    device = first_param.device if first_param is not None else torch.device("cpu")

    effective_batch_size = int(x_np.shape[0]) if batch_size is None or int(batch_size) <= 0 else int(batch_size)
    effective_batch_size = max(1, min(effective_batch_size, int(x_np.shape[0])))

    batches = 0
    for start in range(0, int(x_np.shape[0]), effective_batch_size):
        xb = torch.from_numpy(x_np[start : start + effective_batch_size]).float().to(device)
        _ = model(xb)
        batches += 1

    for module, momentum, track_running_stats, training_mode in zip(
        bn_layers, old_momentums, old_track_running_stats, old_training_modes
    ):
        module.momentum = momentum
        module.track_running_stats = track_running_stats
        module.train(training_mode)

    if old_training_mode:
        model.train()
    else:
        model.eval()

    return {
        "bn_layers": int(len(bn_layers)),
        "adaptation_windows": int(x_np.shape[0]),
        "adaptation_batch_size": int(effective_batch_size),
        "adaptation_batches": int(batches),
        "ema_momentum": ema_momentum,
    }


def apply_runtime_adabn(
    pipeline,
    calibration_windows,
    *,
    enabled: bool,
    batch_size: int | None = None,
    ema_momentum: float = 0.1,
) -> dict[str, Any] | None:
    if not enabled or not calibration_windows:
        return None
    if torch is None:
        return None

    classifier_pipeline = _resolve_classifier_pipeline(pipeline)
    preprocessor = getattr(classifier_pipeline, "preprocessor", None)
    embedder = getattr(classifier_pipeline, "embedder", None)
    inferrer = getattr(classifier_pipeline, "inferrer", None)
    model = getattr(inferrer, "model", None)

    if preprocessor is None or embedder is None or not isinstance(model, torch.nn.Module):
        return None
    if not any(True for _ in _iter_batchnorm_layers(model)):
        return None

    adapted_inputs = preprocessor.preprocess(list(calibration_windows))
    feature_batch = np.asarray(embedder.embed(adapted_inputs), dtype=np.float32)
    summary = recalibrate_batchnorm_stats(
        model,
        feature_batch,
        batch_size=batch_size,
        ema_momentum=ema_momentum,
    )
    summary["feature_shape"] = [int(v) for v in feature_batch.shape]
    summary["preprocessed_windows"] = int(len(adapted_inputs))
    return summary


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
