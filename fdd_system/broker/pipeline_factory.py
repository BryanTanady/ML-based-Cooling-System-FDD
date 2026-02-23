"""Factory utilities for building broker classification pipelines."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

try:
    import joblib
except ImportError:  # pragma: no cover - exercised by runtime environment
    joblib = None

from fdd_system.ML.common.classification.embedder import (
    MLEmbedder1,
    MLEmbedder2,
    Raw1DCNNEmbedder,
    Spectrogram2DEmbedder,
)
from fdd_system.ML.common.classification.inferrer import OnnxInferrer, SklearnMLInferrer
from fdd_system.ML.common.classification.preprocessor import (
    DummyPreprocessor,
    MedianRemoval,
    RMSNormalization,
    StandardZNormal,
)
from fdd_system.ML.inference.classification_pipeline import ClassificationPipeline


def _resolve_model_format(model_path: str, requested_format: str = "auto") -> str:
    if requested_format in {"sklearn", "onnx"}:
        return requested_format

    suffix = Path(model_path).suffix.lower()
    if suffix == ".onnx":
        return "onnx"
    return "sklearn"


def _load_onnx_metadata(model_path: str) -> dict[str, Any] | None:
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
    if embedder_name == "ml1":
        return MLEmbedder1()
    if embedder_name == "ml2":
        return MLEmbedder2(highpass_hz=10)

    kwargs: dict[str, Any] = {}
    embedder_meta = metadata.get("embedder", {}) if metadata else {}
    if isinstance(embedder_meta, dict):
        maybe_kwargs = embedder_meta.get("kwargs", {})
        if isinstance(maybe_kwargs, dict):
            kwargs = dict(maybe_kwargs)

    if embedder_name == "spectrogram2d":
        return Spectrogram2DEmbedder(**kwargs)
    if embedder_name == "raw1dcnn":
        return Raw1DCNNEmbedder(**kwargs)

    raise ValueError(f"Unknown embedder '{embedder_name}'.")


def _build_preprocessor(preprocessor_name: str, *, metadata: dict[str, Any] | None = None):
    kwargs: dict[str, Any] = {}
    pre_meta = metadata.get("preprocessor", {}) if metadata else {}
    if isinstance(pre_meta, dict):
        maybe_kwargs = pre_meta.get("kwargs", {})
        if isinstance(maybe_kwargs, dict):
            kwargs = dict(maybe_kwargs)

    if preprocessor_name in {"basic", "median"}:
        return MedianRemoval()
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
) -> ClassificationPipeline:
    """Construct the end-to-end classification pipeline."""
    resolved_model_format = _resolve_model_format(model_path, model_format)
    metadata = _load_onnx_metadata(model_path) if resolved_model_format == "onnx" else None

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
    return ClassificationPipeline(pre, emb, inf)
