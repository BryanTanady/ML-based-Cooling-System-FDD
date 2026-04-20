"""Backward-compatible wrapper for legacy imports."""

from fdd_system.ML.pipeline import (
    ClassificationPipeline,
    KnownUnknownClassificationPipeline,
    NormalityFaultClassificationPipeline,
)

__all__ = [
    "ClassificationPipeline",
    "KnownUnknownClassificationPipeline",
    "NormalityFaultClassificationPipeline",
]
