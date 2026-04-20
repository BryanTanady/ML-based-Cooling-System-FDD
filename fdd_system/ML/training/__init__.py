"""Backward-compatible wrapper for legacy training package imports."""

from fdd_system.ML.train import DEFAULT_CONFIG_PATH, main, run_training

__all__ = [
    "DEFAULT_CONFIG_PATH",
    "main",
    "run_training",
]
