"""Backward-compatible wrapper for legacy training package imports."""

__all__ = ["DEFAULT_CONFIG_PATH", "main", "run_training"]


def __getattr__(name: str):
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from fdd_system.ML.train import DEFAULT_CONFIG_PATH, main, run_training

    exports = {
        "DEFAULT_CONFIG_PATH": DEFAULT_CONFIG_PATH,
        "main": main,
        "run_training": run_training,
    }
    return exports[name]
