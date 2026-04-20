"""Machine-learning package for the FDD system.

Primary module groups:
- `fdd_system.ML.schema`: shared data types and fixed system constants.
- `fdd_system.ML.components`: reusable preprocessors, embedders, models, inferrers, and detectors.
- `fdd_system.ML.pipeline`: runtime pipeline classes.
- `fdd_system.ML.train`: CLI training entrypoint.
"""

__all__ = [
    "components",
    "pipeline",
    "schema",
    "common",
    "inference",
    "train",
]
