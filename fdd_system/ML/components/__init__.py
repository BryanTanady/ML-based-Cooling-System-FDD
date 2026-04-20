"""Reusable ML building blocks shared by training and runtime code."""

from fdd_system.ML.components.detector import (
    MahalanobisAnomalyDetector,
    Stage0WindowGuard,
    fit_mahalanobis_gatekeeper,
    load_anomaly_detector,
    predict_gatekeeper,
    save_anomaly_detector_artifact,
    save_mahalanobis_gatekeeper,
)
from fdd_system.ML.components.embedding import (
    Embedder,
    MLEmbedder1,
    MLEmbedder2,
    Raw1DCNNEmbedder,
    Spectrogram2DEmbedder,
)
from fdd_system.ML.components.inferrer import Inferrer, OnnxInferrer, SklearnMLInferrer, TorchInferrer
from fdd_system.ML.components.model import (
    Fan1DCNN,
    Fan1DCNNV2,
    FanSpectrogramCNN,
    HybridTimeFreq1DCNN,
    ResBlock1D,
    build_classifier_model,
)
from fdd_system.ML.components.preprocessing import (
    BasicPreprocessor,
    CenteredRMSNormalization,
    DummyPreprocessor,
    MedianRemoval,
    Preprocessor,
    RMSNormalization,
    RobustPreprocessor,
    StandardZNormal,
)

__all__ = [
    "BasicPreprocessor",
    "CenteredRMSNormalization",
    "DummyPreprocessor",
    "Embedder",
    "Fan1DCNN",
    "Fan1DCNNV2",
    "FanSpectrogramCNN",
    "HybridTimeFreq1DCNN",
    "Inferrer",
    "MLEmbedder1",
    "MLEmbedder2",
    "MahalanobisAnomalyDetector",
    "MedianRemoval",
    "OnnxInferrer",
    "Preprocessor",
    "RMSNormalization",
    "Raw1DCNNEmbedder",
    "ResBlock1D",
    "RobustPreprocessor",
    "SklearnMLInferrer",
    "Spectrogram2DEmbedder",
    "Stage0WindowGuard",
    "StandardZNormal",
    "TorchInferrer",
    "build_classifier_model",
    "fit_mahalanobis_gatekeeper",
    "load_anomaly_detector",
    "predict_gatekeeper",
    "save_anomaly_detector_artifact",
    "save_mahalanobis_gatekeeper",
]
