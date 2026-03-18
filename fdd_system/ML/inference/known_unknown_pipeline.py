from __future__ import annotations

import numpy as np

from fdd_system.ML.common.anomaly_detector import MahalanobisAnomalyDetector
from fdd_system.ML.common.config import OperatingCondition, RawInput
from fdd_system.ML.inference.classification_pipeline import ClassificationPipeline


class KnownUnknownClassificationPipeline:
    """Two-stage inference wrapper that emits UNKNOWN before known-class inference."""

    def __init__(
        self,
        classifier_pipeline: ClassificationPipeline,
        anomaly_detector: MahalanobisAnomalyDetector,
        *,
        unknown_label: int = OperatingCondition.UNKNOWN.value,
    ):
        self.classifier_pipeline = classifier_pipeline
        self.anomaly_detector = anomaly_detector
        self.unknown_label = int(unknown_label)

    def predict(self, raw_input: list[RawInput]) -> np.ndarray:
        preds, _ = self.predict_with_confidence(raw_input)
        return preds

    def predict_with_confidence(self, raw_input: list[RawInput]) -> tuple[np.ndarray, np.ndarray]:
        samples = list(raw_input)
        gate_preds, gate_conf = self.anomaly_detector.predict_with_confidence(samples)
        gate_preds = np.asarray(gate_preds, dtype=np.int64).reshape(-1)
        gate_conf = np.asarray(gate_conf, dtype=float).reshape(-1)

        final_preds = np.full(gate_preds.shape, fill_value=self.unknown_label, dtype=np.int64)
        final_conf = gate_conf.copy()

        known_indices = np.flatnonzero(gate_preds == 0)
        if known_indices.size == 0:
            return final_preds, final_conf

        known_inputs = [samples[idx] for idx in known_indices.tolist()]
        class_preds, class_conf = self.classifier_pipeline.predict_with_confidence(known_inputs)
        class_preds = np.asarray(class_preds, dtype=np.int64).reshape(-1)
        class_conf = np.asarray(class_conf, dtype=float).reshape(-1)

        final_preds[known_indices] = class_preds
        final_conf[known_indices] = class_conf
        return final_preds, final_conf
