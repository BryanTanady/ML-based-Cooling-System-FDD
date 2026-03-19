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
        details = self.predict_details(raw_input)
        return details["predictions"], details["confidence"]

    def predict_details(self, raw_input: list[RawInput]) -> dict[str, np.ndarray]:
        samples = list(raw_input)
        gate_details = self.anomaly_detector.predict_details(samples)
        gate_preds = np.asarray(gate_details["is_unknown"], dtype=np.int64).reshape(-1)
        gate_conf = np.asarray(gate_details["decision_confidence"], dtype=float).reshape(-1)

        final_preds = np.full(gate_preds.shape, fill_value=self.unknown_label, dtype=np.int64)
        final_conf = gate_conf.copy()
        rejection_stage = np.full(gate_preds.shape, fill_value="", dtype=object)
        rejection_reason = np.full(gate_preds.shape, fill_value="", dtype=object)

        stage0_valid = np.asarray(
            gate_details.get("stage0_valid", np.ones(gate_preds.shape, dtype=np.int64)),
            dtype=np.int64,
        ).reshape(-1)
        stage0_reason = np.asarray(
            gate_details.get("stage0_reason", np.full(gate_preds.shape, fill_value="", dtype=object)),
            dtype=object,
        ).reshape(-1)

        stage0_rejected = (gate_preds == 1) & (stage0_valid == 0)
        stage1_rejected = (gate_preds == 1) & (stage0_valid != 0)
        rejection_stage[stage0_rejected] = "STAGE0"
        rejection_stage[stage1_rejected] = "STAGE1"
        rejection_reason[stage0_rejected] = stage0_reason[stage0_rejected]

        known_indices = np.flatnonzero(gate_preds == 0)
        if known_indices.size == 0:
            return {
                "predictions": final_preds,
                "confidence": final_conf.astype(np.float32),
                "gate_is_unknown": gate_preds,
                "gate_confidence": gate_conf.astype(np.float32),
                "rejection_stage": rejection_stage,
                "rejection_reason": rejection_reason,
            }

        known_inputs = [samples[idx] for idx in known_indices.tolist()]
        class_preds, class_conf = self.classifier_pipeline.predict_with_confidence(known_inputs)
        class_preds = np.asarray(class_preds, dtype=np.int64).reshape(-1)
        class_conf = np.asarray(class_conf, dtype=float).reshape(-1)

        final_preds[known_indices] = class_preds
        final_conf[known_indices] = class_conf
        return {
            "predictions": final_preds,
            "confidence": final_conf.astype(np.float32),
            "gate_is_unknown": gate_preds,
            "gate_confidence": gate_conf.astype(np.float32),
            "rejection_stage": rejection_stage,
            "rejection_reason": rejection_reason,
        }
