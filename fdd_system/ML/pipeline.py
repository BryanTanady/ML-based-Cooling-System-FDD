from __future__ import annotations

import numpy as np

from fdd_system.ML.components.detector import MahalanobisAnomalyDetector
from fdd_system.ML.components.embedding import Embedder
from fdd_system.ML.components.inferrer import Inferrer
from fdd_system.ML.components.preprocessing import Preprocessor
from fdd_system.ML.schema import OperatingCondition, RawInput


class ClassificationPipeline:
    """Compose preprocessing, embedding, and inference into one classifier."""

    def __init__(self, preprocessor: Preprocessor, embedder: Embedder, inferrer: Inferrer):
        self.preprocessor = preprocessor
        self.embedder = embedder
        self.inferrer = inferrer

    def predict(self, raw_input):
        cleaned_input = self.preprocessor.preprocess(raw_input)
        feature_map = self.embedder.embed(cleaned_input)
        return self.inferrer.infer(feature_map)

    def predict_with_confidence(self, raw_input):
        """Return predictions and confidence scores per sample."""
        cleaned_input = self.preprocessor.preprocess(raw_input)
        feature_map = self.embedder.embed(cleaned_input)
        return self.inferrer.infer_with_confidence(feature_map)


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


class NormalityFaultClassificationPipeline:
    """Two-stage inference wrapper: normality gate first, then fault classification."""

    def __init__(
        self,
        classifier_pipeline: ClassificationPipeline,
        normality_detector: MahalanobisAnomalyDetector,
        *,
        normal_label: int = OperatingCondition.NORMAL.value,
        unknown_label: int = OperatingCondition.UNKNOWN.value,
        fault_confidence_threshold: float | None = None,
        per_class_fault_confidence_thresholds: dict[int, float] | None = None,
        fault_support_threshold: float | None = None,
        per_class_fault_support_thresholds: dict[int, float] | None = None,
        fault_support_stats: dict[int, dict[str, np.ndarray | float | list[float]]] | None = None,
    ):
        self.classifier_pipeline = classifier_pipeline
        self.normality_detector = normality_detector
        self.anomaly_detector = normality_detector
        self.normal_label = int(normal_label)
        self.unknown_label = int(unknown_label)
        self.fault_confidence_threshold = (
            None if fault_confidence_threshold is None else float(fault_confidence_threshold)
        )
        self.per_class_fault_confidence_thresholds = (
            {}
            if per_class_fault_confidence_thresholds is None
            else {int(label): float(threshold) for label, threshold in per_class_fault_confidence_thresholds.items()}
        )
        self.fault_support_threshold = None if fault_support_threshold is None else float(fault_support_threshold)
        self.per_class_fault_support_thresholds = (
            {}
            if per_class_fault_support_thresholds is None
            else {int(label): float(threshold) for label, threshold in per_class_fault_support_thresholds.items()}
        )
        self.fault_support_stats = self._normalize_support_stats(fault_support_stats)

    @staticmethod
    def _normalize_support_stats(
        raw_stats: dict[int, dict[str, np.ndarray | float | list[float]]] | None,
    ) -> dict[int, dict[str, np.ndarray]]:
        if not raw_stats:
            return {}

        normalized: dict[int, dict[str, np.ndarray]] = {}
        for raw_label, stats in raw_stats.items():
            try:
                label = int(raw_label)
            except (TypeError, ValueError):
                continue
            if not isinstance(stats, dict):
                continue
            mean = stats.get("mean")
            var = stats.get("var")
            if mean is None or var is None:
                continue
            mean_arr = np.asarray(mean, dtype=np.float32).reshape(-1)
            var_arr = np.asarray(var, dtype=np.float32).reshape(-1)
            if mean_arr.size == 0 or mean_arr.shape != var_arr.shape:
                continue
            normalized[label] = {
                "mean": mean_arr,
                "var": np.maximum(var_arr, 1e-6),
            }
        return normalized

    def _compute_support_distance(
        self,
        features: np.ndarray,
        predicted_labels: np.ndarray,
    ) -> np.ndarray | None:
        if not self.fault_support_stats:
            return None

        feats = np.asarray(features, dtype=np.float32)
        preds = np.asarray(predicted_labels, dtype=np.int64).reshape(-1)
        if feats.ndim != 2 or feats.shape[0] != preds.shape[0]:
            return None

        support_distance = np.full(preds.shape, fill_value=np.nan, dtype=np.float32)
        for label, stats in self.fault_support_stats.items():
            class_mask = preds == int(label)
            if not np.any(class_mask):
                continue
            mean = stats["mean"]
            var = stats["var"]
            if feats.shape[1] != mean.shape[0]:
                continue
            delta = feats[class_mask] - mean.reshape(1, -1)
            support_distance[class_mask] = np.sqrt(np.sum((delta * delta) / var.reshape(1, -1), axis=1))
        return support_distance

    def predict(self, raw_input: list[RawInput]) -> np.ndarray:
        preds, _ = self.predict_with_confidence(raw_input)
        return preds

    def predict_with_confidence(self, raw_input: list[RawInput]) -> tuple[np.ndarray, np.ndarray]:
        samples = list(raw_input)
        gate_preds, gate_conf = self.normality_detector.predict_with_confidence(samples)
        gate_preds = np.asarray(gate_preds, dtype=np.int64).reshape(-1)
        gate_conf = np.asarray(gate_conf, dtype=float).reshape(-1)

        final_preds = np.full(gate_preds.shape, fill_value=self.normal_label, dtype=np.int64)
        final_conf = gate_conf.copy()

        abnormal_indices = np.flatnonzero(gate_preds == 1)
        if abnormal_indices.size == 0:
            return final_preds, final_conf

        abnormal_inputs = [samples[idx] for idx in abnormal_indices.tolist()]
        fault_details = self.classifier_pipeline.predict_details(abnormal_inputs)
        fault_preds = np.asarray(fault_details["predictions"], dtype=np.int64).reshape(-1)
        fault_conf = np.asarray(fault_details["confidence"], dtype=float).reshape(-1)

        final_preds[abnormal_indices] = fault_preds
        final_conf[abnormal_indices] = fault_conf

        if self.fault_confidence_threshold is not None:
            low_conf = np.isnan(fault_conf) | (fault_conf < self.fault_confidence_threshold)
            rejected_indices = abnormal_indices[low_conf]
            final_preds[rejected_indices] = self.unknown_label
            final_conf[rejected_indices] = np.nan_to_num(fault_conf[low_conf], nan=0.0)

        if self.per_class_fault_confidence_thresholds:
            for label, threshold in self.per_class_fault_confidence_thresholds.items():
                class_mask = fault_preds == int(label)
                if not np.any(class_mask):
                    continue
                class_low_conf = class_mask & (np.isnan(fault_conf) | (fault_conf < float(threshold)))
                if not np.any(class_low_conf):
                    continue
                rejected_indices = abnormal_indices[class_low_conf]
                final_preds[rejected_indices] = self.unknown_label
                final_conf[rejected_indices] = np.nan_to_num(fault_conf[class_low_conf], nan=0.0)

        support_distance = self._compute_support_distance(
            fault_details.get("features", np.empty((0, 0), dtype=np.float32)),
            fault_preds,
        )
        if support_distance is not None:
            if self.fault_support_threshold is not None:
                support_reject = np.isnan(support_distance) | (support_distance > self.fault_support_threshold)
                rejected_indices = abnormal_indices[support_reject]
                final_preds[rejected_indices] = self.unknown_label
                final_conf[rejected_indices] = np.nan_to_num(fault_conf[support_reject], nan=0.0)

            if self.per_class_fault_support_thresholds:
                for label, threshold in self.per_class_fault_support_thresholds.items():
                    class_mask = fault_preds == int(label)
                    if not np.any(class_mask):
                        continue
                    class_reject = class_mask & (
                        np.isnan(support_distance) | (support_distance > float(threshold))
                    )
                    if not np.any(class_reject):
                        continue
                    rejected_indices = abnormal_indices[class_reject]
                    final_preds[rejected_indices] = self.unknown_label
                    final_conf[rejected_indices] = np.nan_to_num(fault_conf[class_reject], nan=0.0)

        return final_preds, final_conf
