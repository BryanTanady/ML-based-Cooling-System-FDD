from abc import abstractmethod
from typing import Protocol, TYPE_CHECKING
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

if TYPE_CHECKING:  # Optional dependencies for type checkers only
    import onnxruntime as ort


class Inferrer():
    """Inferrer represents the actual model (ML/DL) that is trainable and runs the inference."""
    def __init__(self, model):
        self.model = model

    @abstractmethod
    def infer(self, embeddings: np.ndarray) -> np.ndarray:
        """Return predicted labels for each embedding row."""
        pass

    def infer_with_confidence(self, embeddings: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return predicted labels and confidence scores per sample.

        Default implementation uses infer() and returns NaN confidences.
        Subclasses override to provide model-specific confidences.
        """
        preds = self.infer(embeddings)
        conf = np.full_like(preds, fill_value=np.nan, dtype=float)
        return preds, conf


class ClassifierWithPredict(Protocol):
    def predict(self, X: np.ndarray) -> np.ndarray: ...

from sklearn.preprocessing import StandardScaler

class SklearnMLInferrer(Inferrer):
    """Draft version of ML inferrer. This is basically just a wrapper of classical ML inferrer"""

    def infer(self, embeddings: np.ndarray) -> np.ndarray:
        # Accept any sklearn-style estimator (including Pipeline) that exposes predict.
        if not (
            isinstance(self.model, (ClassifierMixin, BaseEstimator))
            or hasattr(self.model, "predict")
        ):
            raise TypeError("Expected an sklearn-style model with predict(X).")

        self.model: ClassifierWithPredict
        return np.asarray(self.model.predict(embeddings))

    def infer_with_confidence(self, embeddings: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        preds = self.infer(embeddings)

        if hasattr(self.model, "predict_proba"):
            probs = np.asarray(self.model.predict_proba(embeddings))
            if probs.ndim == 2 and probs.shape[1] > 1:
                conf = probs.max(axis=1)
            else:
                conf = probs.reshape(-1)
        elif hasattr(self.model, "decision_function"):
            margins = np.asarray(self.model.decision_function(embeddings))
            if margins.ndim == 1:
                margins = margins.reshape(-1, 1)
            # Convert margins to pseudo-probabilities
            if margins.shape[1] == 1:
                pos = 1 / (1 + np.exp(-margins[:, 0]))
                conf = np.maximum(pos, 1 - pos)
            else:
                expm = np.exp(margins - margins.max(axis=1, keepdims=True))
                probs = expm / expm.sum(axis=1, keepdims=True)
                conf = probs.max(axis=1)
        else:
            conf = np.full_like(preds, fill_value=np.nan, dtype=float)

        return preds, np.asarray(conf, dtype=float)


class OnnxInferrer(Inferrer):
    """Inferrer wrapper for ONNX Runtime models."""

    def __init__(self, session: "ort.InferenceSession"):
        import importlib

        ort_spec = importlib.util.find_spec("onnxruntime")
        if ort_spec is None:
            raise ImportError("OnnxInferrer requires onnxruntime to be installed.")

        ort = importlib.import_module("onnxruntime")

        if not isinstance(session, ort.InferenceSession):
            raise TypeError("the model is not onnxruntime.InferenceSession, this inferrer expects ONNX model!!!")

        super().__init__(session)
        self._ort = ort
        self.session = session
        self.input_name = self.session.get_inputs()[0].name

    def infer(self, embeddings: np.ndarray) -> np.ndarray:
        # Ensure batch dimension
        if embeddings.ndim == 1:
            embeddings = np.expand_dims(embeddings, axis=0)

        outputs = self.session.run(None, {self.input_name: embeddings.astype(np.float32)})
        logits = outputs[0]

        if logits.ndim == 2 and logits.shape[1] > 1:
            preds = np.argmax(logits, axis=1)
        else:
            preds = (logits.reshape(-1) > 0).astype(int)

        return preds

    def infer_with_confidence(self, embeddings: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # Ensure batch dimension
        if embeddings.ndim == 1:
            embeddings = np.expand_dims(embeddings, axis=0)

        outputs = self.session.run(None, {self.input_name: embeddings.astype(np.float32)})
        logits = outputs[0]

        if logits.ndim == 2 and logits.shape[1] > 1:
            # softmax for multi-class
            expm = np.exp(logits - logits.max(axis=1, keepdims=True))
            probs = expm / expm.sum(axis=1, keepdims=True)
            preds = np.argmax(probs, axis=1)
            conf = probs.max(axis=1)
        else:
            logits = logits.reshape(-1)
            prob = 1 / (1 + np.exp(-logits))
            preds = (prob > 0.5).astype(int)
            conf = np.maximum(prob, 1 - prob)

        return preds, np.asarray(conf, dtype=float)
