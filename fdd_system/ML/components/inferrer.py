from abc import abstractmethod
import logging
import os
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


class TorchInferrer(Inferrer):
    """Inferrer wrapper for torch.nn.Module classifiers."""

    def __init__(self, model):
        import importlib

        torch_spec = importlib.util.find_spec("torch")
        if torch_spec is None:
            raise ImportError("TorchInferrer requires torch to be installed.")

        torch = importlib.import_module("torch")
        if not isinstance(model, torch.nn.Module):
            raise TypeError("TorchInferrer expects a torch.nn.Module model.")

        super().__init__(model)
        self._torch = torch
        self._log = logging.getLogger(__name__)
        self.model = model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._device_type = self.device.type
        jit_flag = os.getenv("FDD_TORCH_JIT_OPTIMIZE", "1").strip().lower()
        self._jit_optimize_enabled = jit_flag not in {"0", "false", "no", "off"}
        self._jit_optimized = False
        self._jit_opt_attempted = False
        self.model.to(self.device)
        self.model.eval()

    def _try_jit_optimize(self, example_batch) -> None:
        if self._jit_opt_attempted or not self._jit_optimize_enabled or self._device_type != "cpu":
            return

        self._jit_opt_attempted = True
        try:
            if isinstance(self.model, self._torch.jit.ScriptModule):
                optimized = self._torch.jit.optimize_for_inference(self.model)
            else:
                example = example_batch[:1].detach().cpu().contiguous()
                traced = self._torch.jit.trace(self.model.cpu().eval(), example, strict=False)
                optimized = self._torch.jit.optimize_for_inference(traced)
            optimized.eval()
            self.model = optimized
            self._jit_optimized = True
            self._log.info("TorchInferrer enabled JIT optimize_for_inference on CPU.")
        except Exception as exc:  # pragma: no cover - depends on model ops/runtime support
            self._log.warning("TorchInferrer JIT optimization skipped: %s", exc)

    def _forward_logits(self, embeddings: np.ndarray):
        arr = np.asarray(embeddings, dtype=np.float32)
        if arr.ndim == 1:
            arr = np.expand_dims(arr, axis=0)

        with self._torch.inference_mode():
            xb = self._torch.from_numpy(arr)
            if self._device_type == "cpu":
                self._try_jit_optimize(xb)
            else:
                xb = xb.to(self.device, non_blocking=True)
            logits = self.model(xb)
            if isinstance(logits, (tuple, list)):
                logits = logits[0]
            return logits

    def infer(self, embeddings: np.ndarray) -> np.ndarray:
        logits = self._forward_logits(embeddings)
        if logits.ndim == 2 and logits.shape[1] > 1:
            preds = self._torch.argmax(logits, dim=1)
        else:
            preds = (logits.reshape(-1) > 0).to(dtype=self._torch.int64)
        return np.asarray(preds.detach().cpu().numpy(), dtype=np.int64)

    def infer_with_confidence(self, embeddings: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        logits = self._forward_logits(embeddings)
        if logits.ndim == 2 and logits.shape[1] > 1:
            probs = self._torch.softmax(logits, dim=1)
            conf, preds = self._torch.max(probs, dim=1)
        else:
            prob = self._torch.sigmoid(logits.reshape(-1))
            preds = (prob > 0.5).to(dtype=self._torch.int64)
            conf = self._torch.maximum(prob, 1.0 - prob)

        return (
            np.asarray(preds.detach().cpu().numpy(), dtype=np.int64),
            np.asarray(conf.detach().cpu().numpy(), dtype=np.float32),
        )
