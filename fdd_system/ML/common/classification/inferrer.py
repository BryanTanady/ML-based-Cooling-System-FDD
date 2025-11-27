from abc import abstractmethod
from typing import Protocol, TYPE_CHECKING
import numpy as np
from sklearn.base import ClassifierMixin

if TYPE_CHECKING:  # Optional dependencies for type checkers only
    import onnxruntime as ort


class Inferrer():
    """Inferrer represents the actual model (ML/DL) that is trainable and runs the inference."""
    def __init__(self, model):
        self.model = model

    @abstractmethod
    def infer(self, embeddings: np.ndarray) -> np.ndarray:
        """Given embeddings (2D numpy array, each row represents feature for one input)
        outputs the respective labels representing the predicted OperatingCondition for each input"""
        pass



class ClassifierWithPredict(Protocol):
    def predict(self, X: np.ndarray) -> np.ndarray: ...

class SklearnMLInferrer(Inferrer):
    """Draft version of ML inferrer. This is basically just a wrapper of classical ML inferrer"""

    def infer(self, embeddings: np.ndarray) -> np.ndarray:
        if not isinstance(self.model, ClassifierMixin):
            raise TypeError("the model is not sklearn model, this inferrer expects sklearn model!!!")

        if not hasattr(self.model, "predict"):
            raise TypeError("Provided model does not implement .predict(X)")
        
        self.model: ClassifierWithPredict
        
        return self.model.predict(embeddings)


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
