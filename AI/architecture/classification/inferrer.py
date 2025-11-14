from abc import abstractmethod
from sklearn.base import ClassifierMixin
from typing import Protocol
import numpy as np


from config.data import RawInput
from architecture.classification.operating_types import OperatingCondition

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




