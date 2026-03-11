from fdd_system.ML.common.embedder import Embedder
from fdd_system.ML.common.inferrer import Inferrer
from fdd_system.ML.common.preprocessor import Preprocessor

class ClassificationPipeline:
    """A high level classifier composed of:
        Preprocessor: The component that "guarantees" every inputs are "cleaned" or consistent
        Embedder: The component that translates raw inputs into other forms that are better understood by
            ML/DL models.
        Inferrer: The actual ML/DL model(s) that executes the prediction.
    """
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
