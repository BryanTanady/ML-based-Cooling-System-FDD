from fdd_system.AI.common.architecture.classification.embedder import *
from fdd_system.AI.common.architecture.classification.inferrer import *
from fdd_system.AI.common.architecture.classification.preprocessor import *

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
