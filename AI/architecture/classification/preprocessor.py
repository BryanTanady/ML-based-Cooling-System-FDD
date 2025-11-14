from abc import abstractmethod
from config.data import RawInput

class Preprocessor():
    """Preprocessor represents the component that guarantees every inputs
    that comes after the component has some consistency or invariants protected. 
    We can view this as the input cleaner, where everything comes after it can be assumed as "clean".
    Some potential responsibilities of this component may include:
        1. data cleaning (ensure no invalid inputs, such as NAs or unreasonable units eg 1000 m/s^2 in accelerometers).
        2. "Some" Normalization
        3. Validations
    
    The main purpose of this component is to minimize "drift" between different scenario. For instance, we may
    test on different fans with different mountings. Different data "units" or "formats" may confuse the model.
    """
    
    @abstractmethod
    def preprocess(self, raw_inputs: list[RawInput]) -> list:
        pass

class DummyPreprocessor(Preprocessor):
    def preprocess(self, raw_inputs: list[RawInput]) -> list:
        return raw_inputs