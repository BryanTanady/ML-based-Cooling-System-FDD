from abc import abstractmethod
import numpy as np

from fdd_system.ML.common.config.data import RawInput, RawAccWindow
from fdd_system.ML.common.config.system import SensorConfig

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


class BasicPreprocessor(Preprocessor):
    """subtract per-axis median to drop DC/gravity bias.

    Rationale: removing the per-axis median recenters each window to the local
    zero-G baseline, reducing mounting/orientation offsets while preserving the
    energy and spectral content used by downstream features.
    """

    def __init__(self) -> None:
        self.fs = SensorConfig.SAMPLING_RATE

    def _clean_axis(self, arr: np.ndarray) -> np.ndarray:
        if arr.size == 0:
            return arr.astype(float)
        arr = arr.astype(float)
        return arr - np.median(arr)

    def _copy_meta(self, source: RawAccWindow, *, acc_x: np.ndarray, acc_y: np.ndarray, acc_z: np.ndarray) -> RawAccWindow:
        return RawAccWindow(
            acc_x=acc_x,
            acc_y=acc_y,
            acc_z=acc_z,
            label=source.label,
            device_id=source.device_id,
        )

    def preprocess(self, raw_inputs: list[RawInput]) -> list[RawAccWindow]:
        cleaned: list[RawAccWindow] = []
        for w in raw_inputs:
            if not isinstance(w, RawAccWindow):
                cleaned.append(w)
                continue

            ax = self._clean_axis(w.acc_x)
            ay = self._clean_axis(w.acc_y)
            az = self._clean_axis(w.acc_z)

            cleaned.append(self._copy_meta(w, acc_x=ax, acc_y=ay, acc_z=az))

        return cleaned
