from abc import abstractmethod
import logging
import numpy as np
from fdd_system.ML.common.config import RawAccWindow, RawInput, SensorConfig

log = logging.getLogger(__name__)

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


class MedianRemoval(Preprocessor):
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


class StandardZNormal(Preprocessor):
    """Simplified preprocessor that only normalizes each axis by its standard deviation."""

    def __init__(self, eps: float = 1e-8) -> None:
        self._eps = eps

    def _copy_meta(
        self,
        source: RawAccWindow,
        *,
        acc_x: np.ndarray,
        acc_y: np.ndarray,
        acc_z: np.ndarray,
    ) -> RawAccWindow:
        return RawAccWindow(
            acc_x=acc_x,
            acc_y=acc_y,
            acc_z=acc_z,
            label=source.label,
            device_id=source.device_id,
            timestamps=source.timestamps,
            sampling_rate_hz=source.sampling_rate_hz or SensorConfig.SAMPLING_RATE,
        )

    def _align_lengths(self, ax: np.ndarray, ay: np.ndarray, az: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        lengths = [ax.size, ay.size, az.size]
        target_len = min(lengths)
        if target_len == 0:
            return ax.astype(float), ay.astype(float), az.astype(float)

        if len(set(lengths)) > 1:
            ax = ax[:target_len]
            ay = ay[:target_len]
            az = az[:target_len]

        return ax.astype(float), ay.astype(float), az.astype(float)

    def _normalize_axis(self, arr: np.ndarray) -> np.ndarray:
        mean = float(np.mean(arr))
        if not np.isfinite(mean):
            mean = 0.0
        arr = arr - mean
        std = float(np.std(arr))
        if not np.isfinite(std) or std < self._eps:
            std = 1.0
        return arr / std

    def preprocess(self, raw_inputs: list[RawInput]) -> list[RawAccWindow]:
        cleaned: list[RawAccWindow] = []
        for w in raw_inputs:
            if not isinstance(w, RawAccWindow):
                cleaned.append(w)
                continue

            ax = np.asarray(w.acc_x)
            ay = np.asarray(w.acc_y)
            az = np.asarray(w.acc_z)

            ax, ay, az = self._align_lengths(ax, ay, az)

            ax = self._normalize_axis(ax)
            ay = self._normalize_axis(ay)
            az = self._normalize_axis(az)

            out = self._copy_meta(w, acc_x=ax, acc_y=ay, acc_z=az)
            mag = np.sqrt(ax**2 + ay**2 + az**2)
            setattr(out, "acc_mag", mag)
            cleaned.append(out)

        return cleaned


class RMSNormalization(Preprocessor):
    """Normalize each axis by the mean magnitude across the window."""

    def __init__(self, eps: float = 1e-8) -> None:
        self._eps = eps

    def _copy_meta(
        self,
        source: RawAccWindow,
        *,
        acc_x: np.ndarray,
        acc_y: np.ndarray,
        acc_z: np.ndarray,
    ) -> RawAccWindow:
        return RawAccWindow(
            acc_x=acc_x,
            acc_y=acc_y,
            acc_z=acc_z,
            label=source.label,
            device_id=source.device_id,
            timestamps=source.timestamps,
            sampling_rate_hz=source.sampling_rate_hz or SensorConfig.SAMPLING_RATE,
        )

    def _align_lengths(self, ax: np.ndarray, ay: np.ndarray, az: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        lengths = [ax.size, ay.size, az.size]
        target_len = min(lengths)
        if target_len == 0:
            return ax.astype(float), ay.astype(float), az.astype(float)

        if len(set(lengths)) > 1:
            ax = ax[:target_len]
            ay = ay[:target_len]
            az = az[:target_len]

        return ax.astype(float), ay.astype(float), az.astype(float)

    def _normalize_axes(self, ax: np.ndarray, ay: np.ndarray, az: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        mag = np.sqrt(ax**2 + ay**2 + az**2)
        denom = float(np.sum(mag) / mag.size) if mag.size else 1.0
        if not np.isfinite(denom) or denom < self._eps:
            denom = 1.0
        return ax / denom, ay / denom, az / denom, mag

    def preprocess(self, raw_inputs: list[RawInput]) -> list[RawAccWindow]:
        cleaned: list[RawAccWindow] = []
        for w in raw_inputs:
            if not isinstance(w, RawAccWindow):
                cleaned.append(w)
                continue

            ax = np.asarray(w.acc_x)
            ay = np.asarray(w.acc_y)
            az = np.asarray(w.acc_z)

            ax, ay, az = self._align_lengths(ax, ay, az)

            ax, ay, az, mag = self._normalize_axes(ax, ay, az)

            out = self._copy_meta(w, acc_x=ax, acc_y=ay, acc_z=az)
            setattr(out, "acc_mag", mag)
            cleaned.append(out)

        return cleaned


# Preserve the old semantic names under the flattened package layout.
BasicPreprocessor = MedianRemoval
RobustPreprocessor = StandardZNormal
