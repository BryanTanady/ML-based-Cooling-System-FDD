from abc import abstractmethod
import logging
from typing import Mapping
import numpy as np
from scipy.signal import butter, resample, sosfilt, sosfiltfilt

from fdd_system.ML.common.config.data import RawInput, RawAccWindow
from fdd_system.ML.common.config.system import SensorConfig

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


class RobustPreprocessor(Preprocessor):
    """Domain-shift hardened preprocessor for vibration windows.

    Steps (mirrors the design doc request):
      1) Resamples each window onto a uniform time grid at target_fs and enforces
         a fixed window length.
      2) Removes DC/slow drift by subtracting per-axis medians.
      3) Collapses orientation sensitivity by working on the magnitude channel
         (optionally using it as the only channel).
      4) Applies a band-pass (or high/low-pass) filter to keep the fan-relevant band.
      5) Normalizes amplitude per window (optionally anchored to a per-device baseline).
      6) Rejects obvious artifacts (clipping, spikes, heavy jitter).
      7) Tags the resampled sampling rate so STFT-based embedders see a stable axis.
    """

    def __init__(
        self,
        target_fs: float | None = None,
        window_size: int | None = None,
        highpass_hz: float | None = 1.0,
        lowpass_hz: float | None = None,
        spike_zscore: float = 8.0,
        min_valid_ratio: float = 0.9,
        reject_outliers: bool = True,
        magnitude_only: bool = False,
        baseline_rms: Mapping[int | None, float] | None = None,
        normalize_amplitude: bool = True,
        remove_dc: bool = True,
        resample: bool = True,
    ) -> None:
        self.target_fs = float(target_fs) if target_fs is not None else float(SensorConfig.SAMPLING_RATE)
        self.window_size = int(window_size) if window_size is not None else int(SensorConfig.WINDOW_SIZE)
        # Default low-pass just below Nyquist to trim electrical noise.
        self.lowpass_hz = lowpass_hz if lowpass_hz is not None else 0.45 * self.target_fs
        self.highpass_hz = highpass_hz if highpass_hz and highpass_hz > 0 else None
        self.reject_outliers = reject_outliers
        self.min_valid_ratio = min_valid_ratio
        self.magnitude_only = magnitude_only
        self.baseline_rms = dict(baseline_rms) if baseline_rms else {}
        self.normalize_amplitude = normalize_amplitude
        self.remove_dc = remove_dc
        self.resample = resample
        self.spike_zscore = spike_zscore
        self.jitter_cv_tol = 0.2  # allow small jitter in timestamps
        self.fs_tolerance = 0.1   # accept +/-10% sampling-rate drift before dropping
        self._eps = 1e-8
        self._sos = self._design_filter()

    # ---------------- core helpers ----------------
    def _copy_meta(
        self,
        source: RawAccWindow,
        *,
        acc_x: np.ndarray,
        acc_y: np.ndarray,
        acc_z: np.ndarray,
        timestamps: np.ndarray | None,
    ) -> RawAccWindow:
        w = RawAccWindow(
            acc_x=acc_x,
            acc_y=acc_y,
            acc_z=acc_z,
            label=source.label,
            device_id=source.device_id,
            timestamps=timestamps,
            sampling_rate_hz=self.target_fs,
        )
        return w

    def _design_filter(self):
        """Design a butterworth filter in SOS form based on provided cutoffs."""
        nyq = 0.5 * self.target_fs
        hp = self.highpass_hz
        lp = self.lowpass_hz if self.lowpass_hz and self.lowpass_hz < nyq else None

        if hp is not None and hp >= nyq:
            hp = None
        if hp is not None and lp is not None and lp <= hp:
            lp = None  # avoid invalid band; fall back to high-pass only

        if hp is None and lp is None:
            return None

        if hp is not None and lp is not None:
            return butter(4, [hp / nyq, lp / nyq], btype="bandpass", output="sos")
        if hp is not None:
            return butter(4, hp / nyq, btype="highpass", output="sos")
        return butter(4, lp / nyq, btype="lowpass", output="sos")

    def _safe_filter(self, arr: np.ndarray) -> np.ndarray:
        if self._sos is None or arr.size == 0:
            return arr

        try:
            return sosfiltfilt(self._sos, arr)
        except ValueError:
            # Too-short window for filtfilt padding; fall back to causal filtering.
            try:
                return sosfilt(self._sos, arr)
            except ValueError:
                return arr

    def _remove_dc(self, arr: np.ndarray) -> np.ndarray:
        if arr.size == 0:
            return arr
        return arr - np.median(arr)

    def _vector_magnitude(self, ax: np.ndarray, ay: np.ndarray, az: np.ndarray) -> np.ndarray:
        return np.sqrt(ax**2 + ay**2 + az**2)

    def _align_lengths(
        self, ax: np.ndarray, ay: np.ndarray, az: np.ndarray, ts: np.ndarray | None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
        lengths = [ax.size, ay.size, az.size]
        target_len = min(lengths)
        if target_len == 0:
            return ax, ay, az, ts

        if len(set(lengths)) > 1:
            ax = ax[:target_len]
            ay = ay[:target_len]
            az = az[:target_len]
            if ts is not None:
                ts = ts[:target_len]

        return ax, ay, az, ts

    def _resample(
        self,
        ax: np.ndarray,
        ay: np.ndarray,
        az: np.ndarray,
        ts: np.ndarray | None,
    ) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
        """Resample to target_fs and enforce fixed window length."""
        if not self.resample:
            return ax, ay, az, ts

        target_len = self.window_size
        if target_len <= 0:
            return ax, ay, az, ts

        use_ts = ts is not None and ts.size == ax.size and ts.size > 1
        if use_ts:
            ts = np.asarray(ts, dtype=float)
            dt = np.diff(ts)
            if np.any(dt <= 0):
                if self.reject_outliers:
                    return None, None, None, None
                use_ts = False
                ts = None

        if use_ts and ts is not None:
            measured_fs = 1.0 / (np.median(dt) + self._eps)
            rate_err = abs(measured_fs - self.target_fs) / (self.target_fs + self._eps)
            if rate_err > self.fs_tolerance and self.reject_outliers:
                return None, None, None, None

            jitter_cv = np.std(dt) / (np.mean(dt) + self._eps)
            if jitter_cv > self.jitter_cv_tol and self.reject_outliers:
                return None, None, None, None

            uniform_grid = ts[0] + np.arange(target_len) / self.target_fs
            ax = np.interp(uniform_grid, ts, ax)
            ay = np.interp(uniform_grid, ts, ay)
            az = np.interp(uniform_grid, ts, az)
            ts = uniform_grid
            return ax, ay, az, ts

        # Fallback path when timestamps are missing or rejected
        if ax.size < target_len * self.min_valid_ratio and self.reject_outliers:
            return None, None, None, None

        if ax.size != target_len:
            ax = resample(ax, target_len)
            ay = resample(ay, target_len)
            az = resample(az, target_len)

        return ax, ay, az, ts

    def _normalize(
        self,
        ax: np.ndarray,
        ay: np.ndarray,
        az: np.ndarray,
        mag: np.ndarray,
        device_id: int | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if not self.normalize_amplitude:
            return ax, ay, az, mag

        baseline = self.baseline_rms.get(device_id)
        # Per-window RMS falls back when baseline is unavailable
        rms = float(np.sqrt(np.mean(np.square(mag))))
        scale = baseline if baseline is not None else rms
        if not np.isfinite(scale) or scale < self._eps:
            scale = 1.0

        return ax / scale, ay / scale, az / scale, mag / scale

    def _artifact(self, ax: np.ndarray, ay: np.ndarray, az: np.ndarray, mag: np.ndarray) -> bool:
        if not np.all(np.isfinite(mag)):
            return True
        if mag.size < self.window_size * self.min_valid_ratio:
            return True
        if np.max(np.abs(mag)) > 1e6:  # unreasonable saturation
            return True

        # detect spikes relative to robust scale (MAD-based)
        if self.spike_zscore is not None and self.spike_zscore > 0:
            med = np.median(mag)
            mad = np.median(np.abs(mag - med)) + self._eps
            z = 0.6745 * np.abs(mag - med) / mad
            if np.any(z > self.spike_zscore):
                return True

        # clipping: too many samples at the extremes
        spread = np.ptp(mag)
        if spread > self._eps:
            extremes = (mag <= mag.min() + 1e-6 * spread) | (mag >= mag.max() - 1e-6 * spread)
            if extremes.mean() > 0.1:
                return True
        return False

    # ---------------- public API ----------------
    def preprocess(self, raw_inputs: list[RawInput]) -> list[RawAccWindow]:
        cleaned: list[RawAccWindow] = []
        for w in raw_inputs:
            if not isinstance(w, RawAccWindow):
                cleaned.append(w)
                continue

            ax = np.asarray(w.acc_x, dtype=float)
            ay = np.asarray(w.acc_y, dtype=float)
            az = np.asarray(w.acc_z, dtype=float)
            ts = np.asarray(w.timestamps, dtype=float) if w.timestamps is not None else None

            ax, ay, az, ts = self._align_lengths(ax, ay, az, ts)
            ax, ay, az, ts = self._resample(ax, ay, az, ts)
            if ax is None or ay is None or az is None:
                continue

            if self.remove_dc:
                ax = self._remove_dc(ax)
                ay = self._remove_dc(ay)
                az = self._remove_dc(az)

            ax = self._safe_filter(ax)
            ay = self._safe_filter(ay)
            az = self._safe_filter(az)

            mag = self._vector_magnitude(ax, ay, az)

            if self.magnitude_only:
                ax = ay = az = mag.copy()

            if self.reject_outliers and self._artifact(ax, ay, az, mag):
                continue

            ax, ay, az, mag = self._normalize(ax, ay, az, mag, w.device_id)

            out = self._copy_meta(w, acc_x=ax, acc_y=ay, acc_z=az, timestamps=ts)
            # stash magnitude channel for downstream embedders that want an already-robust channel
            setattr(out, "acc_mag", mag)
            cleaned.append(out)

        return cleaned
