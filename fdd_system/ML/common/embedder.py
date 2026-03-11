# External libraries
from abc import abstractmethod
import numpy as np
from typing import Any, Mapping
from scipy.signal import welch, detrend, find_peaks, spectrogram, butter, sosfiltfilt, hilbert
from scipy.stats import skew, kurtosis
from scipy.integrate import trapezoid
import matplotlib.pyplot as plt

# Internal imports
from fdd_system.ML.common.config import FanConfig, RawAccWindow, RawInput, SensorConfig

class Embedder():
    """Embedder is the component that "translates" raw inputs into formats that ML/DL understands."""
    
    @abstractmethod
    def embed(self, data: list) -> np.ndarray:
        """Do feature extraction of the given sequence of data.
        
        Args:
            data: a list of input ordered by time sent by the sensor 
            
        Returns:
            a list of feature representation for each input.
        """
        pass

class MLEmbedder1(Embedder):
    """This is an embedder (feature extractor) I have based on my research. Treat
    this as a baseline, let me know if you can perform better with different combinations
    of features."""
    HARMONICS = [1, 2, 3, 4, 8, 16]
    feat_names: list[str] = []    

    # MLEmbedder
    def embed(self, data: list[RawAccWindow]) -> np.ndarray:
        rows = []
        for d in data:
            # extract directional accelerometer
            acc_x = d.acc_x
            acc_y = d.acc_y
            acc_z = d.acc_z

            # Compute magnitude (there're some tradeoffs here,
            #  I lose directional information, but model may generalize better,
            #  because accelerometer is less sensitive to mounting position)
            acc_mag = np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)

            feats = self.extract_features_from_acc(acc_mag, 
                                                        SensorConfig.SAMPLING_RATE, 
                                                        SensorConfig.WINDOW_SIZE, 
                                                        SensorConfig.STRIDE,
                                                        FanConfig.NUM_BLADES)
            
            if not self.feat_names:
                self.feat_names = list(feats.keys())
            
            feat_vec = np.array([feats[name] for name in self.feat_names], dtype=float)
            rows.append(feat_vec)

        return np.vstack(rows)
    
    def extract_features_from_acc(self, acc: np.ndarray, sampling_rate: int, window_size: int, stride: int, num_blades: int) -> dict[str, Any]:
        """Extract features from accelerometer time-series magnitude."""

        if acc.size == 0:
            return {}

        time_domain_feats = self.extract_time_domain_features(acc, sampling_rate)
        freq_domain_feats = self.extract_freq_domain_features(acc, sampling_rate, window_size, stride, num_blades)
         
        return time_domain_feats | freq_domain_feats

    
    def half_power_damping(self, f: np.ndarray, pxx: np.ndarray) -> dict[str,float]:
        m = f >= 2.0
        if not np.any(m): return {'zeta':0.0,'fn':0.0,'Q':0.0}
        fs, ps = f[m], pxx[m]
        peaks, _ = find_peaks(ps)
        if peaks.size == 0: return {'zeta':0.0,'fn':0.0,'Q':0.0}
        k = peaks[np.argmax(ps[peaks])]
        fn = float(fs[k]); pk = float(ps[k])
        if pk <= 0: return {'zeta':0.0,'fn':fn,'Q':0.0}
        half = pk/2.0
        i=k
        while i>0 and ps[i]>half: i-=1
        j=k
        while j<len(ps)-1 and ps[j]>half: j+=1
        fl = float(fs[i]) if i>0 else fs[0]
        fr = float(fs[j]) if j<len(fs) else fs[-1]
        bw = max(1e-6, fr-fl)
        zeta = 0.5*bw/max(1e-6, fn)
        Q = fn/bw if bw>0 else 0.0
        return {'zeta':float(zeta),'fn':fn,'Q':float(Q)}

    def spectral_entropy(self, pxx: np.ndarray) -> float:
        """Compute spectral entropy from power density.

        Args:
            pxx: power density.

        """
        p = np.clip(pxx, 0, None)
        s = p.sum()
        if s <= 0: return 0.0
        p = p/s
        H = -(p*np.log(p + 1e-12)).sum()
        return float(H/np.log(len(p)))


    def est_f1(self, f: np.ndarray, pxx: np.ndarray) -> float:
        """Estimate fan's spinning speed a.k.a shaft frequency.
        
        Args:
            f: frequency sample
            pxx: power densitiy
        Returns
            The estimated shaft frequency (fan's rotational speed).
        """
        # 10 Hz: 600 RPM
        # 100 Hz: 3000 RPM
        # 600 - 3000 RPM are typical range for CPU fan
        min_f = 10
        max_f = 100

        m = (f >= min_f) & (f <= max_f)

        fs, ps = f[m], pxx[m]
        peaks, _ = find_peaks(ps)
        if peaks.size == 0: return float(fs[np.argmax(ps)])
        pk = peaks[np.argmax(ps[peaks])]
        return float(fs[pk])    

    def extract_time_domain_features(self, acc: np.ndarray, sampling_rate: int):
        rms = np.sqrt(np.mean(acc**2))
        
        # compute derivatives of acc with time
        dx = np.diff(acc, prepend=acc[0]) * sampling_rate
        ddx = np.diff(dx, prepend=dx[0]) * sampling_rate

        
        feats = {
            "acc_mean": np.mean(acc),
            "acc_std": np.std(acc),
            "acc_rms": rms,
            "acc_p2p": np.max(acc) - np.min(acc),
            "acc_crest": np.max(np.abs(acc))/rms if rms > 0 else 0.0,
            "acc_skew": skew(acc),
            "acc_kurtosis": kurtosis(acc),
            "hjorth_mobility": np.sqrt(np.var(dx)/(np.var(acc)+1e-12)),
            "hjorth_mobility_": np.sqrt(np.var(ddx)/(np.var(dx)+1e-12)),
            "jerk_rms"       : np.sqrt(np.mean(dx**2)),
        }

        feats["hjorth_complexity"] = feats["hjorth_mobility_"] / (feats["hjorth_mobility"] + 1e-12)

        return feats


    def band_power(self, f: np.ndarray, pxx: np.ndarray, fc: float, bw: float) -> float:
        """Estimate given frequency energy"""
        if fc <= 0 or bw <= 0: return 0.0
        lo, hi = fc - bw, fc + bw
        m = (f >= max(1e-6, lo)) & (f <= hi)
        return float(trapezoid(pxx[m], x=f[m])) if np.any(m) else 0.0

    def extract_freq_domain_features(self, acc: np.ndarray, sampling_rate: int, window_size: int, stride: int, num_blades: int):
        feats = {}

        # Pick an FFT size dynamically so larger window sizes are supported.
        # Use next power-of-two >= window_size for efficiency and to satisfy nfft >= nperseg.
        nfft = 1 << (int(window_size) - 1).bit_length()

        # Increase spectral averaging smoothness with higher segment overlap than stream stride.
        noverlap = max(0, min(window_size - 1, int(0.6 * window_size)))

        f, pxx = welch(
            acc,
            fs=sampling_rate,
            nperseg=window_size,
            noverlap=noverlap,
            nfft=nfft,
        )

        E_tot = trapezoid(pxx, x=f) + 1e-12
        f1 = self.est_f1(f, pxx)

        feats.update({
            "rot_f1": f1,
            "spec_entropy": self.spectral_entropy(pxx)
        })

        # ======= harmonics & BPF/sidebands ==================================
        # bandwidth is set to ~15% of f1 to estimate power of certain freqs
        bw_h = max(1.0, 0.15*f1)
        e1 = self.band_power(f, pxx, f1, bw_h)

        feats['E_f1'] = e1
        feats['E_f1_over_total'] = e1/E_tot
        
        total_h = e1
        thd = 0.0
        for h in self.HARMONICS:
            fc = h*f1
            e = self.band_power(f, pxx, fc, bw_h)
            feats[f'E_f{h}'] = e
            feats[f'E_f{h}_over_total'] = e/E_tot
            total_h += e
            if h>=2: thd += e

        feats['harmonics_total'] = total_h
        feats['THD_over_f1'] = float(thd/(e1+1e-12))

        bpf = num_blades * f1
        feats['BPF_hz'] = bpf
        if bpf > 0:
            bw_main = max(1.0, 0.08*bpf)
            E_BPF = self.band_power(f, pxx, bpf, bw_main)
            kmax = 3
            bw_sb = max(1.0, 0.15*f1)
            E_SB = 0.0
            for k in range(1, kmax+1):
                E_SB += self.band_power(f, pxx, bpf - k*f1, bw_sb)
                E_SB += self.band_power(f, pxx, bpf + k*f1, bw_sb)
            feats['E_BPF'] = E_BPF
            feats['E_BPF_over_total'] = E_BPF / E_tot
            feats['E_sidebands'] = E_SB
            feats['E_SB_over_total'] = E_SB / E_tot
            feats['ratio_SB_over_BPF'] = float(E_SB/(E_BPF+1e-12))
        else:
            feats.update({'E_BPF':0.0,'E_BPF_over_total':0.0,'E_sidebands':0.0,'E_SB_over_total':0.0,'ratio_SB_over_BPF':0.0})

        # spectral fractions
        low_m  = (f>=0.5) & (f<0.5*f1)
        mid_m  = (f>=0.5*f1) & (f<3.0*f1)
        high_m = (f>=3.0*f1)
    
        low  = float(trapezoid(pxx[low_m],  x=f[low_m])) if np.any(low_m)  else 0.0
        mid  = float(trapezoid(pxx[mid_m],  x=f[mid_m])) if np.any(mid_m)  else 0.0
        high = float(trapezoid(pxx[high_m], x=f[high_m])) if np.any(high_m) else 0.0
        tot = low+mid+high + 1e-12
        feats['frac_low'] = low/tot; feats['frac_mid'] = mid/tot; feats['frac_high'] = high/tot

        # damping (dominant structural peak)
        d = self.half_power_damping(f, pxx)
        feats['zeta'] = d['zeta']; feats['fn_struct_hz'] = d['fn']; feats['Q_peak'] = d['Q']
        return feats

class MLEmbedder2(Embedder):
    """Richer feature embedder tuned for airflow blockage detection.

    Blends time, frequency, and time-frequency cues across all axes and the
    magnitude channel, adds harmonic/BPF ratios, spectral shape measures, and
    simple envelope/spectrogram dynamics to stay robust to mounting and capture
    turbulent broadband growth seen during blockage. Optional per-device
    baselines provide deviation/z-score style features for installation drift.
    """

    def __init__(
        self,
        *,
        highpass_hz: float | None = 1.0,
        lowpass_hz: float | None = None,
        harmonic_bw_ratio: float = 0.18,
        stft_nperseg: int = 64,
        stft_noverlap: int = 32,
        baseline_stats: Mapping[int | None, Mapping[str, tuple[float, float]]] | None = None,
    ) -> None:
        self.highpass_hz = highpass_hz
        self.lowpass_hz = lowpass_hz
        self.harmonic_bw_ratio = harmonic_bw_ratio
        self.stft_nperseg = stft_nperseg
        self.stft_noverlap = stft_noverlap
        self.baseline_stats = dict(baseline_stats) if baseline_stats else {}
        self._harmonics = (1, 2)
        self._eps = 1e-12
        self.feat_names: list[str] = []
        self._filter_cache: dict[tuple[float, float | None, float | None], Any] = {}

    # ---------------- public API ----------------
    def embed(self, data: list[RawAccWindow]) -> np.ndarray:
        rows: list[np.ndarray] = []

        for d in data:
            feats = self.extract_features_from_window(d)
            if not self.feat_names:
                self.feat_names = list(feats.keys())
            feat_vec = np.array([feats[name] for name in self.feat_names], dtype=float)
            rows.append(feat_vec)

        return np.vstack(rows) if rows else np.empty((0, len(self.feat_names)), dtype=float)

    def feature_names(self) -> list[str]:
        """Return the feature names in the embedding order.

        Note: populated after the first successful embed call.
        """
        return list(self.feat_names)

    # ---------------- feature helpers ----------------
    def extract_features_from_window(self, w: RawAccWindow) -> dict[str, float]:
        fs = float(w.sampling_rate_hz) if w.sampling_rate_hz else float(SensorConfig.SAMPLING_RATE)
        ax = self._prep_axis(w.acc_x, fs)
        ay = self._prep_axis(w.acc_y, fs)
        az = self._prep_axis(w.acc_z, fs)
        acc_mag = w.acc_mag if w.acc_mag is not None else np.sqrt(ax**2 + ay**2 + az**2)

        signals = {"x": ax, "y": ay, "z": az, "mag": acc_mag}

        feats: dict[str, float] = {}
        feats.update(self._time_domain_features(signals, fs))
        feats.update(self._orientation_balance(signals))
        feats.update(self._freq_domain_features(acc_mag, fs, FanConfig.NUM_BLADES))
        feats.update(self._time_frequency_features(acc_mag, fs))

        feats = self._apply_baseline(feats, device_id=w.device_id)
        return feats

    def _prep_axis(self, arr: np.ndarray, fs: float) -> np.ndarray:
        sig = np.asarray(arr, dtype=float)
        if sig.size == 0:
            return sig
        sig = detrend(sig, type="linear")
        sos = self._get_filter(fs)
        if sos is not None:
            try:
                sig = sosfiltfilt(sos, sig)
            except ValueError:
                pass
        return sig

    def _get_filter(self, fs: float):
        key = (fs, self.highpass_hz, self.lowpass_hz)
        if key in self._filter_cache:
            return self._filter_cache[key]

        nyq = 0.5 * fs
        hp = self.highpass_hz if self.highpass_hz and self.highpass_hz > 0 else None
        lp = self.lowpass_hz if self.lowpass_hz and self.lowpass_hz < nyq else None

        if hp is None and lp is None:
            self._filter_cache[key] = None
            return None

        if hp is not None and lp is not None and lp <= hp:
            lp = None

        if hp is not None and lp is not None:
            sos = butter(4, [hp / nyq, lp / nyq], btype="bandpass", output="sos")
        elif hp is not None:
            sos = butter(4, hp / nyq, btype="highpass", output="sos")
        else:
            sos = butter(4, lp / nyq, btype="lowpass", output="sos")

        self._filter_cache[key] = sos
        return sos

    # ---- time domain ----
    def _time_domain_features(self, signals: Mapping[str, np.ndarray], fs: float) -> dict[str, float]:
        feats: dict[str, float] = {}
        for name, sig in signals.items():
            feats.update(self._time_stats(sig, fs, prefix=name))
        return feats

    def _time_stats(self, sig: np.ndarray, fs: float, *, prefix: str) -> dict[str, float]:
        if sig.size == 0:
            return {
                f"{prefix}_mean": 0.0,
                f"{prefix}_variance": 0.0,
                f"{prefix}_rms": 0.0,
                f"{prefix}_p2p": 0.0,
                f"{prefix}_crest": 0.0,
                f"{prefix}_skew": 0.0,
                f"{prefix}_kurtosis": 0.0,
                f"{prefix}_hjorth_mobility": 0.0,
                f"{prefix}_hjorth_complexity": 0.0,
                f"{prefix}_peak_abs": 0.0,
            }

        rms = float(np.sqrt(np.mean(sig**2)))
        var = float(np.var(sig))
        dx = np.diff(sig, prepend=sig[0]) * fs
        ddx = np.diff(dx, prepend=dx[0]) * fs
        mob = float(np.sqrt(np.var(dx) / (var + self._eps)))
        mob_2 = float(np.sqrt(np.var(ddx) / (np.var(dx) + self._eps)))
        comp = float(mob_2 / (mob + self._eps))

        return {
            f"{prefix}_mean": float(np.mean(sig)),
            f"{prefix}_variance": var,
            f"{prefix}_rms": rms,
            f"{prefix}_p2p": float(np.max(sig) - np.min(sig)),
            f"{prefix}_crest": float(np.max(np.abs(sig)) / (rms + self._eps)),
            f"{prefix}_skew": float(np.nan_to_num(skew(sig), nan=0.0, posinf=0.0, neginf=0.0)),
            f"{prefix}_kurtosis": float(np.nan_to_num(kurtosis(sig), nan=0.0, posinf=0.0, neginf=0.0)),
            f"{prefix}_hjorth_mobility": mob,
            f"{prefix}_hjorth_complexity": comp,
            f"{prefix}_peak_abs": float(np.max(np.abs(sig))),
        }

    def _orientation_balance(self, signals: Mapping[str, np.ndarray]) -> dict[str, float]:
        ex = float(np.mean(signals["x"]**2)) if signals["x"].size else 0.0
        ey = float(np.mean(signals["y"]**2)) if signals["y"].size else 0.0
        ez = float(np.mean(signals["z"]**2)) if signals["z"].size else 0.0
        etot = ex + ey + ez + self._eps

        return {
            "axis_energy_frac_x": ex / etot,
            "axis_energy_frac_y": ey / etot,
            "axis_energy_frac_z": ez / etot,
            "axis_energy_balance": (max(ex, ey, ez) - min(ex, ey, ez)) / etot,
            "mag_grms": float(np.sqrt(np.mean(signals["mag"]**2))) if signals["mag"].size else 0.0,
        }

    # ---- frequency domain ----
    def _freq_domain_features(self, acc: np.ndarray, fs: float, num_blades: int) -> dict[str, float]:
        feats: dict[str, float] = {}
        if acc.size == 0:
            return {
                "f1_hz": 0.0,
                "bpf_hz": 0.0,
                "E_total": 0.0,
                "E_f1": 0.0,
                "E_f2": 0.0,
                "ratio_2x_over_1x": 0.0,
                "band_low_frac": 0.0,
                "band_mid_frac": 0.0,
                "band_high_frac": 0.0,
                "spec_centroid": 0.0,
                "spec_spread": 0.0,
                "spec_entropy": 0.0,
                "spec_kurtosis": 0.0,
                "broadband_over_tones": 0.0,
                "env_peak_bpf": 0.0,
                "env_peak_2bpf": 0.0,
                "env_max_peak_freq": 0.0,
            }

        nperseg = min(acc.size, max(32, SensorConfig.WINDOW_SIZE))
        nfft = 1 << (int(nperseg) - 1).bit_length()
        noverlap = min(nperseg - 1, int(0.5 * nperseg))

        f, pxx = welch(acc, fs=fs, nperseg=nperseg, noverlap=noverlap, nfft=nfft)
        pxx = np.maximum(pxx, 0)
        E_tot = trapezoid(pxx, x=f) + self._eps

        f1 = self.est_f1(f, pxx)
        bw = max(0.5, self.harmonic_bw_ratio * max(1.0, f1))

        e1 = self.band_power(f, pxx, f1, bw)
        e2 = self.band_power(f, pxx, 2 * f1, bw)
        feats.update({
            "f1_hz": f1,
            "E_total": E_tot,
            "E_f1": e1,
            "E_f2": e2,
            "ratio_2x_over_1x": float(e2 / (e1 + self._eps)),
        })

        bpf = num_blades * f1
        bw_bpf = max(0.5, 0.12 * max(1.0, bpf))
        e_bpf = self.band_power(f, pxx, bpf, bw_bpf) if bpf > 0 else 0.0
        feats["bpf_hz"] = bpf
        feats["E_bpf"] = e_bpf
        feats["broadband_over_tones"] = float(max(E_tot - (e1 + e2 + e_bpf), 0.0) / (e1 + e2 + e_bpf + self._eps))

        centroid = float(np.sum(f * pxx) / E_tot)
        spread = float(np.sqrt(np.sum(((f - centroid) ** 2) * pxx) / E_tot))
        spec_kurt = float(np.nan_to_num(kurtosis(pxx), nan=0.0, posinf=0.0, neginf=0.0))
        feats["spec_centroid"] = centroid
        feats["spec_spread"] = spread
        feats["spec_entropy"] = self.spectral_entropy(pxx)
        feats["spec_kurtosis"] = spec_kurt

        feats.update(self._band_energy_splits(f, pxx, f1))
        feats.update(self._envelope_features(acc, fs, bpf, bw_bpf))
        return feats

    def _band_energy_splits(self, f: np.ndarray, pxx: np.ndarray, f1: float) -> dict[str, float]:
        nyq = f.max() if f.size else 0.0
        if f1 <= 0:
            edges = (0.25 * nyq, 0.6 * nyq)
        else:
            edges = (0.8 * f1, 3.0 * f1)

        low_m = f < edges[0]
        mid_m = (f >= edges[0]) & (f < edges[1])
        high_m = f >= edges[1]

        low = float(trapezoid(pxx[low_m], x=f[low_m])) if np.any(low_m) else 0.0
        mid = float(trapezoid(pxx[mid_m], x=f[mid_m])) if np.any(mid_m) else 0.0
        high = float(trapezoid(pxx[high_m], x=f[high_m])) if np.any(high_m) else 0.0
        tot = low + mid + high + self._eps

        return {
            "band_low_frac": low / tot,
            "band_mid_frac": mid / tot,
            "band_high_frac": high / tot,
            "high_over_low": float(high / (low + self._eps)),
            "mid_over_low": float(mid / (low + self._eps)),
        }

    def _envelope_features(self, acc: np.ndarray, fs: float, bpf: float, bw_bpf: float) -> dict[str, float]:
        env = np.abs(hilbert(acc))
        if env.size == 0:
            return {"env_peak_bpf": 0.0, "env_peak_2bpf": 0.0, "env_max_peak_freq": 0.0}

        nperseg = min(env.size, max(32, SensorConfig.WINDOW_SIZE // 2))
        nfft = 1 << (int(nperseg) - 1).bit_length()
        noverlap = min(nperseg - 1, int(0.5 * nperseg))

        f_env, pxx_env = welch(env, fs=fs, nperseg=nperseg, noverlap=noverlap, nfft=nfft)
        env_peak_freq = float(f_env[np.argmax(pxx_env)]) if pxx_env.size else 0.0

        e_bpf_env = self.band_power(f_env, pxx_env, bpf, bw_bpf) if bpf > 0 else 0.0
        e_2bpf_env = self.band_power(f_env, pxx_env, 2 * bpf, bw_bpf) if bpf > 0 else 0.0

        return {
            "env_peak_bpf": e_bpf_env,
            "env_peak_2bpf": e_2bpf_env,
            "env_max_peak_freq": env_peak_freq,
        }

    # ---- time–frequency ----
    def _time_frequency_features(self, acc: np.ndarray, fs: float) -> dict[str, float]:
        if acc.size == 0:
            return {
                "tf_low_cv": 0.0,
                "tf_mid_cv": 0.0,
                "tf_high_cv": 0.0,
                "tf_low_slope": 0.0,
                "tf_mid_slope": 0.0,
                "tf_high_slope": 0.0,
                "tf_high_over_low_mean": 0.0,
            }

        nperseg = min(self.stft_nperseg, acc.size)
        if nperseg < 8:
            return {
                "tf_low_cv": 0.0,
                "tf_mid_cv": 0.0,
                "tf_high_cv": 0.0,
                "tf_low_slope": 0.0,
                "tf_mid_slope": 0.0,
                "tf_high_slope": 0.0,
                "tf_high_over_low_mean": 0.0,
            }

        nfft = 1 << (int(nperseg) - 1).bit_length()
        noverlap = min(self.stft_noverlap, nperseg - 1)
        f, t, Sxx = spectrogram(acc, fs=fs, window="hann", nperseg=nperseg, noverlap=noverlap, nfft=nfft, scaling="spectrum")
        if Sxx.size == 0:
            return {
                "tf_low_cv": 0.0,
                "tf_mid_cv": 0.0,
                "tf_high_cv": 0.0,
                "tf_low_slope": 0.0,
                "tf_mid_slope": 0.0,
                "tf_high_slope": 0.0,
                "tf_high_over_low_mean": 0.0,
            }

        nyq = 0.5 * fs
        bands = {
            "low": (0.0, 0.25 * nyq),
            "mid": (0.25 * nyq, 0.55 * nyq),
            "high": (0.55 * nyq, nyq),
        }

        feats: dict[str, float] = {}
        band_energies: dict[str, np.ndarray] = {}
        for name, (lo, hi) in bands.items():
            m = (f >= lo) & (f < hi)
            if not np.any(m):
                band_energies[name] = np.zeros_like(t)
                continue
            band_energies[name] = trapezoid(Sxx[m, :], x=f[m], axis=0)

        for name, energy in band_energies.items():
            mean_e = float(np.mean(energy)) if energy.size else 0.0
            std_e = float(np.std(energy)) if energy.size else 0.0
            slope = float((energy[-1] - energy[0]) / (len(energy) + self._eps)) if energy.size > 1 else 0.0
            feats[f"tf_{name}_cv"] = float(std_e / (mean_e + self._eps))
            feats[f"tf_{name}_slope"] = slope

        feats["tf_high_over_low_mean"] = float((np.mean(band_energies["high"]) + self._eps) / (np.mean(band_energies["low"]) + self._eps))
        return feats

    # ---- utility ----
    def spectral_entropy(self, pxx: np.ndarray) -> float:
        p = np.clip(pxx, 0, None)
        s = p.sum()
        if s <= 0:
            return 0.0
        p = p / s
        H = -(p * np.log(p + 1e-12)).sum()
        return float(H / np.log(len(p)))

    def est_f1(self, f: np.ndarray, pxx: np.ndarray) -> float:
        min_f = 5
        max_f = 100
        m = (f >= min_f) & (f <= max_f)
        fs, ps = f[m], pxx[m]
        if fs.size == 0:
            return 0.0
        peaks, _ = find_peaks(ps)
        if peaks.size == 0:
            return float(fs[np.argmax(ps)])
        pk = peaks[np.argmax(ps[peaks])]
        return float(fs[pk])

    def band_power(self, f: np.ndarray, pxx: np.ndarray, fc: float, bw: float) -> float:
        if fc <= 0 or bw <= 0:
            return 0.0
        lo, hi = fc - bw, fc + bw
        m = (f >= max(1e-6, lo)) & (f <= hi)
        return float(trapezoid(pxx[m], x=f[m])) if np.any(m) else 0.0

    def _apply_baseline(self, feats: dict[str, float], device_id: int | None) -> dict[str, float]:
        baseline = self.baseline_stats.get(device_id) or self.baseline_stats.get(None)
        if not baseline:
            return feats

        for name, stats in baseline.items():
            if name not in feats:
                continue
            mean = float(stats[0]) if isinstance(stats, (tuple, list)) and len(stats) >= 1 else float(stats)
            std = float(stats[1]) if isinstance(stats, (tuple, list)) and len(stats) >= 2 else 0.0
            delta = feats[name] - mean
            feats[f"{name}_delta"] = float(delta)
            feats[f"{name}_z"] = float(delta / (std + self._eps))

        return feats

class Spectrogram2DEmbedder(Embedder):
    """
    Convert RawAccWindow into a 2D spectrogram
    """

    def __init__(
        self,
        nperseg: int = 32, # window length (in samples) used for each STFT segment
        noverlap: int = 16, # number of overlapped samples between consecutive segments
        nfft: int = 64, # FFT size
        fmax: float | None = None, # cutoff frequency
        log_eps: float = 1e-12, # small epsilon to avoid log(0)
    ):
        self.nperseg = nperseg
        self.noverlap = noverlap
        self.nfft = nfft
        self.fmax = fmax
        self.log_eps = log_eps

    def _acc_magnitude(self, window: RawAccWindow) -> np.ndarray:
        ax = window.acc_x.astype(float)
        ay = window.acc_y.astype(float)
        az = window.acc_z.astype(float)
        return np.sqrt(ax**2 + ay**2 + az**2)

    def _compute_spectrogram(self, acc: np.ndarray):
        fs = SensorConfig.SAMPLING_RATE

        # Compute STFT from scipy
        f, t, Sxx = spectrogram(
            acc,
            fs=fs,
            window="hann",
            nperseg=self.nperseg,
            noverlap=self.noverlap,
            nfft=self.nfft,
            scaling="density",
            mode="magnitude",
        )

        # cut off freq settings
        if self.fmax is not None:
            mask = f <= self.fmax
            f = f[mask]
            Sxx = Sxx[mask, :]

        # convert to log-magnitude
        Sxx_log = np.log10(Sxx + self.log_eps)

        # normalize to [0, 1]
        Smin = Sxx_log.min()
        Sxx_log -= Smin
        Smax = Sxx_log.max()
        if Smax > 0:
            Sxx_norm = Sxx_log / Smax
        else:
            Sxx_norm = Sxx_log  # edge case: all zeros

        return Sxx_norm.astype(np.float32), f, t

    def embed(self, data: list[RawAccWindow]) -> np.ndarray:
        """ Returns: Numpy array (N, 1, F, T) """
        specs = []
        for w in data:
            acc = self._acc_magnitude(w)
            spec, f, t = self._compute_spectrogram(acc)
            # Add channel dimension for 2D CNN input (C = 1)
            specs.append(spec[None, :, :])  # shape: (1, F, T)

        return np.stack(specs, axis=0)

    def plot_spectrogram_window(self, window: RawAccWindow):
        """
        Visualize a single RawAccWindow as a spectrogram.
        Useful to manually inspect differences between label=0 and label=1.
        """
        acc = self._acc_magnitude(window)
        spec, f, t = self._compute_spectrogram(acc)

        plt.figure(figsize=(5, 4))
        extent = [t[0], t[-1], f[0], f[-1]]
        plt.imshow(
            spec,
            origin="lower",
            aspect="auto",
            extent=extent,
        )
        plt.xlabel("Time (s)")
        plt.ylabel("Frequency (Hz)")
        plt.title(f"Spectrogram (label={window.label})")
        plt.colorbar(label="Normalized log |S(f,t)|")
        plt.tight_layout()
        plt.show()


class Raw1DCNNEmbedder(Embedder):
    """Build channels-first (N, 3, L) tensors for 1D CNN inference."""

    def __init__(
        self,
        target_len: int = SensorConfig.WINDOW_SIZE,
        mean: list[float] | np.ndarray | None = None,
        std: list[float] | np.ndarray | None = None,
    ) -> None:
        if target_len <= 0:
            raise ValueError("target_len must be positive.")
        self.target_len = int(target_len)

        self.mean = None if mean is None else np.asarray(mean, dtype=np.float32).reshape(3, 1)
        self.std = None if std is None else np.asarray(std, dtype=np.float32).reshape(3, 1)
        if self.std is not None:
            self.std = np.clip(self.std, 1e-6, None)

    def _fix_length(self, arr: np.ndarray) -> np.ndarray:
        x = np.asarray(arr, dtype=np.float32).reshape(-1)
        if x.size >= self.target_len:
            return x[: self.target_len]

        out = np.zeros((self.target_len,), dtype=np.float32)
        out[: x.size] = x
        return out

    def embed(self, data: list[RawAccWindow]) -> np.ndarray:
        rows: list[np.ndarray] = []
        for w in data:
            x = self._fix_length(w.acc_x)
            y = self._fix_length(w.acc_y)
            z = self._fix_length(w.acc_z)
            sample = np.stack([x, y, z], axis=0)
            rows.append(sample)

        if not rows:
            return np.empty((0, 3, self.target_len), dtype=np.float32)

        batch = np.stack(rows, axis=0).astype(np.float32)
        if self.mean is not None and self.std is not None:
            batch = (batch - self.mean[None, :, :]) / self.std[None, :, :]
        return batch
