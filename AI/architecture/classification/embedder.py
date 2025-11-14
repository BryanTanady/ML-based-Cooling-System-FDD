# External libraries
from abc import abstractmethod
import numpy as np
from typing import Any
from scipy.signal import welch, detrend, find_peaks
from scipy.stats import skew, kurtosis
from scipy.integrate import trapezoid


# Internal imports
from config.data import *
from config.system import *

from utils.utils import cvt_dict_feats_to_np

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

        feats["hjorth_complexity"] = feats["hjorth_mobility"] / feats["hjorth_mobility_"]

        return feats


    def band_power(self, f: np.ndarray, pxx: np.ndarray, fc: float, bw: float) -> float:
        """Estimate given frequency energy"""
        if fc <= 0 or bw <= 0: return 0.0
        lo, hi = fc - bw, fc + bw
        m = (f >= max(1e-6, lo)) & (f <= hi)
        return float(trapezoid(pxx[m], x=f[m])) if np.any(m) else 0.0

    def extract_freq_domain_features(self, acc: np.ndarray, sampling_rate: int, window_size: int, stride: int, num_blades: int):
        feats = {}

        f, pxx = welch(acc, fs=sampling_rate, nperseg=window_size, noverlap=stride, nfft=512)

        E_tot = trapezoid(pxx, x=f)
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






            


