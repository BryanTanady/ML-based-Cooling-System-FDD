"""Mahalanobis-based anomaly detection utilities for known/unknown gating."""

from __future__ import annotations

import importlib.util
import pickle
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

try:
    import joblib
except ImportError:  # pragma: no cover - exercised by runtime environment
    joblib = None

from fdd_system.ML.schema import OperatingCondition, RawAccWindow
from fdd_system.ML.components.embedding import Raw1DCNNEmbedder
from fdd_system.ML.components.preprocessing import (
    CenteredRMSNormalization,
    DummyPreprocessor,
    MedianRemoval,
    Preprocessor,
    RMSNormalization,
    StandardZNormal,
)

DEFAULT_SEED = 42
DEFAULT_BATCH_SIZE = 512
DEFAULT_COVARIANCE_REG = 1e-3
DEFAULT_FILE_WINDOW_SCORE_Q = 0.95
DEFAULT_FILE_THRESHOLD_MARGIN = 1.5
DEFAULT_AMBIGUITY_RATIO_THRESHOLD = 0.5
DEFAULT_MAX_PROTOTYPES_PER_CLASS = 6
DEFAULT_MIN_WINDOWS_PER_PROTOTYPE = 30
DEFAULT_MIN_SILHOUETTE_FOR_SPLIT = 0.05
DEFAULT_KMEANS_N_INIT = 10
DEFAULT_STAGE0_RMS_MODE = "raw"
DEFAULT_STAGE0_RMS_LOWER_Q = 0.01
DEFAULT_STAGE0_RMS_UPPER_Q = 0.95
DEFAULT_STAGE0_RMS_LOWER_SCALE = 0.85
DEFAULT_STAGE0_RMS_UPPER_SCALE = 1.15


def _is_window_like(value: object) -> bool:
    return hasattr(value, "acc_x") and hasattr(value, "acc_y") and hasattr(value, "acc_z")


class Stage0WindowGuard:
    """Reject malformed or grossly off-scale windows before model inference."""

    REASON_OK = "ok"
    REASON_WRONG_SHAPE = "wrong_shape"
    REASON_EMPTY_WINDOW = "empty_window"
    REASON_MISMATCHED_AXIS_LENGTHS = "mismatched_axis_lengths"
    REASON_NAN_OR_INF = "nan_or_inf"
    REASON_RMS_TOO_LOW = "rms_too_low"
    REASON_RMS_TOO_HIGH = "rms_too_high"

    def __init__(
        self,
        *,
        expected_len: int,
        rms_lower_bound: float | None,
        rms_upper_bound: float | None,
        calibration_rms_mean: float,
        calibration_rms_std: float,
        calibration_rms_median: float,
        calibration_rms_lower_quantile: float,
        calibration_rms_upper_quantile: float,
        rms_mode: str = DEFAULT_STAGE0_RMS_MODE,
        rms_lower_q: float = DEFAULT_STAGE0_RMS_LOWER_Q,
        rms_upper_q: float = DEFAULT_STAGE0_RMS_UPPER_Q,
        rms_lower_scale: float = DEFAULT_STAGE0_RMS_LOWER_SCALE,
        rms_upper_scale: float = DEFAULT_STAGE0_RMS_UPPER_SCALE,
    ):
        if int(expected_len) <= 0:
            raise ValueError("Stage0WindowGuard requires a positive expected_len.")

        if rms_mode not in {"raw", "centered_window"}:
            raise ValueError(f"Unsupported Stage0WindowGuard rms_mode '{rms_mode}'.")

        lower = None if rms_lower_bound is None else float(rms_lower_bound)
        upper = None if rms_upper_bound is None else float(rms_upper_bound)
        if lower is None and upper is None:
            raise ValueError("Stage0WindowGuard requires at least one RMS bound.")
        if lower is not None and (not np.isfinite(lower) or lower < 0.0):
            raise ValueError("Stage0WindowGuard requires a finite non-negative lower RMS bound.")
        if upper is not None and (not np.isfinite(upper) or upper <= 0.0):
            raise ValueError("Stage0WindowGuard requires a finite positive upper RMS bound.")
        if lower is not None and upper is not None and lower >= upper:
            raise ValueError("Stage0WindowGuard requires RMS bounds with lower < upper.")

        self.expected_len = int(expected_len)
        self.rms_mode = str(rms_mode)
        self.rms_lower_bound = lower
        self.rms_upper_bound = upper
        self.calibration_rms_mean = float(calibration_rms_mean)
        self.calibration_rms_std = float(calibration_rms_std)
        self.calibration_rms_median = float(calibration_rms_median)
        self.calibration_rms_lower_quantile = float(calibration_rms_lower_quantile)
        self.calibration_rms_upper_quantile = float(calibration_rms_upper_quantile)
        self.rms_lower_q = float(rms_lower_q)
        self.rms_upper_q = float(rms_upper_q)
        self.rms_lower_scale = float(rms_lower_scale)
        self.rms_upper_scale = float(rms_upper_scale)

    @staticmethod
    def _window_rms(ax: np.ndarray, ay: np.ndarray, az: np.ndarray, *, mode: str) -> float:
        if mode == "centered_window":
            ax = ax.astype(np.float64) - float(np.mean(ax))
            ay = ay.astype(np.float64) - float(np.mean(ay))
            az = az.astype(np.float64) - float(np.mean(az))
        else:
            ax = ax.astype(np.float64)
            ay = ay.astype(np.float64)
            az = az.astype(np.float64)
        signal_power = ax.astype(np.float64) ** 2 + ay.astype(np.float64) ** 2 + az.astype(np.float64) ** 2
        return float(np.sqrt(np.mean(signal_power)))

    @classmethod
    def fit(
        cls,
        raw_inputs: Sequence[RawAccWindow],
        *,
        expected_len: int | None = None,
        rms_mode: str = DEFAULT_STAGE0_RMS_MODE,
        rms_lower_q: float = DEFAULT_STAGE0_RMS_LOWER_Q,
        rms_upper_q: float = DEFAULT_STAGE0_RMS_UPPER_Q,
        rms_lower_scale: float = DEFAULT_STAGE0_RMS_LOWER_SCALE,
        rms_upper_scale: float = DEFAULT_STAGE0_RMS_UPPER_SCALE,
        fit_lower_bound: bool = True,
        fit_upper_bound: bool = True,
    ) -> "Stage0WindowGuard":
        valid_lengths: list[int] = []
        valid_rms: list[float] = []

        for sample in raw_inputs:
            record = cls.inspect_sample(sample, expected_len=None, rms_mode=rms_mode)
            if record["reason"] != cls.REASON_OK:
                continue
            valid_lengths.append(int(record["axis_lengths"][0]))
            valid_rms.append(float(record["rms"]))

        if not valid_lengths or not valid_rms:
            raise ValueError("Stage0WindowGuard.fit() requires at least one valid calibration window.")

        if expected_len is None:
            lengths_np = np.asarray(valid_lengths, dtype=np.int64)
            unique_lengths, counts = np.unique(lengths_np, return_counts=True)
            expected_len = int(unique_lengths[np.argmax(counts)])

        calibration_rms = np.asarray(
            [rms for rms, length in zip(valid_rms, valid_lengths) if int(length) == int(expected_len)],
            dtype=np.float64,
        )
        if calibration_rms.size == 0:
            raise ValueError("Stage0WindowGuard.fit() found no valid calibration windows for the chosen length.")

        rms_lower_q = float(np.clip(rms_lower_q, 0.0, 1.0))
        rms_upper_q = float(np.clip(rms_upper_q, rms_lower_q, 1.0))
        rms_lower_scale = float(np.clip(rms_lower_scale, 1e-6, 1.0))
        rms_upper_scale = float(max(1.0, rms_upper_scale))

        lower_quantile = float(np.quantile(calibration_rms, rms_lower_q))
        upper_quantile = float(np.quantile(calibration_rms, rms_upper_q))
        rms_lower_bound = max(0.0, lower_quantile * rms_lower_scale) if fit_lower_bound else None
        rms_upper_bound = upper_quantile * rms_upper_scale if fit_upper_bound else None
        if rms_lower_bound is None and rms_upper_bound is None:
            raise ValueError("Stage0WindowGuard.fit() must enable at least one RMS bound.")
        if rms_lower_bound is not None and rms_upper_bound is not None and rms_lower_bound >= rms_upper_bound:
            raise ValueError("Stage0WindowGuard.fit() produced invalid RMS bounds.")

        return cls(
            expected_len=int(expected_len),
            rms_lower_bound=rms_lower_bound,
            rms_upper_bound=rms_upper_bound,
            calibration_rms_mean=float(np.mean(calibration_rms)),
            calibration_rms_std=float(np.std(calibration_rms)),
            calibration_rms_median=float(np.median(calibration_rms)),
            calibration_rms_lower_quantile=lower_quantile,
            calibration_rms_upper_quantile=upper_quantile,
            rms_mode=rms_mode,
            rms_lower_q=rms_lower_q,
            rms_upper_q=rms_upper_q,
            rms_lower_scale=rms_lower_scale,
            rms_upper_scale=rms_upper_scale,
        )

    @classmethod
    def inspect_sample(
        cls,
        sample: object,
        *,
        expected_len: int | None,
        rms_mode: str = DEFAULT_STAGE0_RMS_MODE,
        rms_lower_bound: float | None = None,
        rms_upper_bound: float | None = None,
    ) -> dict[str, Any]:
        axis_lengths = (-1, -1, -1)
        if not _is_window_like(sample):
            return {"accepted": False, "reason": cls.REASON_WRONG_SHAPE, "rms": float("nan"), "axis_lengths": axis_lengths}

        try:
            ax = np.asarray(sample.acc_x)
            ay = np.asarray(sample.acc_y)
            az = np.asarray(sample.acc_z)
        except Exception:
            return {"accepted": False, "reason": cls.REASON_WRONG_SHAPE, "rms": float("nan"), "axis_lengths": axis_lengths}

        axis_lengths = tuple(int(arr.size) for arr in (ax, ay, az))
        if any(arr.ndim != 1 for arr in (ax, ay, az)):
            return {"accepted": False, "reason": cls.REASON_WRONG_SHAPE, "rms": float("nan"), "axis_lengths": axis_lengths}
        if any(length <= 0 for length in axis_lengths):
            return {"accepted": False, "reason": cls.REASON_EMPTY_WINDOW, "rms": float("nan"), "axis_lengths": axis_lengths}
        if len(set(axis_lengths)) != 1:
            return {
                "accepted": False,
                "reason": cls.REASON_MISMATCHED_AXIS_LENGTHS,
                "rms": float("nan"),
                "axis_lengths": axis_lengths,
            }
        if expected_len is not None and axis_lengths[0] != int(expected_len):
            return {"accepted": False, "reason": cls.REASON_WRONG_SHAPE, "rms": float("nan"), "axis_lengths": axis_lengths}
        if not (np.all(np.isfinite(ax)) and np.all(np.isfinite(ay)) and np.all(np.isfinite(az))):
            return {"accepted": False, "reason": cls.REASON_NAN_OR_INF, "rms": float("nan"), "axis_lengths": axis_lengths}

        rms = cls._window_rms(ax, ay, az, mode=rms_mode)
        if not np.isfinite(rms):
            return {"accepted": False, "reason": cls.REASON_NAN_OR_INF, "rms": float("nan"), "axis_lengths": axis_lengths}
        if rms_lower_bound is not None and rms < float(rms_lower_bound):
            return {"accepted": False, "reason": cls.REASON_RMS_TOO_LOW, "rms": rms, "axis_lengths": axis_lengths}
        if rms_upper_bound is not None and rms > float(rms_upper_bound):
            return {"accepted": False, "reason": cls.REASON_RMS_TOO_HIGH, "rms": rms, "axis_lengths": axis_lengths}

        return {"accepted": True, "reason": cls.REASON_OK, "rms": rms, "axis_lengths": axis_lengths}

    def evaluate(self, raw_inputs: Sequence[RawAccWindow]) -> dict[str, Any]:
        accepted_mask: list[bool] = []
        rejection_reason: list[str] = []
        rms_values: list[float] = []
        axis_lengths: list[tuple[int, int, int]] = []

        for sample in raw_inputs:
            record = self.inspect_sample(
                sample,
                expected_len=self.expected_len,
                rms_mode=self.rms_mode,
                rms_lower_bound=self.rms_lower_bound,
                rms_upper_bound=self.rms_upper_bound,
            )
            accepted_mask.append(bool(record["accepted"]))
            rejection_reason.append(str(record["reason"]))
            rms_values.append(float(record["rms"]))
            axis_lengths.append(tuple(int(v) for v in record["axis_lengths"]))

        return {
            "accepted_mask": np.asarray(accepted_mask, dtype=bool),
            "rejected_mask": ~np.asarray(accepted_mask, dtype=bool),
            "rejection_reason": np.asarray(rejection_reason, dtype=object),
            "rms": np.asarray(rms_values, dtype=np.float32),
            "axis_lengths": np.asarray(axis_lengths, dtype=np.int32),
        }

    def export_kwargs(self) -> dict[str, Any]:
        return {
            "expected_len": int(self.expected_len),
            "rms_mode": self.rms_mode,
            "rms_lower_bound": None if self.rms_lower_bound is None else float(self.rms_lower_bound),
            "rms_upper_bound": None if self.rms_upper_bound is None else float(self.rms_upper_bound),
            "calibration_rms_mean": float(self.calibration_rms_mean),
            "calibration_rms_std": float(self.calibration_rms_std),
            "calibration_rms_median": float(self.calibration_rms_median),
            "calibration_rms_lower_quantile": float(self.calibration_rms_lower_quantile),
            "calibration_rms_upper_quantile": float(self.calibration_rms_upper_quantile),
            "rms_lower_q": float(self.rms_lower_q),
            "rms_upper_q": float(self.rms_upper_q),
            "rms_lower_scale": float(self.rms_lower_scale),
            "rms_upper_scale": float(self.rms_upper_scale),
        }

    @classmethod
    def from_kwargs(cls, kwargs: Mapping[str, Any]) -> "Stage0WindowGuard":
        return cls(
            expected_len=int(kwargs["expected_len"]),
            rms_lower_bound=(
                None if kwargs.get("rms_lower_bound") is None else float(kwargs["rms_lower_bound"])
            ),
            rms_upper_bound=(
                None if kwargs.get("rms_upper_bound") is None else float(kwargs["rms_upper_bound"])
            ),
            calibration_rms_mean=float(kwargs["calibration_rms_mean"]),
            calibration_rms_std=float(kwargs["calibration_rms_std"]),
            calibration_rms_median=float(kwargs["calibration_rms_median"]),
            calibration_rms_lower_quantile=float(kwargs["calibration_rms_lower_quantile"]),
            calibration_rms_upper_quantile=float(kwargs["calibration_rms_upper_quantile"]),
            rms_mode=str(kwargs.get("rms_mode", DEFAULT_STAGE0_RMS_MODE)),
            rms_lower_q=float(kwargs.get("rms_lower_q", DEFAULT_STAGE0_RMS_LOWER_Q)),
            rms_upper_q=float(kwargs.get("rms_upper_q", DEFAULT_STAGE0_RMS_UPPER_Q)),
            rms_lower_scale=float(kwargs.get("rms_lower_scale", DEFAULT_STAGE0_RMS_LOWER_SCALE)),
            rms_upper_scale=float(kwargs.get("rms_upper_scale", DEFAULT_STAGE0_RMS_UPPER_SCALE)),
        )


def _require_torch():
    if importlib.util.find_spec("torch") is None:
        raise ImportError("Mahalanobis anomaly detector requires torch to load or run the triplet encoder.")

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset

    return torch, nn, F, DataLoader, TensorDataset


def _build_triplet_cnn(*, in_channels: int = 3, out_dim: int = 16):
    _, nn, _, _, _ = _require_torch()

    class TripletCNN(nn.Module):
        def __init__(self, in_channels: int = 3, out_dim: int = 16):
            super().__init__()
            self.time_branch = nn.Sequential(
                nn.Conv1d(in_channels, 16, kernel_size=5, padding=2),
                nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Conv1d(16, 32, kernel_size=5, padding=2),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
            )
            self.freq_branch = nn.Sequential(
                nn.Conv1d(in_channels, 16, kernel_size=3, dilation=2, padding=2),
                nn.ReLU(),
                nn.Conv1d(16, 32, kernel_size=3, dilation=4, padding=4),
                nn.ReLU(),
                nn.Conv1d(32, 32, kernel_size=3, dilation=8, padding=8),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
            )
            self.proj = nn.Sequential(
                nn.Flatten(),
                nn.Linear(64, out_dim),
            )

        def forward(self, x):
            time_emb = self.time_branch(x)
            torch, _, _, _, _ = _require_torch()
            freq = torch.abs(torch.fft.rfft(x, dim=2))
            freq_emb = self.freq_branch(freq)
            embedding = torch.cat([time_emb, freq_emb], dim=1)
            return self.proj(embedding)

    return TripletCNN(in_channels=in_channels, out_dim=out_dim)


def batch_hard_triplet_loss(embeddings, labels, margin: float = 0.5):
    torch, _, F, _, _ = _require_torch()

    if embeddings.size(0) < 2:
        return torch.tensor(0.0, device=embeddings.device)

    distances = torch.cdist(embeddings, embeddings, p=2)
    same_class = labels.unsqueeze(0) == labels.unsqueeze(1)
    eye = torch.eye(labels.size(0), dtype=torch.bool, device=labels.device)

    positive_mask = same_class & ~eye
    negative_mask = ~same_class

    hardest_positive = torch.where(
        positive_mask,
        distances,
        torch.full_like(distances, -1.0),
    ).max(dim=1).values
    hardest_negative = torch.where(
        negative_mask,
        distances,
        torch.full_like(distances, float("inf")),
    ).min(dim=1).values

    valid = (hardest_positive >= 0) & torch.isfinite(hardest_negative)
    if valid.sum() == 0:
        return torch.tensor(0.0, device=embeddings.device)

    return F.relu(hardest_positive[valid] - hardest_negative[valid] + margin).mean()


def train_triplet_encoder_raw(
    x_train: np.ndarray,
    y_train: np.ndarray,
    *,
    emb_dim: int = 16,
    epochs: int = 40,
    batch_size: int = DEFAULT_BATCH_SIZE,
    lr: float = 1e-3,
    margin: float = 0.5,
    device: str | None = None,
):
    torch, _, _, DataLoader, TensorDataset = _require_torch()

    device_obj = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _build_triplet_cnn(in_channels=x_train.shape[1], out_dim=emb_dim).to(device_obj)
    dataset = TensorDataset(
        torch.from_numpy(np.asarray(x_train, dtype=np.float32)),
        torch.from_numpy(np.asarray(y_train, dtype=np.int64)),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for xb, yb in loader:
            xb = xb.to(device_obj)
            yb = yb.to(device_obj)
            optimizer.zero_grad()
            embeddings = model(xb)
            loss = batch_hard_triplet_loss(embeddings, yb, margin=margin)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * yb.size(0)

        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            avg_loss = total_loss / max(len(loader.dataset), 1)
            print(f"triplet epoch={epoch:02d} loss={avg_loss:.4f}")

    return model.cpu()


def encode_embeddings_raw(
    model,
    x_np: np.ndarray,
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    device: str | None = None,
) -> np.ndarray:
    torch, _, _, _, _ = _require_torch()

    x_np = np.asarray(x_np, dtype=np.float32)
    if x_np.ndim != 3:
        raise ValueError("Triplet encoder expects a 3D array shaped like (batch, channels, length).")

    if x_np.shape[0] == 0:
        out_dim = _infer_encoder_config(model)["out_dim"]
        return np.empty((0, out_dim), dtype=np.float32)

    param = next(iter(model.parameters()), None)
    model_device = param.device if param is not None else torch.device("cpu")
    device_obj = torch.device(device) if device else model_device

    model = model.to(device_obj)
    model.eval()

    embeddings: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(x_np), batch_size):
            xb = torch.from_numpy(x_np[start : start + batch_size]).float().to(device_obj)
            embeddings.append(model(xb).detach().cpu().numpy())

    return np.vstack(embeddings).astype(np.float32)


def _covariance_inverse(class_embeddings: np.ndarray, reg: float = DEFAULT_COVARIANCE_REG) -> np.ndarray:
    feature_dim = class_embeddings.shape[1]
    if len(class_embeddings) < 2:
        cov = np.eye(feature_dim, dtype=np.float32)
    else:
        cov = np.cov(class_embeddings, rowvar=False)
        cov = np.atleast_2d(cov)
    cov = cov + reg * np.eye(feature_dim, dtype=np.float32)
    return np.linalg.pinv(cov).astype(np.float32)


def select_num_prototypes(
    class_embeddings: np.ndarray,
    *,
    max_prototypes_per_class: int = DEFAULT_MAX_PROTOTYPES_PER_CLASS,
    min_windows_per_prototype: int = DEFAULT_MIN_WINDOWS_PER_PROTOTYPE,
    min_silhouette_for_split: float = DEFAULT_MIN_SILHOUETTE_FOR_SPLIT,
    random_state: int = DEFAULT_SEED,
    n_init: int = DEFAULT_KMEANS_N_INIT,
) -> tuple[int, np.ndarray | None, float]:
    max_k = min(max_prototypes_per_class, len(class_embeddings) // min_windows_per_prototype)
    if max_k < 2:
        return 1, None, float("nan")

    best_k = 1
    best_labels = None
    best_score = float("-inf")

    for k in range(2, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=n_init)
        cluster_labels = kmeans.fit_predict(class_embeddings)
        counts = np.bincount(cluster_labels, minlength=k)
        if counts.min() < min_windows_per_prototype:
            continue
        score = silhouette_score(class_embeddings, cluster_labels)
        if score > max(best_score, min_silhouette_for_split):
            best_k = k
            best_labels = cluster_labels
            best_score = float(score)

    if best_k == 1:
        return 1, None, float("nan")

    return best_k, best_labels, best_score


def build_multi_prototype_stats(
    embeddings: np.ndarray,
    labels: np.ndarray,
    *,
    reg: float = DEFAULT_COVARIANCE_REG,
    max_prototypes_per_class: int = DEFAULT_MAX_PROTOTYPES_PER_CLASS,
    min_windows_per_prototype: int = DEFAULT_MIN_WINDOWS_PER_PROTOTYPE,
    min_silhouette_for_split: float = DEFAULT_MIN_SILHOUETTE_FOR_SPLIT,
    random_state: int = DEFAULT_SEED,
    n_init: int = DEFAULT_KMEANS_N_INIT,
) -> tuple[list[dict[str, Any]], dict[int, dict[str, Any]]]:
    prototype_table: list[dict[str, Any]] = []
    class_details: dict[int, dict[str, Any]] = {}

    labels = np.asarray(labels, dtype=np.int64)
    for label in sorted(set(labels.tolist())):
        class_embeddings = embeddings[labels == label]
        if len(class_embeddings) == 0:
            continue

        num_prototypes, cluster_labels, silhouette = select_num_prototypes(
            class_embeddings,
            max_prototypes_per_class=max_prototypes_per_class,
            min_windows_per_prototype=min_windows_per_prototype,
            min_silhouette_for_split=min_silhouette_for_split,
            random_state=random_state,
            n_init=n_init,
        )
        if cluster_labels is None:
            cluster_labels = np.zeros(len(class_embeddings), dtype=np.int64)

        cluster_sizes: list[int] = []
        for prototype_index in range(num_prototypes):
            cluster_embeddings = class_embeddings[cluster_labels == prototype_index]
            cluster_sizes.append(int(len(cluster_embeddings)))
            prototype_table.append(
                {
                    "label": int(label),
                    "prototype_index": int(prototype_index),
                    "mu": cluster_embeddings.mean(axis=0).astype(np.float32),
                    "inv_cov": _covariance_inverse(cluster_embeddings, reg=reg),
                    "num_samples": int(len(cluster_embeddings)),
                }
            )

        class_details[int(label)] = {
            "num_prototypes": int(num_prototypes),
            "cluster_sizes": cluster_sizes,
            "silhouette_score": float(silhouette) if np.isfinite(silhouette) else float("nan"),
        }

    return prototype_table, class_details


def prototype_distance_matrix(
    embeddings: np.ndarray,
    prototype_table: Sequence[Mapping[str, Any]],
) -> tuple[list[int], np.ndarray]:
    if not prototype_table:
        return [], np.empty((len(embeddings), 0), dtype=np.float32)

    owner_labels = [int(entry["label"]) for entry in prototype_table]
    distances = np.zeros((len(embeddings), len(prototype_table)), dtype=np.float32)
    for column, entry in enumerate(prototype_table):
        mu = np.asarray(entry["mu"], dtype=np.float32)
        inv_cov = np.asarray(entry["inv_cov"], dtype=np.float32)
        diff = embeddings - mu
        dist_sq = np.einsum("bi,ij,bj->b", diff, inv_cov, diff)
        distances[:, column] = np.sqrt(np.maximum(dist_sq, 0.0))

    return owner_labels, distances


def multi_prototype_scores(
    embeddings: np.ndarray,
    prototype_table: Sequence[Mapping[str, Any]],
) -> dict[str, np.ndarray]:
    owner_labels, prototype_distances = prototype_distance_matrix(embeddings, prototype_table)
    if not owner_labels:
        n = len(embeddings)
        return {
            "nearest_distance": np.full(n, np.inf, dtype=np.float32),
            "nearest_label": np.full(n, -1, dtype=np.int64),
            "nearest_prototype": np.full(n, -1, dtype=np.int64),
            "second_distance": np.full(n, np.inf, dtype=np.float32),
            "second_label": np.full(n, -1, dtype=np.int64),
            "distance_ratio": np.full(n, np.inf, dtype=np.float32),
            "distance_gap": np.full(n, 0.0, dtype=np.float32),
        }

    nearest_prototype = np.argmin(prototype_distances, axis=1).astype(np.int64)
    unique_labels = sorted(set(owner_labels))
    class_distances = np.full((len(embeddings), len(unique_labels)), np.inf, dtype=np.float32)

    for column, label in enumerate(unique_labels):
        class_columns = [idx for idx, owner_label in enumerate(owner_labels) if owner_label == label]
        class_distances[:, column] = prototype_distances[:, class_columns].min(axis=1)

    class_order = np.argsort(class_distances, axis=1)
    nearest_class_index = class_order[:, 0]
    nearest_distance = class_distances[np.arange(len(embeddings)), nearest_class_index]
    nearest_label = np.array([unique_labels[idx] for idx in nearest_class_index], dtype=np.int64)

    if len(unique_labels) > 1:
        second_class_index = class_order[:, 1]
        second_distance = class_distances[np.arange(len(embeddings)), second_class_index]
        second_label = np.array([unique_labels[idx] for idx in second_class_index], dtype=np.int64)
        distance_ratio = nearest_distance / np.maximum(second_distance, 1e-6)
        distance_gap = second_distance - nearest_distance
    else:
        second_distance = np.full(len(embeddings), np.inf, dtype=np.float32)
        second_label = np.full(len(embeddings), -1, dtype=np.int64)
        distance_ratio = np.zeros(len(embeddings), dtype=np.float32)
        distance_gap = np.full(len(embeddings), np.inf, dtype=np.float32)

    return {
        "nearest_distance": nearest_distance.astype(np.float32),
        "nearest_label": nearest_label,
        "nearest_prototype": nearest_prototype,
        "second_distance": second_distance.astype(np.float32),
        "second_label": second_label,
        "distance_ratio": distance_ratio.astype(np.float32),
        "distance_gap": distance_gap.astype(np.float32),
    }


def distance_to_class_prototypes(
    embeddings: np.ndarray,
    prototype_table: Sequence[Mapping[str, Any]],
    target_label: int,
) -> np.ndarray:
    owner_labels, distances = prototype_distance_matrix(embeddings, prototype_table)
    class_columns = [idx for idx, label in enumerate(owner_labels) if label == int(target_label)]
    if not class_columns:
        return np.full(len(embeddings), np.inf, dtype=np.float32)
    return distances[:, class_columns].min(axis=1)


def summarize_file_distance_score(
    distances: np.ndarray,
    *,
    window_quantile: float = DEFAULT_FILE_WINDOW_SCORE_Q,
) -> float:
    distances = np.asarray(distances, dtype=float)
    distances = distances[np.isfinite(distances)]
    if len(distances) == 0:
        return float("nan")
    return float(np.quantile(distances, window_quantile))


def calibrate_thresholds_from_file_groups(
    encoder,
    scaler: StandardScaler,
    prototype_table: Sequence[Mapping[str, Any]],
    train_groups: Sequence[Mapping[str, Any]],
    val_groups: Sequence[Mapping[str, Any]],
    class_prototype_details: Mapping[int, Mapping[str, Any]],
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    file_window_score_q: float = DEFAULT_FILE_WINDOW_SCORE_Q,
    file_threshold_margin: float = DEFAULT_FILE_THRESHOLD_MARGIN,
) -> tuple[dict[int, dict[str, Any]], float]:
    threshold_details: dict[int, dict[str, Any]] = {}
    all_file_scores: list[float] = []

    for label in sorted(int(lbl) for lbl in class_prototype_details):
        label_file_scores: list[float] = []
        num_train_files = 0
        num_val_files = 0

        for source_name, groups in (("train", train_groups), ("val", val_groups)):
            for group in groups:
                if int(group["label"]) != label:
                    continue

                group_embeddings = encode_embeddings_raw(
                    encoder,
                    np.asarray(group["X"], dtype=np.float32),
                    batch_size=batch_size,
                )
                group_embeddings_scaled = scaler.transform(group_embeddings)
                score_details = multi_prototype_scores(group_embeddings_scaled, prototype_table)
                own_assignment_mask = score_details["nearest_label"] == label
                if np.any(own_assignment_mask):
                    calibration_distances = score_details["nearest_distance"][own_assignment_mask]
                else:
                    calibration_distances = distance_to_class_prototypes(group_embeddings_scaled, prototype_table, label)
                file_score = summarize_file_distance_score(
                    calibration_distances,
                    window_quantile=file_window_score_q,
                )
                if not np.isfinite(file_score):
                    continue

                label_file_scores.append(file_score)
                all_file_scores.append(file_score)
                if source_name == "train":
                    num_train_files += 1
                else:
                    num_val_files += 1

        if not label_file_scores:
            continue

        label_file_scores_np = np.asarray(label_file_scores, dtype=float)
        base_file_score = float(np.max(label_file_scores_np))
        class_detail = class_prototype_details[label]
        threshold_details[label] = {
            "threshold": float(base_file_score * file_threshold_margin),
            "base_file_score": base_file_score,
            "num_train_files": int(num_train_files),
            "num_val_files": int(num_val_files),
            "num_calibration_files": int(len(label_file_scores_np)),
            "file_score_median": float(np.median(label_file_scores_np)),
            "file_score_max": float(np.max(label_file_scores_np)),
            "file_score_quantile": float(file_window_score_q),
            "threshold_margin": float(file_threshold_margin),
            "num_prototypes": int(class_detail["num_prototypes"]),
            "cluster_sizes": [int(size) for size in class_detail["cluster_sizes"]],
            "silhouette_score": float(class_detail["silhouette_score"]),
            "used_fallback": False,
            "calibration_mode": "per-file train+val own-nearest windows",
        }

    if all_file_scores:
        fallback_base_file_score = float(np.max(all_file_scores))
    else:
        fallback_base_file_score = 1.0
    fallback_threshold = float(fallback_base_file_score * file_threshold_margin)

    for label in sorted(int(lbl) for lbl in class_prototype_details):
        if label in threshold_details:
            continue

        class_detail = class_prototype_details[label]
        threshold_details[label] = {
            "threshold": fallback_threshold,
            "base_file_score": fallback_base_file_score,
            "num_train_files": 0,
            "num_val_files": 0,
            "num_calibration_files": 0,
            "file_score_median": fallback_base_file_score,
            "file_score_max": fallback_base_file_score,
            "file_score_quantile": float(file_window_score_q),
            "threshold_margin": float(file_threshold_margin),
            "num_prototypes": int(class_detail["num_prototypes"]),
            "cluster_sizes": [int(size) for size in class_detail["cluster_sizes"]],
            "silhouette_score": float(class_detail["silhouette_score"]),
            "used_fallback": True,
            "calibration_mode": "fallback global file max",
        }

    return threshold_details, fallback_threshold


def fit_mahalanobis_gatekeeper(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    train_feature_groups: Sequence[Mapping[str, Any]],
    val_feature_groups: Sequence[Mapping[str, Any]],
    *,
    emb_dim: int = 16,
    epochs: int = 40,
    batch_size: int = DEFAULT_BATCH_SIZE,
    lr: float = 1e-3,
    margin: float = 0.5,
    reg: float = DEFAULT_COVARIANCE_REG,
    file_window_score_q: float = DEFAULT_FILE_WINDOW_SCORE_Q,
    file_threshold_margin: float = DEFAULT_FILE_THRESHOLD_MARGIN,
    ambiguity_ratio_threshold: float = DEFAULT_AMBIGUITY_RATIO_THRESHOLD,
    max_prototypes_per_class: int = DEFAULT_MAX_PROTOTYPES_PER_CLASS,
    min_windows_per_prototype: int = DEFAULT_MIN_WINDOWS_PER_PROTOTYPE,
    min_silhouette_for_split: float = DEFAULT_MIN_SILHOUETTE_FOR_SPLIT,
    random_state: int = DEFAULT_SEED,
    kmeans_n_init: int = DEFAULT_KMEANS_N_INIT,
    device: str | None = None,
) -> dict[str, Any]:
    del x_val, y_val

    encoder = train_triplet_encoder_raw(
        x_train,
        y_train,
        emb_dim=emb_dim,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        margin=margin,
        device=device,
    )
    z_train = encode_embeddings_raw(encoder, x_train, batch_size=batch_size, device=device)

    scaler = StandardScaler().fit(z_train)
    z_train_scaled = scaler.transform(z_train)

    prototype_table, class_prototype_details = build_multi_prototype_stats(
        z_train_scaled,
        y_train,
        reg=reg,
        max_prototypes_per_class=max_prototypes_per_class,
        min_windows_per_prototype=min_windows_per_prototype,
        min_silhouette_for_split=min_silhouette_for_split,
        random_state=random_state,
        n_init=kmeans_n_init,
    )
    threshold_details, fallback_threshold = calibrate_thresholds_from_file_groups(
        encoder,
        scaler,
        prototype_table,
        train_feature_groups,
        val_feature_groups,
        class_prototype_details,
        batch_size=batch_size,
        file_window_score_q=file_window_score_q,
        file_threshold_margin=file_threshold_margin,
    )
    per_class_thresholds = {label: details["threshold"] for label, details in threshold_details.items()}

    return {
        "encoder": encoder,
        "scaler": scaler,
        "prototype_table": prototype_table,
        "class_prototype_details": class_prototype_details,
        "per_class_thresholds": per_class_thresholds,
        "threshold_details": threshold_details,
        "fallback_threshold": fallback_threshold,
        "ambiguity_ratio_threshold": float(ambiguity_ratio_threshold),
        "batch_size": int(batch_size),
    }


def predict_gatekeeper(
    bundle: Mapping[str, Any],
    x_np: np.ndarray,
    *,
    batch_size: int | None = None,
    device: str | None = None,
) -> dict[str, np.ndarray]:
    batch_size = int(batch_size or bundle.get("batch_size", DEFAULT_BATCH_SIZE))
    embeddings = encode_embeddings_raw(
        bundle["encoder"],
        np.asarray(x_np, dtype=np.float32),
        batch_size=batch_size,
        device=device,
    )
    embeddings_scaled = bundle["scaler"].transform(embeddings)
    score_details = multi_prototype_scores(embeddings_scaled, bundle["prototype_table"])
    nearest_label = score_details["nearest_label"]
    applied_threshold = np.array(
        [
            bundle["per_class_thresholds"].get(int(label), bundle["fallback_threshold"])
            for label in nearest_label
        ],
        dtype=np.float32,
    )
    distance_exceeds_threshold = score_details["nearest_distance"] > applied_threshold
    use_ambiguity = bool(bundle.get("use_ambiguity", True))
    if "use_ambiguity" not in bundle:
        owner_labels = [int(entry["label"]) for entry in bundle.get("prototype_table", [])]
        use_ambiguity = len(set(owner_labels)) > 1

    if use_ambiguity:
        ambiguity_exceeds_threshold = (
            score_details["distance_ratio"] > float(bundle["ambiguity_ratio_threshold"])
        )
        is_unknown = (distance_exceeds_threshold & ambiguity_exceeds_threshold).astype(np.int64)
    else:
        ambiguity_exceeds_threshold = np.zeros_like(distance_exceeds_threshold, dtype=bool)
        is_unknown = distance_exceeds_threshold.astype(np.int64)

    return {
        "embeddings": embeddings_scaled.astype(np.float32),
        "distance": score_details["nearest_distance"],
        "nearest_label": nearest_label,
        "nearest_prototype": score_details["nearest_prototype"],
        "second_distance": score_details["second_distance"],
        "second_label": score_details["second_label"],
        "distance_ratio": score_details["distance_ratio"],
        "distance_gap": score_details["distance_gap"],
        "applied_threshold": applied_threshold,
        "distance_exceeds_threshold": distance_exceeds_threshold.astype(np.int64),
        "ambiguity_exceeds_threshold": ambiguity_exceeds_threshold.astype(np.int64),
        "uses_ambiguity": np.full(len(nearest_label), int(use_ambiguity), dtype=np.int64),
        "is_unknown": is_unknown,
    }


def gate_decision_confidence(
    details: Mapping[str, np.ndarray],
    *,
    ambiguity_ratio_threshold: float,
) -> np.ndarray:
    distance_scale = np.asarray(details["distance"], dtype=np.float32) / np.maximum(
        np.asarray(details["applied_threshold"], dtype=np.float32),
        1e-6,
    )
    ambiguity_scale = np.asarray(details["distance_ratio"], dtype=np.float32) / max(
        float(ambiguity_ratio_threshold),
        1e-6,
    )
    is_unknown = np.asarray(details["is_unknown"], dtype=np.int64).astype(bool)
    use_ambiguity = np.asarray(
        details.get("uses_ambiguity", np.ones(is_unknown.shape, dtype=np.int64)),
        dtype=np.int64,
    ).astype(bool).reshape(-1)
    if use_ambiguity.size != is_unknown.size:
        fallback_use_ambiguity = bool(use_ambiguity[0]) if use_ambiguity.size > 0 else True
        use_ambiguity = np.full(is_unknown.shape, fallback_use_ambiguity, dtype=bool)

    ambiguity_unknown_margin = np.minimum(distance_scale - 1.0, ambiguity_scale - 1.0)
    ambiguity_known_margin = np.minimum(1.0 - distance_scale, 1.0 - ambiguity_scale)
    distance_only_unknown_margin = distance_scale - 1.0
    distance_only_known_margin = 1.0 - distance_scale
    unknown_margin = np.where(use_ambiguity, ambiguity_unknown_margin, distance_only_unknown_margin)
    known_margin = np.where(use_ambiguity, ambiguity_known_margin, distance_only_known_margin)
    decision_margin = np.where(is_unknown, unknown_margin, known_margin)

    # Clamp logits before exp() so highly separated samples saturate cleanly
    # without emitting overflow warnings.
    logits = np.clip(np.asarray(decision_margin, dtype=np.float64) * 4.0, -60.0, 60.0)
    conf = 1.0 / (1.0 + np.exp(-logits))
    conf = np.nan_to_num(conf, nan=0.0, posinf=1.0, neginf=0.0)
    return np.clip(conf.astype(np.float32), 0.0, 1.0)


def _infer_preprocessor_kwargs(preprocessor: Preprocessor | None, kwargs: Mapping[str, Any] | None) -> dict[str, Any]:
    if kwargs:
        return {str(k): v for k, v in kwargs.items()}
    if preprocessor is not None and hasattr(preprocessor, "export_kwargs"):
        exported = preprocessor.export_kwargs()
        if isinstance(exported, Mapping):
            return {str(k): v for k, v in exported.items()}
    return {}


def _make_preprocessor(name: str, kwargs: Mapping[str, Any] | None = None) -> Preprocessor:
    normalized = name.strip().lower()
    init_kwargs = dict(kwargs or {})
    if normalized in {"basic", "median"}:
        return MedianRemoval()
    if normalized == "dummy":
        return DummyPreprocessor()
    if normalized in {"robust", "standard"}:
        return StandardZNormal(**init_kwargs)
    if normalized == "rms":
        return RMSNormalization(**init_kwargs)
    if normalized == "centered_rms":
        return CenteredRMSNormalization(**init_kwargs)
    raise ValueError(f"Unknown anomaly-detector preprocessor '{name}'.")


def _build_scaler_from_artifact(artifact: Mapping[str, Any]) -> StandardScaler:
    if "scaler" in artifact:
        scaler = artifact["scaler"]
        if not isinstance(scaler, StandardScaler):
            raise TypeError("Expected 'scaler' to be sklearn.preprocessing.StandardScaler.")
        return scaler

    mean = artifact.get("scaler_mean")
    scale = artifact.get("scaler_scale")
    if mean is None or scale is None:
        raise ValueError("Anomaly detector artifact is missing scaler statistics.")

    mean_arr = np.asarray(mean, dtype=np.float64)
    scale_arr = np.clip(np.asarray(scale, dtype=np.float64), 1e-6, None)
    var_arr = np.asarray(artifact.get("scaler_var", scale_arr**2), dtype=np.float64)
    scaler = StandardScaler()
    scaler.mean_ = mean_arr
    scaler.scale_ = scale_arr
    scaler.var_ = var_arr
    scaler.n_features_in_ = int(artifact.get("scaler_n_features_in", mean_arr.shape[0]))
    scaler.n_samples_seen_ = int(artifact.get("scaler_n_samples_seen", 1))
    return scaler


def _normalize_prototype_table(prototype_table: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for entry in prototype_table:
        normalized.append(
            {
                "label": int(entry["label"]),
                "prototype_index": int(entry["prototype_index"]),
                "mu": np.asarray(entry["mu"], dtype=np.float32),
                "inv_cov": np.asarray(entry["inv_cov"], dtype=np.float32),
                "num_samples": int(entry["num_samples"]),
            }
        )
    return normalized


def _normalize_nested_scalars(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {int(k) if str(k).isdigit() else k: _normalize_nested_scalars(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_normalize_nested_scalars(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _infer_encoder_config(encoder) -> dict[str, int]:
    time_branch = getattr(encoder, "time_branch", None)
    proj = getattr(encoder, "proj", None)
    if time_branch is None or proj is None:
        raise ValueError("Unsupported triplet encoder; could not infer encoder configuration.")

    in_channels = getattr(time_branch[0], "in_channels", None)
    out_dim = getattr(proj[1], "out_features", None)
    if in_channels is None or out_dim is None:
        raise ValueError("Unsupported triplet encoder; missing expected convolution or projection layers.")

    return {
        "in_channels": int(in_channels),
        "out_dim": int(out_dim),
    }


def serialize_mahalanobis_gatekeeper(
    bundle: Mapping[str, Any],
    *,
    mean: np.ndarray,
    std: np.ndarray,
    window_len: int,
    preprocessor_name: str = "rms",
    preprocessor_kwargs: Mapping[str, Any] | None = None,
    stage0_guard: Stage0WindowGuard | None = None,
    batch_size: int | None = None,
) -> dict[str, Any]:
    encoder = bundle["encoder"]
    scaler = bundle["scaler"]
    if not isinstance(scaler, StandardScaler):
        raise TypeError("Mahalanobis gatekeeper bundle must contain a StandardScaler for serialization.")

    return {
        "artifact_type": "mahalanobis_triplet_gatekeeper",
        "encoder_config": _infer_encoder_config(encoder),
        "encoder_state_dict": encoder.state_dict(),
        "scaler_mean": np.asarray(scaler.mean_, dtype=np.float32),
        "scaler_scale": np.asarray(scaler.scale_, dtype=np.float32),
        "scaler_var": np.asarray(getattr(scaler, "var_", scaler.scale_**2), dtype=np.float32),
        "scaler_n_features_in": int(getattr(scaler, "n_features_in_", scaler.mean_.shape[0])),
        "scaler_n_samples_seen": int(getattr(scaler, "n_samples_seen_", 1)),
        "prototype_table": _normalize_prototype_table(bundle["prototype_table"]),
        "class_prototype_details": _normalize_nested_scalars(bundle.get("class_prototype_details", {})),
        "per_class_thresholds": {
            int(label): float(threshold)
            for label, threshold in bundle["per_class_thresholds"].items()
        },
        "threshold_details": _normalize_nested_scalars(bundle.get("threshold_details", {})),
        "fallback_threshold": float(bundle["fallback_threshold"]),
        "ambiguity_ratio_threshold": float(bundle["ambiguity_ratio_threshold"]),
        "mean": np.asarray(mean, dtype=np.float32),
        "std": np.asarray(std, dtype=np.float32),
        "window_len": int(window_len),
        "preprocessor_name": str(preprocessor_name),
        "preprocessor_kwargs": _normalize_nested_scalars(
            _infer_preprocessor_kwargs(None, preprocessor_kwargs or bundle.get("preprocessor_kwargs"))
        ),
        "stage0_guard_kwargs": None if stage0_guard is None else _normalize_nested_scalars(stage0_guard.export_kwargs()),
        "batch_size": int(batch_size or bundle.get("batch_size", DEFAULT_BATCH_SIZE)),
    }


def save_mahalanobis_gatekeeper(
    path: str | Path,
    bundle: Mapping[str, Any],
    *,
    mean: np.ndarray,
    std: np.ndarray,
    window_len: int,
    preprocessor_name: str = "rms",
    preprocessor_kwargs: Mapping[str, Any] | None = None,
    stage0_guard: Stage0WindowGuard | None = None,
    batch_size: int | None = None,
) -> Path:
    artifact = serialize_mahalanobis_gatekeeper(
        bundle,
        mean=mean,
        std=std,
        window_len=window_len,
        preprocessor_name=preprocessor_name,
        preprocessor_kwargs=preprocessor_kwargs,
        stage0_guard=stage0_guard,
        batch_size=batch_size,
    )
    return save_anomaly_detector_artifact(path, artifact)


def save_anomaly_detector_artifact(path: str | Path, artifact: Mapping[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.suffix.lower() in {".pt", ".pth"}:
        torch, _, _, _, _ = _require_torch()
        torch.save(dict(artifact), path)
        return path

    if joblib is not None:
        joblib.dump(dict(artifact), path)
        return path

    with path.open("wb") as handle:
        pickle.dump(dict(artifact), handle)
    return path


def _load_artifact_file(path: str | Path) -> Any:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Anomaly detector artifact not found: {path}")

    if path.suffix.lower() in {".pt", ".pth"}:
        torch, _, _, _, _ = _require_torch()
        # These artifacts are written by save_anomaly_detector_artifact() via
        # torch.save(dict(...)), so newer PyTorch releases may reject them under
        # the default weights_only=True policy. Prefer the safe load path first,
        # then fall back to the legacy pickle-backed behavior for trusted files.
        try:
            return torch.load(path, map_location="cpu", weights_only=True)
        except pickle.UnpicklingError as exc:
            if "Weights only load failed" not in str(exc):
                raise
            return torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            return torch.load(path, map_location="cpu")

    if joblib is not None:
        return joblib.load(path)

    with path.open("rb") as handle:
        return pickle.load(handle)


def _load_encoder_from_artifact(artifact: Mapping[str, Any]):
    if "encoder" in artifact:
        return artifact["encoder"]

    state_dict = artifact.get("encoder_state_dict")
    if state_dict is None:
        raise ValueError("Anomaly detector artifact is missing an encoder state dict.")

    encoder_config = artifact.get("encoder_config", {})
    encoder = _build_triplet_cnn(
        in_channels=int(encoder_config.get("in_channels", 3)),
        out_dim=int(encoder_config.get("out_dim", encoder_config.get("embedding_dim", 16))),
    )
    encoder.load_state_dict(state_dict)
    return encoder.cpu()


class MahalanobisAnomalyDetector:
    """Known/unknown gate based on a triplet encoder and Mahalanobis prototypes."""

    def __init__(
        self,
        *,
        encoder,
        scaler: StandardScaler,
        prototype_table: Sequence[Mapping[str, Any]],
        per_class_thresholds: Mapping[int, float],
        fallback_threshold: float,
        ambiguity_ratio_threshold: float = DEFAULT_AMBIGUITY_RATIO_THRESHOLD,
        window_len: int,
        mean: np.ndarray | None = None,
        std: np.ndarray | None = None,
        preprocessor_name: str = "rms",
        preprocessor_kwargs: Mapping[str, Any] | None = None,
        preprocessor: Preprocessor | None = None,
        stage0_guard: Stage0WindowGuard | None = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
        threshold_details: Mapping[int, Any] | None = None,
        class_prototype_details: Mapping[int, Any] | None = None,
    ):
        self.encoder = encoder
        self.scaler = scaler
        self.prototype_table = _normalize_prototype_table(prototype_table)
        self.per_class_thresholds = {
            int(label): float(threshold)
            for label, threshold in per_class_thresholds.items()
        }
        self.fallback_threshold = float(fallback_threshold)
        self.ambiguity_ratio_threshold = float(ambiguity_ratio_threshold)
        self.window_len = int(window_len)
        self.mean = None if mean is None else np.asarray(mean, dtype=np.float32)
        self.std = None if std is None else np.asarray(std, dtype=np.float32)
        self.preprocessor_name = str(preprocessor_name)
        self.preprocessor_kwargs = _normalize_nested_scalars(
            _infer_preprocessor_kwargs(preprocessor, preprocessor_kwargs)
        )
        self.preprocessor = preprocessor or _make_preprocessor(self.preprocessor_name, self.preprocessor_kwargs)
        self.stage0_guard = stage0_guard
        self.batch_size = int(batch_size)
        self.threshold_details = _normalize_nested_scalars(threshold_details or {})
        self.class_prototype_details = _normalize_nested_scalars(class_prototype_details or {})
        owner_labels = [int(entry["label"]) for entry in self.prototype_table]
        self.use_ambiguity = len(set(owner_labels)) > 1
        self.raw_embedder = Raw1DCNNEmbedder(
            target_len=self.window_len,
            mean=self.mean,
            std=self.std,
        )
        self.bundle = {
            "encoder": self.encoder,
            "scaler": self.scaler,
            "prototype_table": self.prototype_table,
            "per_class_thresholds": self.per_class_thresholds,
            "fallback_threshold": self.fallback_threshold,
            "ambiguity_ratio_threshold": self.ambiguity_ratio_threshold,
            "use_ambiguity": self.use_ambiguity,
            "batch_size": self.batch_size,
            "preprocessor_kwargs": self.preprocessor_kwargs,
        }

    def predict(self, raw_inputs: Sequence[RawAccWindow]) -> np.ndarray:
        preds, _ = self.predict_with_confidence(raw_inputs)
        return preds

    def predict_with_confidence(
        self,
        raw_inputs: Sequence[RawAccWindow],
    ) -> tuple[np.ndarray, np.ndarray]:
        details = self.predict_details(raw_inputs)
        return details["is_unknown"], details["decision_confidence"]

    def predict_details(self, raw_inputs: Sequence[RawAccWindow]) -> dict[str, np.ndarray]:
        samples = list(raw_inputs)
        feature_dim = int(getattr(self.scaler, "n_features_in_", len(getattr(self.scaler, "mean_", []))))
        if not samples:
            empty = np.empty((0,), dtype=np.float32)
            return {
                "embeddings": np.empty((0, feature_dim), dtype=np.float32),
                "distance": empty.copy(),
                "nearest_label": np.empty((0,), dtype=np.int64),
                "nearest_prototype": np.empty((0,), dtype=np.int64),
                "second_distance": empty.copy(),
                "second_label": np.empty((0,), dtype=np.int64),
                "distance_ratio": empty.copy(),
                "distance_gap": empty.copy(),
                "applied_threshold": empty.copy(),
                "distance_exceeds_threshold": np.empty((0,), dtype=np.int64),
                "ambiguity_exceeds_threshold": np.empty((0,), dtype=np.int64),
                "uses_ambiguity": np.empty((0,), dtype=np.int64),
                "is_unknown": np.empty((0,), dtype=np.int64),
                "decision_confidence": empty.copy(),
                "stage0_valid": np.empty((0,), dtype=np.int64),
                "stage0_reason": np.empty((0,), dtype=object),
                "stage0_rms": empty.copy(),
                "stage0_axis_lengths": np.empty((0, 3), dtype=np.int32),
            }

        if self.stage0_guard is None:
            processed_inputs = self.preprocessor.preprocess(samples)
            x_np = self.raw_embedder.embed(processed_inputs)
            details = predict_gatekeeper(self.bundle, x_np, batch_size=self.batch_size)
            details["decision_confidence"] = gate_decision_confidence(
                details,
                ambiguity_ratio_threshold=self.ambiguity_ratio_threshold,
            )
            details["stage0_valid"] = np.ones(len(samples), dtype=np.int64)
            details["stage0_reason"] = np.full(len(samples), Stage0WindowGuard.REASON_OK, dtype=object)
            details["stage0_rms"] = np.full(len(samples), np.nan, dtype=np.float32)
            details["stage0_axis_lengths"] = np.full((len(samples), 3), -1, dtype=np.int32)
            return details

        stage0 = self.stage0_guard.evaluate(samples)
        accepted_mask = np.asarray(stage0["accepted_mask"], dtype=bool)
        accepted_indices = np.flatnonzero(accepted_mask)

        details = {
            "embeddings": np.full((len(samples), feature_dim), np.nan, dtype=np.float32),
            "distance": np.full(len(samples), np.nan, dtype=np.float32),
            "nearest_label": np.full(len(samples), -1, dtype=np.int64),
            "nearest_prototype": np.full(len(samples), -1, dtype=np.int64),
            "second_distance": np.full(len(samples), np.nan, dtype=np.float32),
            "second_label": np.full(len(samples), -1, dtype=np.int64),
            "distance_ratio": np.full(len(samples), np.nan, dtype=np.float32),
            "distance_gap": np.full(len(samples), np.nan, dtype=np.float32),
            "applied_threshold": np.full(len(samples), np.nan, dtype=np.float32),
            "distance_exceeds_threshold": np.zeros(len(samples), dtype=np.int64),
            "ambiguity_exceeds_threshold": np.zeros(len(samples), dtype=np.int64),
            "uses_ambiguity": np.full(len(samples), int(self.use_ambiguity), dtype=np.int64),
            "is_unknown": np.ones(len(samples), dtype=np.int64),
            "decision_confidence": np.ones(len(samples), dtype=np.float32),
            "stage0_valid": accepted_mask.astype(np.int64),
            "stage0_reason": np.asarray(stage0["rejection_reason"], dtype=object),
            "stage0_rms": np.asarray(stage0["rms"], dtype=np.float32),
            "stage0_axis_lengths": np.asarray(stage0["axis_lengths"], dtype=np.int32),
        }

        if accepted_indices.size == 0:
            return details

        processed_inputs = self.preprocessor.preprocess([samples[idx] for idx in accepted_indices.tolist()])
        x_np = self.raw_embedder.embed(processed_inputs)
        accepted_details = predict_gatekeeper(self.bundle, x_np, batch_size=self.batch_size)
        accepted_conf = gate_decision_confidence(
            accepted_details,
            ambiguity_ratio_threshold=self.ambiguity_ratio_threshold,
        )

        for key in [
            "embeddings",
            "distance",
            "nearest_label",
            "nearest_prototype",
            "second_distance",
            "second_label",
            "distance_ratio",
            "distance_gap",
            "applied_threshold",
            "distance_exceeds_threshold",
            "ambiguity_exceeds_threshold",
            "uses_ambiguity",
            "is_unknown",
        ]:
            details[key][accepted_indices] = accepted_details[key]
        details["decision_confidence"][accepted_indices] = accepted_conf
        return details

    def to_artifact(self) -> dict[str, Any]:
        return {
            **serialize_mahalanobis_gatekeeper(
                {
                    **self.bundle,
                    "threshold_details": self.threshold_details,
                    "class_prototype_details": self.class_prototype_details,
                },
                mean=np.zeros((1, 3, 1), dtype=np.float32) if self.mean is None else self.mean,
                std=np.ones((1, 3, 1), dtype=np.float32) if self.std is None else self.std,
                window_len=self.window_len,
                preprocessor_name=self.preprocessor_name,
                preprocessor_kwargs=self.preprocessor_kwargs,
                stage0_guard=self.stage0_guard,
                batch_size=self.batch_size,
            ),
        }

    def save(self, path: str | Path) -> Path:
        return save_anomaly_detector_artifact(path, self.to_artifact())

    @classmethod
    def from_artifact(cls, artifact: Mapping[str, Any]):
        encoder = _load_encoder_from_artifact(artifact)
        scaler = _build_scaler_from_artifact(artifact)
        window_len = artifact.get("window_len", artifact.get("target_len"))
        if window_len is None:
            raise ValueError("Anomaly detector artifact is missing 'window_len'.")

        preprocessor = artifact.get("preprocessor")
        preprocessor_name = str(artifact.get("preprocessor_name", "rms"))
        preprocessor_kwargs = artifact.get("preprocessor_kwargs")
        stage0_guard_kwargs = artifact.get("stage0_guard_kwargs")
        if preprocessor is not None and not isinstance(preprocessor, Preprocessor):
            raise TypeError("Expected anomaly detector artifact 'preprocessor' to implement Preprocessor.")

        return cls(
            encoder=encoder,
            scaler=scaler,
            prototype_table=artifact["prototype_table"],
            per_class_thresholds=artifact.get("per_class_thresholds", {}),
            fallback_threshold=float(artifact["fallback_threshold"]),
            ambiguity_ratio_threshold=float(
                artifact.get("ambiguity_ratio_threshold", DEFAULT_AMBIGUITY_RATIO_THRESHOLD)
            ),
            window_len=int(window_len),
            mean=artifact.get("mean"),
            std=artifact.get("std"),
            preprocessor_name=preprocessor_name,
            preprocessor_kwargs=preprocessor_kwargs if isinstance(preprocessor_kwargs, Mapping) else None,
            preprocessor=preprocessor,
            stage0_guard=(
                Stage0WindowGuard.from_kwargs(stage0_guard_kwargs)
                if isinstance(stage0_guard_kwargs, Mapping)
                else None
            ),
            batch_size=int(artifact.get("batch_size", DEFAULT_BATCH_SIZE)),
            threshold_details=artifact.get("threshold_details"),
            class_prototype_details=artifact.get("class_prototype_details"),
        )


def load_anomaly_detector(path: str | Path) -> MahalanobisAnomalyDetector:
    artifact = _load_artifact_file(path)
    if isinstance(artifact, MahalanobisAnomalyDetector):
        return artifact
    if not isinstance(artifact, Mapping):
        raise TypeError("Anomaly detector artifact must deserialize to a mapping or detector instance.")
    return MahalanobisAnomalyDetector.from_artifact(artifact)
