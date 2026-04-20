"""Classifier training and prediction helpers for ML training."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset

try:
    import joblib
except ImportError:  # pragma: no cover - optional unless using sklearn backend
    joblib = None

from fdd_system.ML.components.embedding import MLEmbedder2
from fdd_system.ML.components.model import build_classifier_model
from fdd_system.ML.schema import RawAccWindow
from fdd_system.ML.training.common import to_serializable


def _apply_random_amplitude_scaling_batch(
    xb: torch.Tensor,
    *,
    enabled: bool,
    scale_min: float,
    scale_max: float,
) -> torch.Tensor:
    if not enabled or (scale_min == 1.0 and scale_max == 1.0):
        return xb
    if scale_min <= 0.0 or scale_max < scale_min:
        raise ValueError(f"Invalid amplitude scaling bounds: min={scale_min}, max={scale_max}")
    scales = torch.empty((xb.shape[0], 1, 1), device=xb.device, dtype=xb.dtype).uniform_(scale_min, scale_max)
    return xb * scales


def _make_loader(x_np: np.ndarray, y_np: np.ndarray, batch_size: int, *, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(torch.from_numpy(x_np).float(), torch.from_numpy(y_np).long())
    return DataLoader(dataset, batch_size=int(batch_size), shuffle=shuffle)


def sanitize_feature_matrix(x_np: np.ndarray) -> np.ndarray:
    arr = np.asarray(x_np, dtype=np.float32)
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def train_cnn_classifier(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    *,
    label_to_idx: dict[int, int],
    idx_to_label: dict[int, int],
    save_path: Path,
    architecture: str,
    epochs: int,
    batch_size: int,
    early_stop_patience: int,
    label_smoothing: float,
    train_random_amp_scaling: bool,
    amp_scale_min: float,
    amp_scale_max: float,
    onnx_opset: int,
    export_onnx: bool,
    target_len: int,
    preprocessor_name: str,
    preprocessor_kwargs: dict[str, Any],
    classifier_mean: np.ndarray,
    classifier_std: np.ndarray,
    axis_names: list[str],
    device: torch.device,
) -> dict[str, Any]:
    if x_val.shape[0] == 0:
        raise ValueError("CNN training requires at least one validation window.")

    train_loader = _make_loader(x_train, y_train, batch_size, shuffle=True)
    val_loader = _make_loader(x_val, y_val, batch_size, shuffle=False)

    model = build_classifier_model(architecture, n_classes=len(label_to_idx), in_channels=int(x_train.shape[1])).to(device)
    counts = np.bincount(y_train, minlength=len(label_to_idx)).astype(np.float32)
    class_weights = counts.sum() / np.maximum(counts * len(label_to_idx), 1.0)
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float32, device=device),
        label_smoothing=float(label_smoothing),
    )
    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    @torch.no_grad()
    def evaluate(loader: DataLoader) -> tuple[float, float]:
        model.eval()
        total_loss = 0.0
        all_true: list[np.ndarray] = []
        all_pred: list[np.ndarray] = []
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            total_loss += loss.item() * yb.size(0)
            preds = torch.argmax(logits, dim=1)
            all_true.append(yb.cpu().numpy())
            all_pred.append(preds.cpu().numpy())
        y_true = np.concatenate(all_true, axis=0)
        y_pred = np.concatenate(all_pred, axis=0)
        avg_loss = total_loss / max(len(loader.dataset), 1)
        acc = float(accuracy_score(y_true, y_pred))
        return avg_loss, acc

    def train_one_epoch(loader: DataLoader) -> float:
        model.train()
        total_loss = 0.0
        for xb, yb in loader:
            xb = xb.to(device)
            xb = _apply_random_amplitude_scaling_batch(
                xb,
                enabled=train_random_amp_scaling,
                scale_min=float(amp_scale_min),
                scale_max=float(amp_scale_max),
            )
            yb = yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * yb.size(0)
        return total_loss / max(len(loader.dataset), 1)

    history = {"train_loss": [], "val_loss": [], "val_acc": []}
    best_state = None
    best_metric = (-1.0, float("inf"))
    wait = 0

    for epoch in range(1, int(epochs) + 1):
        train_loss = train_one_epoch(train_loader)
        val_loss, val_acc = evaluate(val_loader)
        scheduler.step(val_loss)

        history["train_loss"].append(float(train_loss))
        history["val_loss"].append(float(val_loss))
        history["val_acc"].append(float(val_acc))

        lr_now = optimizer.param_groups[0]["lr"]
        print(
            f"{architecture} epoch={epoch:02d} train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} lr={lr_now:.2e}"
        )

        metric = (val_acc, -val_loss)
        if metric > best_metric:
            best_metric = metric
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= int(early_stop_patience):
                print("Early stopping triggered.")
                break

    if best_state is None:
        raise RuntimeError("Training did not produce a valid CNN state.")

    model.load_state_dict(best_state)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "label_to_idx": {int(k): int(v) for k, v in label_to_idx.items()},
            "idx_to_label": {int(k): int(v) for k, v in idx_to_label.items()},
            "mean": torch.from_numpy(np.asarray(classifier_mean, dtype=np.float32)),
            "std": torch.from_numpy(np.asarray(classifier_std, dtype=np.float32)),
            "classifier_input_normalization": "global_zscore" if np.any(classifier_std != 1.0) else "identity",
            "window_len": int(target_len),
            "architecture": architecture,
            "preprocessor_name": preprocessor_name,
            "preprocessor_kwargs": dict(preprocessor_kwargs),
            "model_axis_names": list(axis_names),
        },
        save_path,
    )

    onnx_path = save_path.with_suffix(".onnx")
    if export_onnx:
        dummy = torch.randn(1, int(x_train.shape[1]), int(target_len), dtype=torch.float32, device=device)
        model.eval()
        with torch.no_grad():
            torch.onnx.export(
                model,
                dummy,
                onnx_path.as_posix(),
                export_params=True,
                do_constant_folding=True,
                input_names=["input"],
                output_names=["logits"],
                dynamic_axes={"input": {0: "batch_size"}, "logits": {0: "batch_size"}},
                opset_version=int(onnx_opset),
            )

    meta_path = save_path.with_suffix(".meta.json")
    metadata = {
        "torch_path": save_path.as_posix(),
        "onnx_path": onnx_path.as_posix() if export_onnx else None,
        "input_shape": [1, int(x_train.shape[1]), int(target_len)],
        "architecture": architecture,
        "embedder": {
            "name": "raw1dcnn",
            "kwargs": {
                "target_len": int(target_len),
                "mean": np.asarray(classifier_mean, dtype=np.float32).reshape(-1).astype(float).tolist(),
                "std": np.asarray(classifier_std, dtype=np.float32).reshape(-1).astype(float).tolist(),
                "axis_names": list(axis_names),
            },
        },
        "preprocessor": {
            "name": preprocessor_name,
            "kwargs": dict(preprocessor_kwargs),
        },
        "labels": {int(k): int(v) for k, v in idx_to_label.items()},
        "model_axis_names": list(axis_names),
    }
    meta_path.write_text(json.dumps(to_serializable(metadata), indent=2), encoding="utf-8")

    return {
        "backend": "cnn1d",
        "model": model,
        "history": history,
        "label_to_idx": label_to_idx,
        "idx_to_label": idx_to_label,
        "save_path": save_path,
        "meta_path": meta_path,
        "onnx_path": onnx_path if export_onnx else None,
    }


def train_ml2_lda_classifier(
    train_windows: list[RawAccWindow],
    train_labels_raw: np.ndarray,
    val_windows: list[RawAccWindow],
    val_labels_raw: np.ndarray,
    *,
    save_path: Path,
    ml2_embedder_kwargs: dict[str, Any],
    preprocessor_name: str,
    preprocessor_kwargs: dict[str, Any],
) -> dict[str, Any]:
    embedder = MLEmbedder2(**dict(ml2_embedder_kwargs))
    x_train = sanitize_feature_matrix(embedder.embed(list(train_windows)))
    y_train = np.asarray(train_labels_raw, dtype=np.int64).reshape(-1)
    if x_train.ndim != 2 or x_train.shape[0] == 0:
        raise ValueError(f"MLEmbedder2 produced invalid training features with shape {x_train.shape}.")

    model = make_pipeline(StandardScaler(), LinearDiscriminantAnalysis(solver="svd"))
    model.fit(x_train, y_train)

    val_acc = float("nan")
    if val_windows:
        x_val = sanitize_feature_matrix(embedder.embed(list(val_windows)))
        y_val = np.asarray(val_labels_raw, dtype=np.int64).reshape(-1)
        val_pred = np.asarray(model.predict(x_val), dtype=np.int64)
        val_acc = float(accuracy_score(y_val, val_pred))

    if joblib is None:
        raise ImportError("ml_lda backend requires joblib to be installed.")

    save_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, save_path)

    meta_path = save_path.with_suffix(".meta.json")
    labels_sorted = sorted(int(v) for v in np.unique(y_train))
    metadata = {
        "sklearn_path": save_path.as_posix(),
        "classifier": {
            "name": "LinearDiscriminantAnalysis",
            "backend": "ml_lda",
            "pipeline": ["StandardScaler", "LinearDiscriminantAnalysis"],
        },
        "embedder": {
            "name": "ml2",
            "kwargs": dict(ml2_embedder_kwargs),
        },
        "preprocessor": {
            "name": preprocessor_name,
            "kwargs": dict(preprocessor_kwargs),
        },
        "labels": {int(label): int(label) for label in labels_sorted},
        "feature_names": embedder.feature_names(),
    }
    meta_path.write_text(json.dumps(to_serializable(metadata), indent=2), encoding="utf-8")

    return {
        "backend": "ml2_lda",
        "model": model,
        "embedder": embedder,
        "history": {"val_acc": [] if not np.isfinite(val_acc) else [float(val_acc)]},
        "save_path": save_path,
        "meta_path": meta_path,
    }


@torch.no_grad()
def _predict_cnn(bundle: dict[str, Any], x_np: np.ndarray, *, batch_size: int, device: torch.device) -> np.ndarray:
    model = bundle["model"]
    model.eval()
    preds: list[np.ndarray] = []
    for start in range(0, len(x_np), int(batch_size)):
        xb = torch.from_numpy(x_np[start : start + int(batch_size)]).float().to(device)
        logits = model(xb)
        pred_idx = torch.argmax(logits, dim=1).cpu().numpy()
        preds.append(pred_idx)
    pred_idx = np.concatenate(preds, axis=0)
    return np.array([bundle["idx_to_label"][int(idx)] for idx in pred_idx], dtype=np.int64)


def _predict_ml_lda(bundle: dict[str, Any], windows_pre: list[RawAccWindow]) -> np.ndarray:
    features = sanitize_feature_matrix(bundle["embedder"].embed(list(windows_pre)))
    return np.asarray(bundle["model"].predict(features), dtype=np.int64)


def predict_classifier(
    bundle: dict[str, Any],
    *,
    x_np: np.ndarray | None,
    windows_pre: list[RawAccWindow] | None,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    backend = str(bundle["backend"]).strip().lower()
    if backend == "cnn1d":
        if x_np is None:
            raise ValueError("CNN prediction requires x_np.")
        return _predict_cnn(bundle, x_np, batch_size=batch_size, device=device)
    if backend == "ml2_lda":
        if windows_pre is None:
            raise ValueError("ml_lda prediction requires windows_pre.")
        return _predict_ml_lda(bundle, windows_pre)
    raise ValueError(f"Unsupported classifier backend: {backend!r}")
