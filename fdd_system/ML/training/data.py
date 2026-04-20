"""Dataset preparation helpers for config-driven ML training."""

from __future__ import annotations

from collections import Counter, OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.model_selection import train_test_split

from experiment.utils import prepare_training_data
from fdd_system.ML.components.detector import Stage0WindowGuard
from fdd_system.ML.components.preprocessing import (
    CenteredRMSNormalization,
    DummyPreprocessor,
    MedianRemoval,
    RMSNormalization,
    StandardZNormal,
)
from fdd_system.ML.schema import RawAccWindow
from fdd_system.ML.training.common import UNKNOWN_LABEL, parse_known_folders, resolve_path

DEFAULT_DATA_COLUMNS = ["X", "Y", "Z"]


@dataclass
class PreparedDataset:
    dataset_path: Path
    data_columns: list[str]
    remove_first_second: float
    unknown_dirname: str
    has_unknown: bool
    known_folder_to_label: OrderedDict[str, int]
    split_index: OrderedDict[str, dict[str, list[str]]]
    split_summary: list[dict[str, Any]]
    file_maps: dict[str, OrderedDict[int, list[str]]]
    preprocessor: object
    preprocessor_name: str
    preprocessor_kwargs: dict[str, Any]
    preprocessor_display_name: str
    stage0_guard: Stage0WindowGuard | None
    stage0_profile: dict[str, Any] | None
    raw_windows: dict[str, list[RawAccWindow]]
    stage0_details: dict[str, dict[str, Any]]
    preprocessed_windows: dict[str, list[RawAccWindow]]
    grouped_windows: dict[str, list[dict[str, Any]]]


@dataclass
class PreparedModelInputs:
    axis_names: list[str]
    classifier_train_pre: list[RawAccWindow]
    target_len: int
    x_train_known_raw: np.ndarray
    y_train_known_raw: np.ndarray
    x_val_known_raw: np.ndarray
    y_val_known_raw: np.ndarray
    x_known_test_raw: np.ndarray
    y_known_test_raw: np.ndarray
    x_full_test_raw: np.ndarray
    y_full_test_raw: np.ndarray
    x_train_known: np.ndarray
    x_val_known: np.ndarray
    x_full_test: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    train_feature_groups: list[dict[str, Any]]
    val_feature_groups: list[dict[str, Any]]
    x_train_classifier_raw: np.ndarray
    y_train_classifier_raw: np.ndarray
    classifier_mean: np.ndarray
    classifier_std: np.ndarray
    x_train_classifier: np.ndarray
    x_val_classifier_input: np.ndarray
    x_known_test_classifier_input: np.ndarray
    label_to_idx: dict[int, int]
    idx_to_label: dict[int, int]
    y_train_classifier: np.ndarray
    y_val_classifier: np.ndarray


def split_files(files: list[str], *, seed: int, test_ratio: float, val_ratio_in_trainval: float) -> dict[str, list[str]]:
    files = list(files)
    if not files:
        return {"train": [], "val": [], "test": []}
    if len(files) == 1:
        return {"train": files, "val": [], "test": []}
    if len(files) == 2:
        return {"train": files[:1], "val": [], "test": files[1:]}

    trainval_files, test_files = train_test_split(
        files,
        test_size=float(test_ratio),
        shuffle=True,
        random_state=int(seed),
    )

    if len(trainval_files) <= 1 or float(val_ratio_in_trainval) <= 0.0:
        return {"train": trainval_files, "val": [], "test": test_files}

    train_files, val_files = train_test_split(
        trainval_files,
        test_size=float(val_ratio_in_trainval),
        shuffle=True,
        random_state=int(seed),
    )
    return {"train": train_files, "val": val_files, "test": test_files}


def build_split_index(
    dataset_path: Path,
    *,
    known_folder_to_label: OrderedDict[str, int],
    unknown_dirname: str,
    seed: int,
    test_ratio: float,
    val_ratio_in_trainval: float,
) -> tuple[OrderedDict[str, dict[str, list[str]]], bool]:
    available_folders = sorted(path.name.lower() for path in dataset_path.iterdir() if path.is_dir())
    unsupported = sorted(set(available_folders) - (set(known_folder_to_label) | {unknown_dirname}))
    if unsupported:
        raise ValueError(f"Unsupported folders under {dataset_path}: {unsupported}")

    known_folders = [folder for folder in known_folder_to_label if folder in available_folders]
    if not known_folders:
        raise ValueError(f"No known folders found under {dataset_path}.")

    has_unknown = unknown_dirname in available_folders
    split_index: OrderedDict[str, dict[str, list[str]]] = OrderedDict()
    for folder_name in known_folders + ([unknown_dirname] if has_unknown else []):
        files = sorted(str(path) for path in (dataset_path / folder_name).glob("*.csv"))
        split_index[folder_name] = split_files(
            files,
            seed=seed,
            test_ratio=test_ratio,
            val_ratio_in_trainval=val_ratio_in_trainval,
        )
    return split_index, has_unknown


def file_map_for(
    split_index: OrderedDict[str, dict[str, list[str]]],
    known_folder_to_label: OrderedDict[str, int],
    *,
    split_name: str,
    include_unknown: bool,
    unknown_dirname: str,
) -> OrderedDict[int, list[str]]:
    mapping: OrderedDict[int, list[str]] = OrderedDict()
    for folder_name, label in known_folder_to_label.items():
        if folder_name not in split_index:
            continue
        mapping[int(label)] = split_index[folder_name][split_name]
    if include_unknown and unknown_dirname in split_index:
        mapping[UNKNOWN_LABEL] = split_index[unknown_dirname][split_name]
    return mapping


def prepare_raw_windows_for_map(
    file_map: OrderedDict[int, list[str]],
    *,
    cols: list[str],
    remove_first_second: float,
    shuffle: bool,
) -> list[RawAccWindow]:
    return prepare_training_data(
        file_map,
        shuffle=shuffle,
        col_names=cols,
        remove_first_second=remove_first_second,
    )


def make_selected_preprocessor(name: str, kwargs: dict[str, Any]) -> tuple[object, str, dict[str, Any], str]:
    normalized = str(name).strip().lower()
    kwargs = dict(kwargs)

    if normalized in {"basic", "median"}:
        preprocessor = MedianRemoval()
        display_name = "Median removal"
    elif normalized == "dummy":
        preprocessor = DummyPreprocessor()
        display_name = "Dummy"
    elif normalized == "rms":
        preprocessor = RMSNormalization(**kwargs)
        display_name = "RMS normalization"
    elif normalized == "centered_rms":
        preprocessor = CenteredRMSNormalization(**kwargs)
        display_name = "Centered RMS normalization"
    elif normalized in {"robust", "standard"}:
        preprocessor = StandardZNormal(**kwargs)
        display_name = "Standard Z normalization"
    else:
        raise ValueError(f"Unsupported preprocessor '{name}'.")

    exported_kwargs = preprocessor.export_kwargs() if hasattr(preprocessor, "export_kwargs") else dict(kwargs)
    return preprocessor, normalized, exported_kwargs, display_name


def stage0_details_for_windows(
    windows: list[RawAccWindow],
    *,
    stage0_guard: Stage0WindowGuard | None,
) -> dict[str, Any]:
    if stage0_guard is None:
        n = len(windows)
        return {
            "accepted_mask": np.ones(n, dtype=bool),
            "rejected_mask": np.zeros(n, dtype=bool),
            "rejection_reason": np.full(n, Stage0WindowGuard.REASON_OK, dtype=object),
            "rms": np.full(n, np.nan, dtype=np.float32),
            "axis_lengths": np.full((n, 3), -1, dtype=np.int32),
        }
    return stage0_guard.evaluate(windows)


def filter_windows_by_stage0(windows: list[RawAccWindow], stage0_details: dict[str, Any]) -> list[RawAccWindow]:
    return [
        window
        for window, keep in zip(windows, np.asarray(stage0_details["accepted_mask"], dtype=bool).tolist())
        if keep
    ]


def stage0_summary_row(split_name: str, stage0_details: dict[str, Any]) -> dict[str, Any]:
    total_windows = int(len(stage0_details["accepted_mask"]))
    accepted_windows = int(np.asarray(stage0_details["accepted_mask"], dtype=bool).sum())
    rejected_windows = int(np.asarray(stage0_details["rejected_mask"], dtype=bool).sum())
    reason_counts = Counter(str(reason) for reason in np.asarray(stage0_details["rejection_reason"], dtype=object).tolist())

    row: dict[str, Any] = {
        "split": split_name,
        "total_windows": total_windows,
        "accepted_windows": accepted_windows,
        "rejected_windows": rejected_windows,
        "accepted_pct": 100.0 * accepted_windows / max(total_windows, 1),
    }
    for reason in (
        Stage0WindowGuard.REASON_OK,
        Stage0WindowGuard.REASON_WRONG_SHAPE,
        Stage0WindowGuard.REASON_EMPTY_WINDOW,
        Stage0WindowGuard.REASON_MISMATCHED_AXIS_LENGTHS,
        Stage0WindowGuard.REASON_NAN_OR_INF,
        Stage0WindowGuard.REASON_RMS_TOO_LOW,
        Stage0WindowGuard.REASON_RMS_TOO_HIGH,
    ):
        row[reason] = int(reason_counts.get(reason, 0))
    return row


def prepare_grouped_windows_for_map(
    file_map: OrderedDict[int, list[str]],
    *,
    cols: list[str],
    remove_first_second: float,
    stage0_guard: Stage0WindowGuard | None,
    preprocessor,
) -> list[dict[str, Any]]:
    grouped: list[dict[str, Any]] = []
    for label, paths in file_map.items():
        for path in paths:
            raw_windows = prepare_training_data(
                {int(label): [path]},
                shuffle=False,
                col_names=cols,
                remove_first_second=remove_first_second,
            )
            raw_windows = filter_windows_by_stage0(
                raw_windows,
                stage0_details_for_windows(raw_windows, stage0_guard=stage0_guard),
            )
            if not raw_windows:
                continue
            windows = preprocessor.preprocess(raw_windows)
            if not windows:
                continue
            grouped.append(
                {
                    "label": int(label),
                    "path": str(path),
                    "windows": windows,
                }
            )
    return grouped


def stack_windows(
    windows: list[RawAccWindow],
    *,
    target_len: int | None = None,
) -> tuple[np.ndarray, np.ndarray, int]:
    if not windows:
        raise ValueError("No windows found for the requested split.")

    if target_len is None:
        target_len = min(min(len(window.acc_x), len(window.acc_y), len(window.acc_z)) for window in windows)

    x = np.empty((len(windows), 3, target_len), dtype=np.float32)
    y = np.empty((len(windows),), dtype=np.int64)
    for idx, window in enumerate(windows):
        x[idx, 0] = np.asarray(window.acc_x, dtype=np.float32)[:target_len]
        x[idx, 1] = np.asarray(window.acc_y, dtype=np.float32)[:target_len]
        x[idx, 2] = np.asarray(window.acc_z, dtype=np.float32)[:target_len]
        y[idx] = int(window.label)
    return x, y, int(target_len)


def build_group_feature_batches(
    grouped_windows: list[dict[str, Any]],
    *,
    target_len: int,
    mean: np.ndarray,
    std: np.ndarray,
) -> list[dict[str, Any]]:
    feature_groups: list[dict[str, Any]] = []
    for group in grouped_windows:
        x_group_raw, y_group, _ = stack_windows(group["windows"], target_len=target_len)
        feature_groups.append(
            {
                "label": int(group["label"]),
                "path": group["path"],
                "num_windows": int(len(y_group)),
                "X": (x_group_raw - mean) / std,
                "y": y_group,
            }
        )
    return feature_groups


def with_noise(window: RawAccWindow, noise_std: float) -> RawAccWindow:
    return RawAccWindow(
        acc_x=window.acc_x + np.random.normal(0, noise_std, size=window.acc_x.shape),
        acc_y=window.acc_y + np.random.normal(0, noise_std, size=window.acc_y.shape),
        acc_z=window.acc_z + np.random.normal(0, noise_std, size=window.acc_z.shape),
        label=window.label,
        device_id=window.device_id,
        timestamps=window.timestamps,
        sampling_rate_hz=window.sampling_rate_hz,
        acc_mag=window.acc_mag,
    )


def prepare_training_dataset(
    data_cfg: dict[str, Any],
    stage0_cfg: dict[str, Any],
    *,
    seed: int,
) -> PreparedDataset:
    dataset_path = resolve_path(data_cfg["dataset_path"])
    cols = list(DEFAULT_DATA_COLUMNS)
    remove_first_second = float(data_cfg.get("remove_first_second", 0.0))
    unknown_dirname = str(data_cfg.get("unknown_folder", "unknown")).strip().lower()
    known_folder_to_label = parse_known_folders(
        data_cfg.get(
            "known_folders",
            {
                "normal": "NORMAL",
                "blocked": "BLOCKED_AIRFLOW",
                "interfere": "INTERFERENCE",
                "imbalance": "IMBALANCE",
            },
        )
    )

    split_cfg = dict(data_cfg.get("split", {}))
    split_index, has_unknown = build_split_index(
        dataset_path,
        known_folder_to_label=known_folder_to_label,
        unknown_dirname=unknown_dirname,
        seed=seed,
        test_ratio=float(split_cfg.get("test_ratio", 0.2)),
        val_ratio_in_trainval=float(split_cfg.get("val_ratio_in_trainval", 0.2)),
    )

    split_summary = [
        {
            "folder": folder_name,
            "train_files": len(parts["train"]),
            "val_files": len(parts["val"]),
            "test_files": len(parts["test"]),
            "total_files": sum(len(parts[name]) for name in ("train", "val", "test")),
        }
        for folder_name, parts in split_index.items()
    ]

    file_maps = {
        "known_train": file_map_for(
            split_index,
            known_folder_to_label,
            split_name="train",
            include_unknown=False,
            unknown_dirname=unknown_dirname,
        ),
        "known_val": file_map_for(
            split_index,
            known_folder_to_label,
            split_name="val",
            include_unknown=False,
            unknown_dirname=unknown_dirname,
        ),
        "known_test": file_map_for(
            split_index,
            known_folder_to_label,
            split_name="test",
            include_unknown=False,
            unknown_dirname=unknown_dirname,
        ),
        "full_test": file_map_for(
            split_index,
            known_folder_to_label,
            split_name="test",
            include_unknown=True,
            unknown_dirname=unknown_dirname,
        ),
    }

    preprocessor, preprocessor_name, preprocessor_kwargs, preprocessor_display_name = make_selected_preprocessor(
        str(data_cfg.get("preprocessor", "centered_rms")),
        dict(data_cfg.get("preprocessor_kwargs", {})),
    )

    stage0_guard: Stage0WindowGuard | None = None
    stage0_profile: dict[str, Any] | None = None
    if bool(stage0_cfg.get("enabled", True)):
        reference_folder = str(stage0_cfg.get("reference_folder", "normal")).strip().lower()
        if reference_folder not in split_index:
            raise ValueError(f"Stage 0 reference folder '{reference_folder}' is not available in {dataset_path}.")
        stage0_label = int(known_folder_to_label[reference_folder])
        stage0_fit_windows_raw = prepare_training_data(
            {stage0_label: split_index[reference_folder]["train"]},
            shuffle=False,
            col_names=cols,
            remove_first_second=remove_first_second,
        )
        if not stage0_fit_windows_raw:
            raise ValueError("Stage 0 requires reference training windows to fit RMS sanity thresholds.")
        stage0_guard = Stage0WindowGuard.fit(
            stage0_fit_windows_raw,
            rms_mode=str(stage0_cfg.get("rms_mode", "centered_window")),
            rms_lower_q=float(stage0_cfg.get("rms_lower_q", 0.25)),
            rms_upper_q=float(stage0_cfg.get("rms_upper_q", 0.95)),
            rms_lower_scale=float(stage0_cfg.get("rms_lower_scale", 0.5)),
            rms_upper_scale=float(stage0_cfg.get("rms_upper_scale", 1.15)),
            fit_lower_bound=bool(stage0_cfg.get("fit_lower_bound", True)),
            fit_upper_bound=bool(stage0_cfg.get("fit_upper_bound", False)),
        )
        stage0_profile = {
            "fit_source": f"{reference_folder}_train",
            "fit_windows": int(len(stage0_fit_windows_raw)),
            "expected_len": int(stage0_guard.expected_len),
            "rms_mode": stage0_guard.rms_mode,
            "fit_lower_bound": bool(stage0_guard.rms_lower_bound is not None),
            "fit_upper_bound": bool(stage0_guard.rms_upper_bound is not None),
            "rms_lower_bound": None if stage0_guard.rms_lower_bound is None else float(stage0_guard.rms_lower_bound),
            "rms_upper_bound": None if stage0_guard.rms_upper_bound is None else float(stage0_guard.rms_upper_bound),
            "calibration_rms_mean": float(stage0_guard.calibration_rms_mean),
            "calibration_rms_std": float(stage0_guard.calibration_rms_std),
            "calibration_rms_median": float(stage0_guard.calibration_rms_median),
            "calibration_rms_q_low": float(stage0_guard.calibration_rms_lower_quantile),
            "calibration_rms_q_high": float(stage0_guard.calibration_rms_upper_quantile),
        }

    raw_windows = {
        "known_train": prepare_raw_windows_for_map(
            file_maps["known_train"],
            cols=cols,
            remove_first_second=remove_first_second,
            shuffle=True,
        ),
        "known_val": prepare_raw_windows_for_map(
            file_maps["known_val"],
            cols=cols,
            remove_first_second=remove_first_second,
            shuffle=False,
        ),
        "known_test": prepare_raw_windows_for_map(
            file_maps["known_test"],
            cols=cols,
            remove_first_second=remove_first_second,
            shuffle=False,
        ),
        "full_test": prepare_raw_windows_for_map(
            file_maps["full_test"],
            cols=cols,
            remove_first_second=remove_first_second,
            shuffle=False,
        ),
    }

    stage0_details = {
        name: stage0_details_for_windows(windows, stage0_guard=stage0_guard)
        for name, windows in raw_windows.items()
    }
    preprocessed_windows = {
        name: preprocessor.preprocess(filter_windows_by_stage0(raw_windows[name], stage0_details[name]))
        for name in raw_windows
    }

    if not all(preprocessed_windows[name] for name in ("known_train", "known_val", "known_test", "full_test")):
        raise ValueError("Training requires non-empty train/val/test windows after Stage 0 and preprocessing.")

    grouped_windows = {
        "known_train": prepare_grouped_windows_for_map(
            file_maps["known_train"],
            cols=cols,
            remove_first_second=remove_first_second,
            stage0_guard=stage0_guard,
            preprocessor=preprocessor,
        ),
        "known_val": prepare_grouped_windows_for_map(
            file_maps["known_val"],
            cols=cols,
            remove_first_second=remove_first_second,
            stage0_guard=stage0_guard,
            preprocessor=preprocessor,
        ),
    }

    return PreparedDataset(
        dataset_path=dataset_path,
        data_columns=cols,
        remove_first_second=remove_first_second,
        unknown_dirname=unknown_dirname,
        has_unknown=has_unknown,
        known_folder_to_label=known_folder_to_label,
        split_index=split_index,
        split_summary=split_summary,
        file_maps=file_maps,
        preprocessor=preprocessor,
        preprocessor_name=preprocessor_name,
        preprocessor_kwargs=preprocessor_kwargs,
        preprocessor_display_name=preprocessor_display_name,
        stage0_guard=stage0_guard,
        stage0_profile=stage0_profile,
        raw_windows=raw_windows,
        stage0_details=stage0_details,
        preprocessed_windows=preprocessed_windows,
        grouped_windows=grouped_windows,
    )


def prepare_model_inputs(
    prepared: PreparedDataset,
    classifier_cfg: dict[str, Any],
) -> PreparedModelInputs:
    known_train_pre = prepared.preprocessed_windows["known_train"]
    known_val_pre = prepared.preprocessed_windows["known_val"]
    known_test_pre = prepared.preprocessed_windows["known_test"]
    full_test_pre = prepared.preprocessed_windows["full_test"]

    classifier_train_pre = list(known_train_pre)
    noise_copies = int(classifier_cfg.get("noise_copies", 0))
    noise_std_g = float(classifier_cfg.get("noise_std_g", 0.0))
    if noise_copies > 0 and noise_std_g > 0.0:
        noisy_windows: list[RawAccWindow] = []
        for _ in range(noise_copies):
            noisy_windows.extend(with_noise(window, noise_std_g) for window in classifier_train_pre)
        classifier_train_pre = classifier_train_pre + noisy_windows

    x_train_known_raw, y_train_known_raw, target_len = stack_windows(known_train_pre)
    x_val_known_raw, y_val_known_raw, _ = stack_windows(known_val_pre, target_len=target_len)
    x_known_test_raw, y_known_test_raw, _ = stack_windows(known_test_pre, target_len=target_len)
    x_full_test_raw, y_full_test_raw, _ = stack_windows(full_test_pre, target_len=target_len)
    x_train_classifier_raw, y_train_classifier_raw, _ = stack_windows(classifier_train_pre, target_len=target_len)

    mean = x_train_known_raw.mean(axis=(0, 2), keepdims=True)
    std = x_train_known_raw.std(axis=(0, 2), keepdims=True) + 1e-6
    x_train_known = (x_train_known_raw - mean) / std
    x_val_known = (x_val_known_raw - mean) / std
    x_full_test = (x_full_test_raw - mean) / std

    train_feature_groups = build_group_feature_batches(
        prepared.grouped_windows["known_train"],
        target_len=target_len,
        mean=mean,
        std=std,
    )
    val_feature_groups = build_group_feature_batches(
        prepared.grouped_windows["known_val"],
        target_len=target_len,
        mean=mean,
        std=std,
    )

    classifier_mean = np.zeros((1, x_train_classifier_raw.shape[1], 1), dtype=np.float32)
    classifier_std = np.ones((1, x_train_classifier_raw.shape[1], 1), dtype=np.float32)
    x_train_classifier = np.asarray(x_train_classifier_raw, dtype=np.float32)
    x_val_classifier_input = np.asarray(x_val_known_raw, dtype=np.float32)
    x_known_test_classifier_input = np.asarray(x_known_test_raw, dtype=np.float32)

    known_labels = [
        int(label)
        for folder, label in prepared.known_folder_to_label.items()
        if folder in prepared.split_index
    ]
    label_to_idx = {int(label): idx for idx, label in enumerate(known_labels)}
    idx_to_label = {idx: int(label) for label, idx in label_to_idx.items()}
    y_train_classifier = np.array([label_to_idx[int(label)] for label in y_train_classifier_raw], dtype=np.int64)
    y_val_classifier = np.array([label_to_idx[int(label)] for label in y_val_known_raw], dtype=np.int64)

    return PreparedModelInputs(
        axis_names=["x", "y", "z"],
        classifier_train_pre=classifier_train_pre,
        target_len=target_len,
        x_train_known_raw=x_train_known_raw,
        y_train_known_raw=y_train_known_raw,
        x_val_known_raw=x_val_known_raw,
        y_val_known_raw=y_val_known_raw,
        x_known_test_raw=x_known_test_raw,
        y_known_test_raw=y_known_test_raw,
        x_full_test_raw=x_full_test_raw,
        y_full_test_raw=y_full_test_raw,
        x_train_known=x_train_known,
        x_val_known=x_val_known,
        x_full_test=x_full_test,
        mean=mean,
        std=std,
        train_feature_groups=train_feature_groups,
        val_feature_groups=val_feature_groups,
        x_train_classifier_raw=x_train_classifier_raw,
        y_train_classifier_raw=y_train_classifier_raw,
        classifier_mean=np.asarray(classifier_mean, dtype=np.float32),
        classifier_std=np.asarray(classifier_std, dtype=np.float32),
        x_train_classifier=x_train_classifier,
        x_val_classifier_input=x_val_classifier_input,
        x_known_test_classifier_input=x_known_test_classifier_input,
        label_to_idx=label_to_idx,
        idx_to_label=idx_to_label,
        y_train_classifier=y_train_classifier,
        y_val_classifier=y_val_classifier,
    )
