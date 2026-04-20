"""Shared helpers for the config-driven ML training flow."""

from __future__ import annotations

import random
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from fdd_system.ML.schema import OperatingCondition

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = ROOT / "fdd_system" / "ML" / "config.yaml"
UNKNOWN_LABEL = int(OperatingCondition.UNKNOWN.value)


def resolve_path(path_like: str | Path) -> Path:
    path = Path(path_like).expanduser()
    if not path.is_absolute():
        path = (ROOT / path).resolve()
    return path


def load_config(config_path: str | Path) -> dict[str, Any]:
    path = resolve_path(config_path)
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected YAML mapping in {path}, got {type(payload)!r}.")
    payload["_config_path"] = path.as_posix()
    return payload


def to_serializable(value: Any) -> Any:
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, OrderedDict):
        return {str(k): to_serializable(v) for k, v in value.items()}
    if isinstance(value, dict):
        return {str(k): to_serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_serializable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def seed_everything(seed: int, *, torch_threads: int | None = None) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch_threads is not None and int(torch_threads) > 0:
        torch.set_num_threads(int(torch_threads))


def resolve_device(device_name: str | None) -> torch.device:
    normalized = str(device_name or "auto").strip().lower()
    if normalized == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(normalized)


def label_name(label: int) -> str:
    try:
        return OperatingCondition(int(label)).name
    except ValueError:
        return str(label)


def named_label_counts(labels: np.ndarray) -> dict[str, int]:
    counts = Counter(int(label) for label in np.asarray(labels, dtype=np.int64).reshape(-1).tolist())
    return {label_name(label): int(counts[label]) for label in sorted(counts)}


def parse_known_folders(raw_mapping: Any) -> OrderedDict[str, int]:
    if not isinstance(raw_mapping, dict) or not raw_mapping:
        raise ValueError("data.known_folders must be a non-empty mapping of folder name to OperatingCondition name.")

    parsed: OrderedDict[str, int] = OrderedDict()
    for folder_name, enum_name in raw_mapping.items():
        normalized_folder = str(folder_name).strip().lower()
        condition = OperatingCondition[str(enum_name).strip().upper()]
        parsed[normalized_folder] = int(condition.value)
    return parsed
