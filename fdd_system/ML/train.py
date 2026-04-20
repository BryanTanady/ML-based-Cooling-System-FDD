"""Config-driven training entrypoint for the end-to-end ML workflow."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/cpen491-matplotlib")

ENV_PYTHON = ROOT / ".env" / "bin" / "python"
if ENV_PYTHON.exists() and Path(sys.executable) != ENV_PYTHON:
    try:
        import numpy  # noqa: F401
    except ModuleNotFoundError:
        os.execv(ENV_PYTHON.as_posix(), [ENV_PYTHON.as_posix(), __file__, *sys.argv[1:]])

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix

from fdd_system.ML.components.detector import (
    fit_mahalanobis_gatekeeper,
    predict_gatekeeper,
    save_mahalanobis_gatekeeper,
)
from fdd_system.ML.training.classifier import (
    predict_classifier,
    train_cnn_classifier,
    train_ml2_lda_classifier,
)
from fdd_system.ML.training.common import (
    DEFAULT_CONFIG_PATH,
    UNKNOWN_LABEL,
    label_name,
    load_config,
    named_label_counts,
    resolve_device,
    resolve_path,
    seed_everything,
    to_serializable,
)
from fdd_system.ML.training.data import (
    prepare_model_inputs,
    prepare_training_dataset,
    stage0_summary_row,
)
from fdd_system.broker.prediction_utils import build_pipeline


def _train_classifier_bundle(
    classifier_cfg: dict[str, Any],
    *,
    prepared,
    model_inputs,
    device,
) -> tuple[str, dict[str, Any]]:
    classifier_backend = str(classifier_cfg.get("backend", "cnn1d")).strip().lower()
    classifier_save_path = resolve_path(classifier_cfg["artifact_path"])

    if classifier_backend == "cnn1d":
        classifier_bundle = train_cnn_classifier(
            model_inputs.x_train_classifier,
            model_inputs.y_train_classifier,
            model_inputs.x_val_classifier_input,
            model_inputs.y_val_classifier,
            label_to_idx=model_inputs.label_to_idx,
            idx_to_label=model_inputs.idx_to_label,
            save_path=classifier_save_path,
            architecture=str(classifier_cfg.get("architecture", "hybrid_timefreq")),
            epochs=int(classifier_cfg.get("epochs", 5)),
            batch_size=int(classifier_cfg.get("batch_size", 8)),
            early_stop_patience=int(classifier_cfg.get("early_stop_patience", 10)),
            label_smoothing=float(classifier_cfg.get("label_smoothing", 0.05)),
            train_random_amp_scaling=bool(classifier_cfg.get("train_random_amp_scaling", True)),
            amp_scale_min=float(classifier_cfg.get("amp_scale_min", 0.8)),
            amp_scale_max=float(classifier_cfg.get("amp_scale_max", 1.2)),
            onnx_opset=int(classifier_cfg.get("onnx_opset", 18)),
            export_onnx=bool(classifier_cfg.get("export_onnx", True)),
            target_len=model_inputs.target_len,
            preprocessor_name=prepared.preprocessor_name,
            preprocessor_kwargs=prepared.preprocessor_kwargs,
            classifier_mean=model_inputs.classifier_mean,
            classifier_std=model_inputs.classifier_std,
            axis_names=model_inputs.axis_names,
            device=device,
        )
        return classifier_backend, classifier_bundle

    if classifier_backend == "ml_lda":
        classifier_bundle = train_ml2_lda_classifier(
            model_inputs.classifier_train_pre,
            model_inputs.y_train_classifier_raw,
            prepared.preprocessed_windows["known_val"],
            model_inputs.y_val_known_raw,
            save_path=classifier_save_path,
            ml2_embedder_kwargs=dict(classifier_cfg.get("ml2_embedder_kwargs", {"highpass_hz": 10.0})),
            preprocessor_name=prepared.preprocessor_name,
            preprocessor_kwargs=prepared.preprocessor_kwargs,
        )
        return classifier_backend, classifier_bundle

    raise ValueError(f"Unsupported classifier backend '{classifier_backend}'.")


def run_training(config_path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    cfg = load_config(config_path)
    seed = int(cfg.get("seed", 42))
    training_cfg = dict(cfg.get("training", {}))
    data_cfg = dict(cfg.get("data", {}))
    stage0_cfg = dict(cfg.get("stage0", {}))
    gate_cfg = dict(cfg.get("gatekeeper", {}))
    classifier_cfg = dict(cfg.get("classifier", {}))
    outputs_cfg = dict(cfg.get("outputs", {}))

    seed_everything(seed, torch_threads=training_cfg.get("torch_threads"))
    device = resolve_device(training_cfg.get("device"))
    print(f"Repo root: {ROOT}")
    print(f"Device: {device}")

    prepared = prepare_training_dataset(data_cfg, stage0_cfg, seed=seed)
    model_inputs = prepare_model_inputs(prepared, classifier_cfg)

    gatekeeper = fit_mahalanobis_gatekeeper(
        model_inputs.x_train_known,
        model_inputs.y_train_known_raw,
        model_inputs.x_val_known,
        model_inputs.y_val_known_raw,
        model_inputs.train_feature_groups,
        model_inputs.val_feature_groups,
        emb_dim=int(gate_cfg.get("embedding_dim", 16)),
        epochs=int(gate_cfg.get("epochs", 40)),
        batch_size=int(gate_cfg.get("batch_size", 512)),
        lr=float(gate_cfg.get("lr", 1e-3)),
        margin=float(gate_cfg.get("margin", 0.5)),
        reg=float(gate_cfg.get("covariance_reg", 1e-3)),
        file_window_score_q=float(gate_cfg.get("file_window_score_q", 0.99)),
        file_threshold_margin=float(gate_cfg.get("file_threshold_margin", 2.1)),
        ambiguity_ratio_threshold=float(gate_cfg.get("ambiguity_ratio_threshold", 1.0)),
        max_prototypes_per_class=int(gate_cfg.get("max_prototypes_per_class", 6)),
        min_windows_per_prototype=int(gate_cfg.get("min_windows_per_prototype", 30)),
        min_silhouette_for_split=float(gate_cfg.get("min_silhouette_for_split", 0.05)),
        random_state=seed,
        kmeans_n_init=int(gate_cfg.get("kmeans_n_init", 10)),
        device=str(device),
    )
    gatekeeper_save_path = resolve_path(gate_cfg["artifact_path"])
    save_mahalanobis_gatekeeper(
        gatekeeper_save_path,
        gatekeeper,
        mean=model_inputs.mean,
        std=model_inputs.std,
        window_len=model_inputs.target_len,
        preprocessor_name=prepared.preprocessor_name,
        preprocessor_kwargs=prepared.preprocessor_kwargs,
        stage0_guard=prepared.stage0_guard,
        batch_size=int(gate_cfg.get("batch_size", 512)),
    )

    classifier_backend, classifier_bundle = _train_classifier_bundle(
        classifier_cfg,
        prepared=prepared,
        model_inputs=model_inputs,
        device=device,
    )

    classifier_known_test_pred = predict_classifier(
        classifier_bundle,
        x_np=model_inputs.x_known_test_classifier_input if classifier_backend == "cnn1d" else None,
        windows_pre=prepared.preprocessed_windows["known_test"] if classifier_backend == "ml_lda" else None,
        batch_size=int(classifier_cfg.get("eval_batch_size", 256)),
        device=device,
    )
    known_test_accuracy = float(accuracy_score(model_inputs.y_known_test_raw, classifier_known_test_pred))

    gatekeeper_eval = predict_gatekeeper(
        gatekeeper,
        model_inputs.x_full_test,
        batch_size=int(gate_cfg.get("batch_size", 512)),
        device=str(device),
    )
    full_test_raw_all = prepared.raw_windows["full_test"]
    full_test_stage0 = prepared.stage0_details["full_test"]
    full_true = np.asarray([int(window.label) for window in full_test_raw_all], dtype=np.int64)
    y_stage1_true_binary = (full_true == UNKNOWN_LABEL).astype(np.int64)
    y_stage1_pred = np.ones(len(full_test_raw_all), dtype=np.int64)
    full_test_stage0_accepted_indices = np.flatnonzero(np.asarray(full_test_stage0["accepted_mask"], dtype=bool))
    y_stage1_pred[full_test_stage0_accepted_indices] = np.asarray(gatekeeper_eval["is_unknown"], dtype=np.int64)
    stage1_cm = confusion_matrix(y_stage1_true_binary, y_stage1_pred, labels=[0, 1])
    tn, fp, fn, tp = (int(v) for v in stage1_cm.ravel())

    runtime_pipeline = build_pipeline(
        classifier_bundle["save_path"].as_posix(),
        anomaly_detector_path=gatekeeper_save_path.as_posix(),
    )
    full_details = runtime_pipeline.predict_details(full_test_raw_all)
    full_preds = np.asarray(full_details["predictions"], dtype=np.int64).reshape(-1)
    full_accuracy = float(accuracy_score(full_true, full_preds))

    smoke_limit = min(int(outputs_cfg.get("smoke_test_samples", 4)), len(full_test_raw_all))
    smoke_windows = full_test_raw_all[:smoke_limit]
    smoke_predictions = (
        np.asarray(runtime_pipeline.predict(smoke_windows), dtype=np.int64).tolist()
        if smoke_windows
        else []
    )

    model_format = "torch" if classifier_backend == "cnn1d" else "sklearn"
    broker_command = (
        "python -m fdd_system.broker.main "
        f"--port /dev/ttyACM0 --baudrate 115200 --input-format bin --fs-hz 800 "
        f"--model-path {classifier_bundle['save_path'].as_posix()} "
        f"--model-format {model_format} --embedder auto --preprocessor auto "
        f"--anomaly-detector-path {gatekeeper_save_path.as_posix()}"
    )

    summary = {
        "config_path": cfg["_config_path"],
        "dataset_path": prepared.dataset_path.as_posix(),
        "device": str(device),
        "classifier_backend": classifier_backend,
        "split_summary": prepared.split_summary,
        "stage0_profile": prepared.stage0_profile,
        "stage0_summary": [
            stage0_summary_row("known_train", prepared.stage0_details["known_train"]),
            stage0_summary_row("known_val", prepared.stage0_details["known_val"]),
            stage0_summary_row("known_test", prepared.stage0_details["known_test"]),
            stage0_summary_row("full_test", prepared.stage0_details["full_test"]),
        ],
        "window_counts_after_preprocessing": {
            "known_train": {"count": len(prepared.preprocessed_windows["known_train"]), "labels": named_label_counts(model_inputs.y_train_known_raw)},
            "known_val": {"count": len(prepared.preprocessed_windows["known_val"]), "labels": named_label_counts(model_inputs.y_val_known_raw)},
            "known_test": {"count": len(prepared.preprocessed_windows["known_test"]), "labels": named_label_counts(model_inputs.y_known_test_raw)},
            "full_test": {"count": len(prepared.preprocessed_windows["full_test"]), "labels": named_label_counts(model_inputs.y_full_test_raw)},
            "classifier_train": {"count": len(model_inputs.classifier_train_pre), "labels": named_label_counts(model_inputs.y_train_classifier_raw)},
            "preprocessor": prepared.preprocessor_display_name,
        },
        "artifacts": {
            "classifier_model_path": classifier_bundle["save_path"].as_posix(),
            "classifier_meta_path": classifier_bundle["meta_path"].as_posix(),
            "classifier_onnx_path": None
            if classifier_bundle.get("onnx_path") is None
            else classifier_bundle["onnx_path"].as_posix(),
            "anomaly_detector_path": gatekeeper_save_path.as_posix(),
        },
        "gatekeeper": {
            "num_prototypes": int(len(gatekeeper["prototype_table"])),
            "fallback_threshold": float(gatekeeper["fallback_threshold"]),
            "ambiguity_ratio_threshold": float(gatekeeper["ambiguity_ratio_threshold"]),
            "threshold_details": gatekeeper["threshold_details"],
            "stage1_binary_evaluation": {
                "accuracy": float(accuracy_score(y_stage1_true_binary, y_stage1_pred)),
                "known_false_positive_rate": float(fp / (fp + tn + 1e-12)),
                "unknown_recall": float(tp / (tp + fn + 1e-12)),
                "stage0_rejected_windows": int(np.asarray(full_test_stage0["rejected_mask"], dtype=bool).sum()),
                "stage0_rejected_known": int(
                    np.sum(np.asarray(full_test_stage0["rejected_mask"], dtype=bool) & (full_true != UNKNOWN_LABEL))
                ),
                "stage0_rejected_unknown": int(
                    np.sum(np.asarray(full_test_stage0["rejected_mask"], dtype=bool) & (full_true == UNKNOWN_LABEL))
                ),
                "unknown_folder_present": bool(prepared.has_unknown),
            },
        },
        "classifier": {
            "known_test_accuracy": known_test_accuracy,
            "history": classifier_bundle["history"],
        },
        "full_pipeline_evaluation": {
            "window_accuracy": full_accuracy,
            "true_label_counts": named_label_counts(full_true),
            "pred_label_counts": named_label_counts(full_preds),
        },
        "smoke_test": {
            "num_windows": int(smoke_limit),
            "predictions": smoke_predictions,
            "prediction_names": [label_name(int(pred)) for pred in smoke_predictions],
        },
        "broker_command": broker_command,
    }

    summary_path = resolve_path(
        outputs_cfg.get("summary_json", "fdd_system/ML/weights/end_to_end_training_summary.json")
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(to_serializable(summary), indent=2), encoding="utf-8")

    print()
    print(f"Saved classifier: {classifier_bundle['save_path']}")
    print(f"Saved anomaly detector: {gatekeeper_save_path}")
    print(f"Known-test classifier accuracy: {known_test_accuracy:.4f}")
    print(f"Full-pipeline window accuracy: {full_accuracy:.4f}")
    print(f"Summary JSON: {summary_path}")
    print("Broker command:")
    print(broker_command)

    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train the end-to-end ML stack from fdd_system/ML/config.yaml.")
    parser.add_argument(
        "--config",
        type=str,
        default=DEFAULT_CONFIG_PATH.as_posix(),
        help="Path to the YAML config file. All training parameters are read from this file.",
    )
    args = parser.parse_args(argv)
    run_training(args.config)
    return 0


__all__ = [
    "DEFAULT_CONFIG_PATH",
    "main",
    "run_training",
]


if __name__ == "__main__":
    raise SystemExit(main())
