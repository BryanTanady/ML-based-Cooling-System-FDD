"""Broker CLI entrypoint and runtime orchestration.

Example usage:
    python -m fdd_system.broker.main \
      --port /dev/ttyACM0 \
      --baudrate 115200 \
      --input-format bin \
      --fs-hz 800 \
      --model-path experiment/weights/end_to_end_cnn1d_hybrid_calaug.pt \
      --model-format torch \
      --embedder auto \
      --preprocessor auto \
      --anomaly-detector-path experiment/weights/end_to_end_anomaly_gate.pt \
      --no-adabn-runtime-calibration \
      --no-gate-runtime-calibration
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import time
from collections import Counter, deque
from pathlib import Path
from typing import Deque

import numpy as np
import serial
from tqdm import tqdm

from data_collection.binary_protocol import ADXLBinaryParser
from fdd_system.ML.common.anomaly_detector import Stage0WindowGuard
from fdd_system.ML.common.config import OperatingCondition, SensorConfig
from fdd_system.ML.inference.known_unknown_pipeline import KnownUnknownClassificationPipeline
from fdd_system.ML.inference.normality_fault_pipeline import NormalityFaultClassificationPipeline
from fdd_system.broker.io_helpers import AlertSender, SerialReader, WindowBuilder, parse_sample
from fdd_system.broker.prediction_utils import (
    apply_runtime_adabn,
    apply_runtime_gate_calibration,
    build_pipeline,
    format_label_counts,
    log_live_debug_stats,
    log_prediction_counts,
    record_predictions,
    supports_runtime_adabn,
    supports_runtime_gate_calibration,
)

EXAMPLE_USAGE = """Examples:
  python -m fdd_system.broker.main --port /dev/ttyACM0 --baudrate 115200 --input-format bin --fs-hz 800 --model-path experiment/weights/end_to_end_cnn1d_hybrid_calaug.pt --model-format torch --embedder auto --preprocessor auto --no-adabn-runtime-calibration --no-gate-runtime-calibration
"""


class BrokerDataRecorder:
    """Write broker samples to the same CSV schema used by data_collection/getData2.py."""

    def __init__(self, path: str, *, fs_hz: float):
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = out_path.open("w", encoding="utf-8", newline="")
        self._writer = csv.writer(self._fh)
        self._writer.writerow(["idx", "t_us", "X", "Y", "Z", "t_s"])
        self._fh.flush()
        self._interval_us = int(round(1_000_000.0 / float(fs_hz))) if float(fs_hz) > 0 else 0
        self._next_idx = 0

    def record_sample(
        self,
        *,
        ax: float,
        ay: float,
        az: float,
        idx: int | None = None,
        t_us: int | None = None,
    ) -> None:
        row_idx = int(idx) if idx is not None else self._next_idx
        row_t_us = int(t_us) if t_us is not None else row_idx * self._interval_us
        self._writer.writerow(
            [
                row_idx,
                row_t_us,
                float(ax),
                float(ay),
                float(az),
                float(row_t_us) / 1_000_000.0,
            ]
        )
        self._next_idx = row_idx + 1
        self._fh.flush()

    def record_prediction(
        self,
        *,
        preds,
        confs,
        rejection_stage=None,
        rejection_reason=None,
    ) -> None:
        # Keep signature for existing call sites; no-op by design so recorded CSV
        # exactly matches getData2.py format (sample rows only).
        _ = (preds, confs, rejection_stage, rejection_reason)

    def close(self) -> None:
        self._fh.close()


def _resolve_adabn_ema_momentum(model_path: str, cli_value: float | None, log: logging.Logger) -> tuple[float, str]:
    """Resolve AdaBN EMA momentum from CLI, then model metadata, then default."""
    if cli_value is not None:
        return float(cli_value), "cli"

    metadata, meta_source = _load_runtime_calibration_metadata(model_path, log)
    if metadata is not None:
        value = metadata.get("runtime_calibration", {}).get("adabn", {}).get("ema_momentum")
        if value is not None:
            try:
                return float(value), f"metadata:{meta_source}"
            except (TypeError, ValueError):
                log.warning("Invalid runtime_calibration.adabn.ema_momentum in %s: %r", meta_source, value)

    return 0.1, "default"


def _load_runtime_calibration_metadata(
    model_path: str,
    log: logging.Logger,
) -> tuple[dict[str, object] | None, str | None]:
    meta_candidates = [
        Path(model_path).with_suffix(".meta.json"),
        Path(model_path).with_name(f"{Path(model_path).stem}.meta.json"),
    ]
    for meta_path in meta_candidates:
        if not meta_path.exists():
            continue
        try:
            payload = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception as exc:
            log.warning("Failed to parse model metadata at %s: %s", meta_path, exc)
            continue
        if isinstance(payload, dict):
            return payload, meta_path.name
        log.warning("Ignoring non-mapping model metadata at %s", meta_path)
    return None, None


def _coerce_optional_bool(value: object) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        if value in {0, 1}:
            return bool(value)
        return None
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    return None


def _resolve_runtime_calibration_enabled(
    model_path: str,
    cli_value: bool | None,
    log: logging.Logger,
    *,
    kind: str,
) -> tuple[bool, str]:
    if cli_value is not None:
        return bool(cli_value), "cli"

    metadata, meta_source = _load_runtime_calibration_metadata(model_path, log)
    if metadata is not None:
        value = metadata.get("runtime_calibration", {}).get(kind, {}).get("enabled")
        parsed = _coerce_optional_bool(value)
        if parsed is not None:
            return parsed, f"metadata:{meta_source}"
        if value is not None:
            log.warning("Invalid runtime_calibration.%s.enabled in %s: %r", kind, meta_source, value)

    return False, "default"


def build_arg_parser() -> argparse.ArgumentParser:
    """Build broker CLI parser."""
    parser = argparse.ArgumentParser(
        description="Minimal Broker CLI (serial -> AI -> alert API)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=EXAMPLE_USAGE,
    )
    parser.add_argument(
        "--port",
        type=str,
        default="COM3",
        help="Serial port (e.g., COM3, /dev/ttyACM0), or socket://127.0.0.1:9999",
    )
    parser.add_argument("--baudrate", type=int, default=9600, help="Serial baudrate")
    parser.add_argument("--timeout", type=float, default=1.0, help="Serial timeout (seconds)")
    parser.add_argument(
        "--input-format",
        choices=["csv", "bin"],
        default="csv",
        help="Input format from microcontroller: newline-delimited CSV/text ('csv') or binary frames ('bin').",
    )
    parser.add_argument(
        "--fs-hz",
        type=float,
        default=float(SensorConfig.SAMPLING_RATE),
        help="Sampling rate (Hz) used to synthesize timestamps for 9-byte frames.",
    )
    parser.add_argument("--model-path", type=str, required=True, help="Path to trained model (.joblib/.pkl/.onnx/.pt)")
    parser.add_argument(
        "--normality-detector-path",
        type=str,
        default=None,
        help=(
            "Optional path to a serialized normality detector artifact. When provided, broker uses it as "
            "Stage 2 normal gate before downstream unknown/fault inference."
        ),
    )
    parser.add_argument(
        "--anomaly-detector-path",
        type=str,
        default=None,
        help=(
            "Optional path to a serialized anomaly detector artifact. When provided with --normality-detector-path, "
            "this acts as the Stage 3 fault unknown detector."
        ),
    )
    parser.add_argument(
        "--model-format",
        choices=["auto", "sklearn", "onnx", "torch"],
        default="auto",
        help="Model serialization format. Default auto-detects from --model-path suffix.",
    )
    parser.add_argument(
        "--embedder",
        choices=["auto", "ml1", "ml2", "spectrogram2d", "raw1dcnn"],
        default="auto",
        help="Feature embedder to pair with the model.",
    )
    parser.add_argument(
        "--preprocessor",
        choices=[
            "auto",
            "basic",
            "dummy",
            "robust",
            "median",
            "standard",
            "rms",
            "centered_rms",
        ],
        default="auto",
        help=(
            "Input preprocessor. Default is basic unless overridden by ONNX metadata. "
            "'centered_rms' subtracts per-axis window bias before RMS scaling."
        ),
    )
    parser.add_argument(
        "--calibration-seconds",
        type=float,
        default=20.0,
        help="Collect this many seconds of startup data before inference for runtime adaptation (gate/AdaBN).",
    )
    parser.add_argument(
        "--calibration-discard-seconds",
        type=float,
        default=2.0,
        help=(
            "Discard this many initial startup seconds before collecting calibration data. "
            "Use this to skip spin-up transients and match notebook preprocessing."
        ),
    )
    parser.add_argument(
        "--gate-runtime-calibration",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Adapt the KNOWN/UNKNOWN gate using the target-normal startup calibration windows "
            "after preprocessing with the runtime pipeline. "
            "If omitted, broker falls back to model metadata when available; otherwise disabled."
        ),
    )
    parser.add_argument(
        "--gate-distance-quantile",
        type=float,
        default=0.99,
        help="Quantile of calibration-window gate distances used for runtime threshold adaptation.",
    )
    parser.add_argument(
        "--gate-distance-margin",
        type=float,
        default=1.10,
        help="Multiplicative slack applied to the runtime gate distance floor.",
    )
    parser.add_argument(
        "--gate-ambiguity-quantile",
        type=float,
        default=0.99,
        help="Quantile of calibration-window ambiguity ratios used for runtime gate adaptation.",
    )
    parser.add_argument(
        "--gate-ambiguity-slack",
        type=float,
        default=0.05,
        help="Additive slack applied to the runtime ambiguity-ratio threshold.",
    )
    parser.add_argument(
        "--adabn-runtime-calibration",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Recalibrate torch classifier BatchNorm running statistics (AdaBN) using startup calibration windows "
            "after preprocessing/embedder normalization. "
            "If omitted, broker falls back to model metadata when available; otherwise disabled."
        ),
    )
    parser.add_argument(
        "--adabn-batch-size",
        type=int,
        default=0,
        help=(
            "Batch size used for runtime AdaBN recalibration. "
            "Set <=0 to run adaptation in one batch (recommended for exact global stats)."
        ),
    )
    parser.add_argument(
        "--adabn-ema-momentum",
        type=float,
        default=None,
        help=(
            "EMA momentum for runtime AdaBN BatchNorm running-stat updates. "
            "Must be in (0, 1]. Higher values adapt faster to startup calibration data. "
            "If omitted, broker uses model metadata runtime_calibration.adabn.ema_momentum when present; "
            "otherwise defaults to 0.1."
        ),
    )
    parser.add_argument("--loop-delay", type=float, default=0.05, help="Sleep between loop iterations (seconds)")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level")
    parser.add_argument(
        "--run-seconds",
        type=float,
        default=None,
        help="How long to run before exiting. If not set, runs indefinitely.",
    )
    parser.add_argument(
        "--alert-api-url",
        type=str,
        default="http://127.0.0.1:8001/api/alert",
        help="Backend endpoint to receive non-normal prediction alerts.",
    )
    parser.add_argument("--asset-id", type=str, default="FAN-01", help="Asset ID attached to sent alerts.")
    parser.add_argument(
        "--alert-timeout",
        type=float,
        default=1.0,
        help="Alert POST timeout in seconds.",
    )
    parser.add_argument(
        "--record-data-path",
        "-record-data-path",
        type=str,
        default=None,
        help="Optional CSV output path to record raw samples in getData2 format (idx,t_us,X,Y,Z,t_s).",
    )
    parser.add_argument(
        "--debug-live-stats",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Log per-window raw/preprocessed/embedder channel stats plus gate distance details. "
            "Useful for comparing notebook windows against live deployment drift."
        ),
    )
    return parser


def run_broker(args: argparse.Namespace) -> int:
    """Runtime entrypoint: read serial, infer, and publish alerts."""
    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    log = logging.getLogger("broker")

    alert_sender = AlertSender(
        api_url=args.alert_api_url,
        asset_id=args.asset_id,
        timeout_sec=float(args.alert_timeout),
        logger=log,
    )

    buffer: Deque[str] = deque()
    reader: SerialReader | None = None
    ser: serial.SerialBase | None = None
    bin_parser: ADXLBinaryParser | None = None
    recorder: BrokerDataRecorder | None = None

    if args.record_data_path:
        recorder = BrokerDataRecorder(args.record_data_path, fs_hz=float(args.fs_hz))
        log.info("Recording raw samples to %s (idx,t_us,X,Y,Z,t_s)", args.record_data_path)

    if args.input_format == "csv":
        reader = SerialReader(port=args.port, baudrate=args.baudrate, timeout=args.timeout, buffer=buffer)
    else:
        ser = serial.serial_for_url(args.port, baudrate=args.baudrate, timeout=args.timeout)
        bin_parser = ADXLBinaryParser(protocol="9", fs_hz=args.fs_hz)

    wb_fs = float(args.fs_hz) if args.input_format == "bin" else float(SensorConfig.SAMPLING_RATE)
    window_builder = WindowBuilder(SensorConfig.WINDOW_SIZE, sampling_rate_hz=wb_fs)
    pipeline = build_pipeline(
        args.model_path,
        model_format=args.model_format,
        embedder=args.embedder,
        preprocessor=args.preprocessor,
        anomaly_detector_path=args.anomaly_detector_path,
        normality_detector_path=args.normality_detector_path,
    )

    def resolve_stage0_guard(root_pipeline) -> tuple[Stage0WindowGuard | None, str]:
        visited_ids: set[int] = set()
        current = root_pipeline
        while current is not None and id(current) not in visited_ids:
            visited_ids.add(id(current))

            normality_detector = getattr(current, "normality_detector", None)
            normality_guard = getattr(normality_detector, "stage0_guard", None)
            if isinstance(normality_guard, Stage0WindowGuard):
                return normality_guard, "normality_detector"

            anomaly_detector = getattr(current, "anomaly_detector", None)
            anomaly_guard = getattr(anomaly_detector, "stage0_guard", None)
            if isinstance(anomaly_guard, Stage0WindowGuard):
                return anomaly_guard, "anomaly_detector"

            current = getattr(current, "classifier_pipeline", None)

        return None, "disabled"

    stage0_guard, stage0_guard_source = resolve_stage0_guard(pipeline)
    if isinstance(pipeline, NormalityFaultClassificationPipeline):
        if isinstance(getattr(pipeline, "classifier_pipeline", None), KnownUnknownClassificationPipeline):
            pipeline_mode = "stage1+stage2+stage3+stage4"
        else:
            pipeline_mode = "stage1+stage2+stage4"
    elif isinstance(pipeline, KnownUnknownClassificationPipeline):
        pipeline_mode = "stage1+stage3+stage4"
    else:
        pipeline_mode = "stage1+stage4"

    effective_normality_detector = args.normality_detector_path
    if effective_normality_detector is None and isinstance(pipeline, NormalityFaultClassificationPipeline):
        effective_normality_detector = "auto-detected"

    requested_gate_calibration, gate_calibration_source = _resolve_runtime_calibration_enabled(
        args.model_path,
        args.gate_runtime_calibration,
        log,
        kind="gate",
    )
    requested_adabn_calibration, adabn_enabled_source = _resolve_runtime_calibration_enabled(
        args.model_path,
        args.adabn_runtime_calibration,
        log,
        kind="adabn",
    )
    gate_calibration_supported = supports_runtime_gate_calibration(pipeline)
    adabn_calibration_supported = supports_runtime_adabn(pipeline)
    gate_calibration_enabled = requested_gate_calibration and gate_calibration_supported
    adabn_calibration_enabled = requested_adabn_calibration and adabn_calibration_supported
    calibration_enabled = gate_calibration_enabled or adabn_calibration_enabled
    calibration_seconds = max(0.0, float(args.calibration_seconds))
    calibration_discard_seconds = max(0.0, float(args.calibration_discard_seconds)) if calibration_enabled else 0.0
    calibration_target_samples = int(round(calibration_seconds * wb_fs)) if calibration_enabled else 0
    calibration_discard_samples = int(round(calibration_discard_seconds * wb_fs)) if calibration_enabled else 0
    calibration_total_samples = calibration_discard_samples + calibration_target_samples
    calibration_window_builder = (
        WindowBuilder(SensorConfig.WINDOW_SIZE, sampling_rate_hz=wb_fs)
        if calibration_enabled and calibration_target_samples > 0
        else None
    )
    calibration_windows = []
    calibration_samples_seen = 0
    calibration_complete = not calibration_enabled or calibration_target_samples <= 0
    discard_logged_complete = calibration_discard_samples == 0
    calibration_progress = (
        tqdm(total=calibration_discard_seconds + calibration_seconds, desc="Startup", unit="s")
        if calibration_enabled and not calibration_complete
        else None
    )
    adabn_ema_momentum, adabn_momentum_source = _resolve_adabn_ema_momentum(
        args.model_path,
        args.adabn_ema_momentum,
        log,
    )
    if not (0.0 < adabn_ema_momentum <= 1.0):
        raise ValueError(f"--adabn-ema-momentum must be in (0, 1], got {adabn_ema_momentum}.")
    prediction_counts: Counter[int] = Counter()
    end_time = time.time() + args.run_seconds if args.run_seconds else None

    log.info(
        (
            "Broker started. Reading from %s @ %s baud (format=%s, fs_hz=%.3f, alert_api=%s, asset_id=%s, "
            "pipeline=%s, "
            "normality_detector=%s, anomaly_detector=%s, stage0_validator=%s, calibration=%s, gate_calibration=%s, "
            "adabn=%s, adabn_ema_momentum=%.4f (%s), debug_live_stats=%s)"
        ),
        args.port,
        args.baudrate,
        args.input_format,
        float(args.fs_hz),
        args.alert_api_url,
        args.asset_id,
        pipeline_mode,
        effective_normality_detector or "disabled",
        args.anomaly_detector_path or "disabled",
        stage0_guard_source,
        (
            f"discard={calibration_discard_seconds:.1f}s + calibrate={calibration_seconds:.1f}s"
            if calibration_enabled
            else "disabled"
        ),
        (
            f"enabled ({gate_calibration_source})"
            if gate_calibration_enabled
            else (
                f"disabled:unsupported ({gate_calibration_source})"
                if requested_gate_calibration and not gate_calibration_supported
                else f"disabled ({gate_calibration_source})"
            )
        ),
        (
            f"enabled ({adabn_enabled_source})"
            if adabn_calibration_enabled
            else (
                f"disabled:unsupported ({adabn_enabled_source})"
                if requested_adabn_calibration and not adabn_calibration_supported
                else f"disabled ({adabn_enabled_source})"
            )
        ),
        adabn_ema_momentum,
        adabn_momentum_source,
        "enabled" if bool(args.debug_live_stats) else "disabled",
    )
    if calibration_enabled and not calibration_complete:
        if calibration_discard_seconds > 0:
            log.info(
                "Calibration enabled. Discarding %.1f seconds of startup data, then collecting %.1f seconds for calibration before inference.",
                calibration_discard_seconds,
                calibration_seconds,
            )
        else:
            log.info("Calibration enabled. Collecting %.1f seconds of startup data before inference.", calibration_seconds)

    def maybe_finish_calibration() -> None:
        nonlocal calibration_complete, window_builder

        if calibration_complete:
            return
        if calibration_samples_seen < calibration_total_samples:
            return
        if not calibration_windows:
            raise RuntimeError("Calibration did not yield any windows; cannot run runtime adaptation.")

        effective_calibration_windows = list(calibration_windows)
        if stage0_guard is not None:
            calibration_stage0 = stage0_guard.evaluate(effective_calibration_windows)
            accepted_mask = np.asarray(calibration_stage0["accepted_mask"], dtype=bool).reshape(-1)
            if accepted_mask.size != len(effective_calibration_windows):
                raise RuntimeError(
                    "Stage0 calibration mask length mismatch: "
                    f"mask={accepted_mask.size}, windows={len(effective_calibration_windows)}"
                )
            effective_calibration_windows = [
                window for window, accepted in zip(effective_calibration_windows, accepted_mask.tolist()) if accepted
            ]
            log.info(
                "Calibration Stage0 filtering: accepted=%d / total=%d (rejected=%d)",
                len(effective_calibration_windows),
                len(calibration_windows),
                len(calibration_windows) - len(effective_calibration_windows),
            )
            if not effective_calibration_windows:
                raise RuntimeError("All calibration windows were rejected by Stage 0; cannot run runtime adaptation.")

        gate_summary = apply_runtime_gate_calibration(
            pipeline,
            effective_calibration_windows,
            enabled=gate_calibration_enabled,
            distance_quantile=float(args.gate_distance_quantile),
            distance_margin=float(args.gate_distance_margin),
            ambiguity_quantile=float(args.gate_ambiguity_quantile),
            ambiguity_slack=float(args.gate_ambiguity_slack),
        )
        adabn_summary = apply_runtime_adabn(
            pipeline,
            effective_calibration_windows,
            enabled=adabn_calibration_enabled,
            batch_size=int(args.adabn_batch_size) if int(args.adabn_batch_size) > 0 else None,
            ema_momentum=adabn_ema_momentum,
        )
        calibration_complete = True
        window_builder = WindowBuilder(SensorConfig.WINDOW_SIZE, sampling_rate_hz=wb_fs)
        if calibration_progress is not None:
            remaining = (calibration_discard_seconds + calibration_seconds) - float(calibration_progress.n)
            if remaining > 0:
                calibration_progress.update(remaining)
            calibration_progress.close()
        if gate_summary is not None:
            log.info(
                (
                    "Runtime gate calibration updated thresholds: windows=%d, nearest_labels=%s, "
                    "distance_floor=%.4f (effective=%.4f), fallback=%.4f->%.4f, ambiguity=%.4f->%.4f, "
                    "unknown_before=%d, unknown_after=%d"
                ),
                gate_summary["num_windows"],
                format_label_counts(gate_summary["nearest_label_counts"]),
                gate_summary["distance_floor"],
                gate_summary.get("effective_distance_floor", gate_summary["distance_floor"]),
                gate_summary["old_fallback_threshold"],
                gate_summary["new_fallback_threshold"],
                gate_summary["old_ambiguity_ratio_threshold"],
                gate_summary["new_ambiguity_ratio_threshold"],
                gate_summary["unknown_before"],
                gate_summary["unknown_after"],
            )
            if gate_summary.get("normality_threshold_cap_applied", False):
                log.info(
                    "Runtime normality-gate calibration cap applied: cap=%.4f, detector_labels=%s",
                    gate_summary.get("normality_threshold_cap", float("nan")),
                    gate_summary.get("detector_labels", []),
                )
        if adabn_summary is not None:
            log.info(
                (
                    "Runtime AdaBN recalibration complete: windows=%d, preprocessed_windows=%d, feature_shape=%s, "
                    "bn_layers=%d, batch_size=%d, batches=%d, ema_momentum=%.4f"
                ),
                adabn_summary["adaptation_windows"],
                adabn_summary.get("preprocessed_windows", adabn_summary["adaptation_windows"]),
                adabn_summary.get("feature_shape"),
                adabn_summary["bn_layers"],
                adabn_summary["adaptation_batch_size"],
                adabn_summary["adaptation_batches"],
                adabn_summary.get("ema_momentum", adabn_ema_momentum),
            )
        log.info("Calibration complete. Broker ready for inference.")

    def handle_sample(ax: float, ay: float, az: float, *, idx: int | None = None, t_us: int | None = None) -> None:
        nonlocal calibration_samples_seen, discard_logged_complete

        if recorder is not None:
            recorder.record_sample(ax=ax, ay=ay, az=az, idx=idx, t_us=t_us)

        if not calibration_complete:
            seconds_before = calibration_samples_seen / wb_fs if wb_fs > 0 else 0.0
            calibration_samples_seen += 1
            seconds_after = calibration_samples_seen / wb_fs if wb_fs > 0 else (calibration_discard_seconds + calibration_seconds)
            if calibration_progress is not None:
                progress_total_seconds = calibration_discard_seconds + calibration_seconds
                delta_seconds = min(seconds_after, progress_total_seconds) - min(seconds_before, progress_total_seconds)
                if delta_seconds > 0:
                    calibration_progress.update(delta_seconds)
            if calibration_samples_seen <= calibration_discard_samples:
                if (not discard_logged_complete) and calibration_samples_seen == calibration_discard_samples:
                    log.info("Calibration discard complete. Collecting steady-state calibration windows.")
                    discard_logged_complete = True
                maybe_finish_calibration()
                return

            assert calibration_window_builder is not None
            calibration_window = calibration_window_builder.add(ax, ay, az)
            if calibration_window is not None:
                calibration_windows.append(calibration_window)
            maybe_finish_calibration()
            return

        window = window_builder.add(ax, ay, az)
        if window:
            if stage0_guard is not None:
                stage0_details = stage0_guard.evaluate([window])
                accepted_mask = np.asarray(stage0_details["accepted_mask"], dtype=bool)
                if accepted_mask.size > 0 and not bool(accepted_mask[0]):
                    preds = np.asarray([OperatingCondition.UNKNOWN.value], dtype=np.int64)
                    confs = np.asarray([1.0], dtype=np.float32)
                    rejection_stage = np.asarray(["STAGE0"], dtype=object)
                    rejection_reason = np.asarray(
                        [str(np.asarray(stage0_details["rejection_reason"], dtype=object)[0])],
                        dtype=object,
                    )
                    if recorder is not None:
                        recorder.record_prediction(
                            preds=preds,
                            confs=confs,
                            rejection_stage=rejection_stage,
                            rejection_reason=rejection_reason,
                        )
                    record_predictions(
                        preds,
                        confs,
                        prediction_counts,
                        alert_sender,
                        log,
                        rejection_stage=rejection_stage,
                        rejection_reason=rejection_reason,
                    )
                    if bool(args.debug_live_stats):
                        log_live_debug_stats(pipeline, [window], log)
                    return

            predict_details = getattr(pipeline, "predict_details", None)
            if callable(predict_details):
                details = predict_details([window])
                preds = details["predictions"]
                confs = details["confidence"]
                if recorder is not None:
                    recorder.record_prediction(
                        preds=preds,
                        confs=confs,
                        rejection_stage=details.get("rejection_stage"),
                        rejection_reason=details.get("rejection_reason"),
                    )
                record_predictions(
                    preds,
                    confs,
                    prediction_counts,
                    alert_sender,
                    log,
                    rejection_stage=details.get("rejection_stage"),
                    rejection_reason=details.get("rejection_reason"),
                )
                if bool(args.debug_live_stats):
                    log_live_debug_stats(pipeline, [window], log)
            else:
                preds, confs = pipeline.predict_with_confidence([window])
                if recorder is not None:
                    recorder.record_prediction(preds=preds, confs=confs)
                record_predictions(preds, confs, prediction_counts, alert_sender, log)
                if bool(args.debug_live_stats):
                    log_live_debug_stats(pipeline, [window], log)

    try:
        while True:
            if end_time and time.time() >= end_time:
                log.info("Run duration reached; stopping.")
                break

            if args.input_format == "csv":
                if not buffer:
                    time.sleep(args.loop_delay)
                    continue

                line = buffer.popleft()
                sample = parse_sample(line)
                if not sample:
                    log.info("Skipping unparsable line: %s", line.strip())
                    continue

                handle_sample(*sample)
                continue

            assert ser is not None and bin_parser is not None
            chunk = ser.read(4096)
            if not chunk:
                time.sleep(args.loop_delay)
                continue

            samples = bin_parser.feed(chunk)
            if not samples:
                continue

            for sample in samples:
                handle_sample(float(sample.x), float(sample.y), float(sample.z), idx=int(sample.idx), t_us=int(sample.t_us))

    except KeyboardInterrupt:
        log.info("Broker stopping (Ctrl+C).")
    finally:
        if calibration_progress is not None:
            calibration_progress.close()
        log_prediction_counts(prediction_counts, log)

        if reader is not None:
            reader.stop()

        if ser is not None:
            try:
                ser.close()
            except Exception:
                pass
        if recorder is not None:
            recorder.close()

    return 0


def main() -> int:
    """Parse CLI args and start broker runtime."""
    args = build_arg_parser().parse_args()
    return run_broker(args)


if __name__ == "__main__":
    raise SystemExit(main())
