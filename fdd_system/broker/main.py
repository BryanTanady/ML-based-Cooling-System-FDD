"""Broker CLI entrypoint and runtime orchestration.

Example usage:
    python -m fdd_system.broker.main \
      --port /dev/ttyACM0 \
      --baudrate 115200 \
      --input-format bin \
      --fs-hz 800 \
      --model-path experiment/weights/end_to_end_cnn1d_hybrid_calaug.onnx \
      --model-format onnx \
      --embedder auto \
      --preprocessor auto \
      --anomaly-detector-path experiment/weights/end_to_end_anomaly_gate.pt \
      --calibration-seconds 20 \
      --calibration-discard-seconds 2 \
      --gate-runtime-calibration
"""

from __future__ import annotations

import argparse
import logging
import time
from collections import Counter, deque
from typing import Deque

import serial
from tqdm import tqdm

from data_collection.binary_protocol import ADXLBinaryParser
from fdd_system.ML.common.config import OperatingCondition, SensorConfig
from fdd_system.ML.common.preprocessor import CalibrationZNormalizer
from fdd_system.broker.SerialReader import SerialReader
from fdd_system.broker.alerts import AlertSender
from fdd_system.broker.parsing import parse_sample
from fdd_system.broker.pipeline_factory import build_pipeline
from fdd_system.broker.predictions import record_predictions
from fdd_system.broker.windowing import WindowBuilder

EXAMPLE_USAGE = """Examples:
  python -m fdd_system.broker.main --port /dev/ttyACM0 --baudrate 115200 --input-format bin --fs-hz 800 --model-path experiment/weights/end_to_end_cnn1d_hybrid_calaug.onnx --model-format onnx --embedder auto --preprocessor auto --anomaly-detector-path experiment/weights/end_to_end_anomaly_gate.pt --calibration-seconds 20 --calibration-discard-seconds 2 --gate-runtime-calibration
"""


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
    parser.add_argument("--model-path", type=str, required=True, help="Path to trained model (.joblib/.pkl/.onnx)")
    parser.add_argument(
        "--anomaly-detector-path",
        type=str,
        default=None,
        help=(
            "Optional path to a serialized anomaly detector artifact. When provided, broker emits "
            "UNKNOWN for anomalous windows before running the classifier."
        ),
    )
    parser.add_argument(
        "--model-format",
        choices=["auto", "sklearn", "onnx"],
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
            "calibration",
            "calibration_z",
        ],
        default="auto",
        help="Input preprocessor. Default is basic unless overridden by ONNX metadata.",
    )
    parser.add_argument(
        "--calibration-seconds",
        type=float,
        default=20.0,
        help="When using calibration-aware preprocessing, collect this many seconds of startup data before inference.",
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
        default=True,
        help=(
            "Adapt the KNOWN/UNKNOWN gate using the target-normal startup calibration windows "
            "after fitting the runtime preprocessor."
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
    return parser


def _log_prediction_counts(prediction_counts: Counter[int], log: logging.Logger) -> None:
    if not prediction_counts:
        return

    log.info("Prediction counts:")
    for cls_id, count in prediction_counts.items():
        try:
            cls_name = OperatingCondition(cls_id).name
        except ValueError:
            cls_name = f"Unknown({cls_id})"
        log.info("  %s: %s", cls_name, count)


def _is_calibration_preprocessor(value: object) -> bool:
    return isinstance(value, CalibrationZNormalizer)


def _calibration_preprocessor_name(value: object) -> str:
    if isinstance(value, CalibrationZNormalizer):
        return "calibration_z"
    raise TypeError(f"Unsupported calibration preprocessor: {type(value)!r}")


def _fit_runtime_calibration_preprocessor(
    template,
    calibration_windows,
):
    if isinstance(template, CalibrationZNormalizer):
        return CalibrationZNormalizer.fit(calibration_windows)
    raise TypeError(f"Unsupported calibration preprocessor template: {type(template)!r}")

def _collect_calibration_targets(pipeline) -> list[tuple[object, str, str]]:
    targets: list[tuple[object, str, str]] = []

    direct_pre = getattr(pipeline, "preprocessor", None)
    if _is_calibration_preprocessor(direct_pre):
        targets.append((pipeline, "preprocessor", "preprocessor"))

    classifier_pipeline = getattr(pipeline, "classifier_pipeline", None)
    classifier_pre = getattr(classifier_pipeline, "preprocessor", None)
    if _is_calibration_preprocessor(classifier_pre):
        targets.append((classifier_pipeline, "preprocessor", "preprocessor"))

    anomaly_detector = getattr(pipeline, "anomaly_detector", None)
    anomaly_pre = getattr(anomaly_detector, "preprocessor", None)
    if _is_calibration_preprocessor(anomaly_pre):
        targets.append((anomaly_detector, "preprocessor", "preprocessor"))

    return targets


def _apply_runtime_calibration(pipeline, calibration_windows) -> None:
    for owner, attr, target_kind in _collect_calibration_targets(pipeline):
        template = getattr(owner, attr)
        if target_kind == "preprocessor":
            calibrated = _fit_runtime_calibration_preprocessor(template, calibration_windows)
            preprocessor_name = _calibration_preprocessor_name(calibrated)
            preprocessor_kwargs = calibrated.export_kwargs()
            setattr(owner, attr, calibrated)
            if hasattr(owner, "preprocessor_name"):
                owner.preprocessor_name = preprocessor_name
            if hasattr(owner, "preprocessor_kwargs"):
                owner.preprocessor_kwargs = preprocessor_kwargs
            if hasattr(owner, "bundle") and isinstance(getattr(owner, "bundle"), dict):
                owner.bundle["preprocessor_name"] = preprocessor_name
                owner.bundle["preprocessor_kwargs"] = preprocessor_kwargs
            continue

        raise ValueError(f"Unknown calibration target kind: {target_kind}")


def _apply_runtime_gate_calibration(
    pipeline,
    calibration_windows,
    *,
    enabled: bool,
    distance_quantile: float,
    distance_margin: float,
    ambiguity_quantile: float,
    ambiguity_slack: float,
):
    if not enabled:
        return None

    anomaly_detector = getattr(pipeline, "anomaly_detector", None)
    if anomaly_detector is None:
        return None

    recalibrate = getattr(anomaly_detector, "recalibrate_from_normal_data", None)
    if not callable(recalibrate):
        return None

    return recalibrate(
        calibration_windows,
        distance_quantile=distance_quantile,
        distance_margin=distance_margin,
        ambiguity_quantile=ambiguity_quantile,
        ambiguity_slack=ambiguity_slack,
    )


def _format_label_counts(counts: dict[int, int]) -> str:
    if not counts:
        return "none"

    parts = []
    for cls_id, count in sorted(counts.items()):
        try:
            cls_name = OperatingCondition(cls_id).name
        except ValueError:
            cls_name = f"Unknown({cls_id})"
        parts.append(f"{cls_name}={count}")
    return ", ".join(parts)


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
    )
    calibration_targets = _collect_calibration_targets(pipeline)
    calibration_enabled = bool(calibration_targets)
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
    prediction_counts: Counter[int] = Counter()
    end_time = time.time() + args.run_seconds if args.run_seconds else None

    log.info(
        "Broker started. Reading from %s @ %s baud (format=%s, fs_hz=%.3f, alert_api=%s, asset_id=%s, anomaly_detector=%s, calibration=%s)",
        args.port,
        args.baudrate,
        args.input_format,
        float(args.fs_hz),
        args.alert_api_url,
        args.asset_id,
        args.anomaly_detector_path or "disabled",
        (
            f"discard={calibration_discard_seconds:.1f}s + calibrate={calibration_seconds:.1f}s"
            if calibration_enabled
            else "disabled"
        ),
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
            raise RuntimeError("Calibration did not yield any windows; cannot fit runtime preprocessor.")

        _apply_runtime_calibration(pipeline, calibration_windows)
        gate_summary = _apply_runtime_gate_calibration(
            pipeline,
            calibration_windows,
            enabled=bool(args.gate_runtime_calibration),
            distance_quantile=float(args.gate_distance_quantile),
            distance_margin=float(args.gate_distance_margin),
            ambiguity_quantile=float(args.gate_ambiguity_quantile),
            ambiguity_slack=float(args.gate_ambiguity_slack),
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
                    "distance_floor=%.4f, fallback=%.4f->%.4f, ambiguity=%.4f->%.4f, "
                    "unknown_before=%d, unknown_after=%d"
                ),
                gate_summary["num_windows"],
                _format_label_counts(gate_summary["nearest_label_counts"]),
                gate_summary["distance_floor"],
                gate_summary["old_fallback_threshold"],
                gate_summary["new_fallback_threshold"],
                gate_summary["old_ambiguity_ratio_threshold"],
                gate_summary["new_ambiguity_ratio_threshold"],
                gate_summary["unknown_before"],
                gate_summary["unknown_after"],
            )
        log.info("Calibration complete. Broker ready for inference.")

    def handle_sample(ax: float, ay: float, az: float) -> None:
        nonlocal calibration_samples_seen, discard_logged_complete

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
            preds, confs = pipeline.predict_with_confidence([window])
            record_predictions(preds, confs, prediction_counts, alert_sender, log)

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
                handle_sample(float(sample.x), float(sample.y), float(sample.z))

    except KeyboardInterrupt:
        log.info("Broker stopping (Ctrl+C).")
    finally:
        if calibration_progress is not None:
            calibration_progress.close()
        _log_prediction_counts(prediction_counts, log)

        if reader is not None:
            reader.stop()

        if ser is not None:
            try:
                ser.close()
            except Exception:
                pass

    return 0


def main() -> int:
    """Parse CLI args and start broker runtime."""
    args = build_arg_parser().parse_args()
    return run_broker(args)


if __name__ == "__main__":
    raise SystemExit(main())
