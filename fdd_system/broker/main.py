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
from fdd_system.ML.common.config import SensorConfig
from fdd_system.broker.io_helpers import AlertSender, SerialReader, WindowBuilder, parse_sample
from fdd_system.broker.prediction_utils import (
    apply_runtime_calibration,
    apply_runtime_gate_calibration,
    build_pipeline,
    collect_calibration_targets,
    format_label_counts,
    log_prediction_counts,
    record_predictions,
)

EXAMPLE_USAGE = """Examples:
  python -m fdd_system.broker.main --port /dev/ttyACM0 --baudrate 115200 --input-format bin --fs-hz 800 --model-path experiment/weights/end_to_end_cnn1d_hybrid_calaug.onnx --model-format onnx --embedder auto --preprocessor auto --anomaly-detector-path experiment/weights/end_to_end_anomaly_gate.pt --calibration-seconds 20 --calibration-discard-seconds 2 --gate-runtime-calibration
"""


class BrokerDataRecorder:
    """Append broker samples and predictions to a CSV file."""

    def __init__(self, path: str):
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        write_header = (not out_path.exists()) or out_path.stat().st_size == 0
        self._fh = out_path.open("a", encoding="utf-8", newline="")
        self._writer = csv.writer(self._fh)
        if write_header:
            self._writer.writerow(
                [
                    "row_type",
                    "ts",
                    "input_format",
                    "ax",
                    "ay",
                    "az",
                    "predictions",
                    "confidences",
                    "rejection_stage",
                    "rejection_reason",
                ]
            )
            self._fh.flush()

    def record_sample(self, *, input_format: str, ax: float, ay: float, az: float) -> None:
        self._writer.writerow(
            [
                "sample",
                f"{time.time():.6f}",
                input_format,
                f"{ax:.9f}",
                f"{ay:.9f}",
                f"{az:.9f}",
                "",
                "",
                "",
                "",
            ]
        )
        self._fh.flush()

    def record_prediction(
        self,
        *,
        input_format: str,
        preds,
        confs,
        rejection_stage=None,
        rejection_reason=None,
    ) -> None:
        preds_arr = np.asarray(preds).ravel()
        conf_arr = np.asarray(confs, dtype=float).ravel()
        stage_arr = np.asarray(rejection_stage, dtype=object).ravel() if rejection_stage is not None else np.array([])
        reason_arr = np.asarray(rejection_reason, dtype=object).ravel() if rejection_reason is not None else np.array([])

        preds_payload = [int(v) for v in preds_arr]
        conf_payload = [float(v) if np.isfinite(v) else None for v in conf_arr]
        stage_payload = [str(v) if v is not None else None for v in stage_arr]
        reason_payload = [str(v) if v is not None else None for v in reason_arr]

        self._writer.writerow(
            [
                "prediction",
                f"{time.time():.6f}",
                input_format,
                "",
                "",
                "",
                json.dumps(preds_payload),
                json.dumps(conf_payload),
                json.dumps(stage_payload),
                json.dumps(reason_payload),
            ]
        )
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()


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
    parser.add_argument(
        "--record-data-path",
        "-record-data-path",
        type=str,
        default=None,
        help="Optional CSV output path to record raw samples and model predictions.",
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
        recorder = BrokerDataRecorder(args.record_data_path)
        log.info("Recording samples and predictions to %s", args.record_data_path)

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
    calibration_targets = collect_calibration_targets(pipeline)
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

        apply_runtime_calibration(pipeline, calibration_windows)
        gate_summary = apply_runtime_gate_calibration(
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
                format_label_counts(gate_summary["nearest_label_counts"]),
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

        if recorder is not None:
            recorder.record_sample(input_format=args.input_format, ax=ax, ay=ay, az=az)

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
            predict_details = getattr(pipeline, "predict_details", None)
            if callable(predict_details):
                details = predict_details([window])
                preds = details["predictions"]
                confs = details["confidence"]
                if recorder is not None:
                    recorder.record_prediction(
                        input_format=args.input_format,
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
            else:
                preds, confs = pipeline.predict_with_confidence([window])
                if recorder is not None:
                    recorder.record_prediction(input_format=args.input_format, preds=preds, confs=confs)
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
