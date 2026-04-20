"""Broker CLI entrypoint and runtime orchestration.

Example usage:
    python -m fdd_system.broker.main \
      --port /dev/ttyACM0 \
      --baudrate 115200 \
      --input-format bin \
      --fs-hz 800 \
      --model-path experiment/weights/end_to_end_cnn1d_hybrid.pt \
      --model-format torch \
      --embedder auto \
      --preprocessor auto \
      --anomaly-detector-path experiment/weights/end_to_end_anomaly_gate.pt
"""

from __future__ import annotations

import argparse
import csv
import logging
import time
from collections import Counter, deque
from pathlib import Path
from typing import Deque

import numpy as np
import serial

from data_collection.binary_protocol import ADXLBinaryParser
from fdd_system.ML.components.detector import Stage0WindowGuard
from fdd_system.ML.schema import OperatingCondition, SensorConfig
from fdd_system.ML.pipeline import KnownUnknownClassificationPipeline, NormalityFaultClassificationPipeline
from fdd_system.broker.io_helpers import AlertSender, SerialReader, WindowBuilder, parse_sample
from fdd_system.broker.prediction_utils import (
    build_pipeline,
    log_live_debug_stats,
    log_prediction_counts,
    record_predictions,
)

EXAMPLE_USAGE = """Examples:
  python -m fdd_system.broker.main --port /dev/ttyACM0 --baudrate 115200 --input-format bin --fs-hz 800 --model-path experiment/weights/end_to_end_cnn1d_hybrid.pt --model-format torch --embedder auto --preprocessor auto --anomaly-detector-path experiment/weights/end_to_end_anomaly_gate.pt
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
            "Input preprocessor. Default is basic unless overridden by model metadata. "
            "'centered_rms' subtracts per-axis window bias before RMS scaling."
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
        bin_parser = ADXLBinaryParser(fs_hz=args.fs_hz)

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

    prediction_counts: Counter[int] = Counter()
    end_time = time.time() + args.run_seconds if args.run_seconds else None

    log.info(
        (
            "Broker started. Reading from %s @ %s baud (format=%s, fs_hz=%.3f, alert_api=%s, asset_id=%s, "
            "pipeline=%s, normality_detector=%s, anomaly_detector=%s, stage0_validator=%s, debug_live_stats=%s)"
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
        "enabled" if bool(args.debug_live_stats) else "disabled",
    )

    def handle_sample(ax: float, ay: float, az: float, *, idx: int | None = None, t_us: int | None = None) -> None:
        if recorder is not None:
            recorder.record_sample(ax=ax, ay=ay, az=az, idx=idx, t_us=t_us)

        window = window_builder.add(ax, ay, az)
        if not window:
            return

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
            return

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
