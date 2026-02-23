"""Broker CLI entrypoint and runtime orchestration.

Example usage:
    python -m fdd_system.broker.main \
      --port /dev/ttyACM0 \
      --baudrate 115200 \
      --model-path experiment/weights/model_best.joblib \
      --model-format sklearn \
      --embedder ml2 \
      --preprocessor basic

    python -m fdd_system.broker.main \
      --port /dev/ttyACM0 \
      --baudrate 115200 \
      --input-format bin \
      --fs-hz 800 \
      --model-path experiment/weights/model_best_cnn1d.onnx \
      --model-format onnx \
      --embedder raw1dcnn \
      --preprocessor rms
"""

from __future__ import annotations

import argparse
import logging
import time
from collections import Counter, deque
from typing import Deque

import serial

from data_collection.binary_protocol import ADXLBinaryParser
from fdd_system.ML.common.config.operating_types import OperatingCondition
from fdd_system.ML.common.config.system import SensorConfig
from fdd_system.broker.SerialReader import SerialReader
from fdd_system.broker.alerts import AlertSender
from fdd_system.broker.parsing import parse_sample
from fdd_system.broker.pipeline_factory import build_pipeline
from fdd_system.broker.predictions import record_predictions
from fdd_system.broker.windowing import WindowBuilder

EXAMPLE_USAGE = """Examples:
  python -m fdd_system.broker.main --port /dev/ttyACM0 --baudrate 115200 --model-path experiment/weights/model_best.joblib --model-format sklearn --embedder ml2 --preprocessor basic
  python -m fdd_system.broker.main --port /dev/ttyACM0 --baudrate 115200 --input-format bin --fs-hz 800 --model-path experiment/weights/model_best_cnn1d.onnx --model-format onnx --embedder raw1dcnn --preprocessor robust
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
        choices=["auto", "basic", "dummy", "robust", "median", "standard", "rms"],
        default="auto",
        help="Input preprocessor. Default is basic unless overridden by ONNX metadata.",
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
    )
    prediction_counts: Counter[int] = Counter()
    end_time = time.time() + args.run_seconds if args.run_seconds else None

    log.info(
        "Broker started. Reading from %s @ %s baud (format=%s, fs_hz=%.3f, alert_api=%s, asset_id=%s)",
        args.port,
        args.baudrate,
        args.input_format,
        float(args.fs_hz),
        args.alert_api_url,
        args.asset_id,
    )

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

                window = window_builder.add(*sample)
                if window:
                    preds, confs = pipeline.predict_with_confidence([window])
                    record_predictions(preds, confs, prediction_counts, alert_sender, log)
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
                window = window_builder.add(float(sample.x), float(sample.y), float(sample.z))
                if not window:
                    continue
                preds, confs = pipeline.predict_with_confidence([window])
                record_predictions(preds, confs, prediction_counts, alert_sender, log)

    except KeyboardInterrupt:
        log.info("Broker stopping (Ctrl+C).")
    finally:
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
