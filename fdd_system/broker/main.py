"""Minimal broker CLI that reads serial data, builds windows, and calls the AI pipeline.

Example:

    python3 -m fdd_system.broker.main \
        --port /dev/ttyACM0 \
        --baudrate 9600 \
        --model-path fdd_system/AI/training/ML/weights/rf_model.joblib \
        --loop-delay 0.05
"""

import argparse
import json
import logging
import time
from collections import Counter, deque
import re
from typing import Deque, Optional, Tuple
import joblib

import numpy as np

from fdd_system.broker.SerialReader import SerialReader
from fdd_system.ML.common.config.system import SensorConfig
from fdd_system.ML.common.config.data import RawAccWindow
from fdd_system.ML.common.config.operating_types import OperatingCondition

from fdd_system.ML.common.classification.embedder import MLEmbedder1
from fdd_system.ML.common.classification.inferrer import SklearnMLInferrer
from fdd_system.ML.common.classification.preprocessor import DummyPreprocessor, BasicPreprocessor, RobustPreprocessor
from fdd_system.ML.inference.classification_pipeline import ClassificationPipeline
class WindowBuilder:
    """Accumulates accelerometer samples into RawAccWindow objects."""

    def __init__(self, window_size: int):
        """Initialize a window builder.

        Args:
            window_size: Number of samples per window (from SensorConfig).
        """
        self.window_size = window_size
        self.samples: Deque[Tuple[float, float, float]] = deque()

    def add(self, ax: float, ay: float, az: float) -> Optional[RawAccWindow]:
        """Add one accelerometer sample and emit a window when full.

        Args:
            ax: Acceleration on X axis.
            ay: Acceleration on Y axis.
            az: Acceleration on Z axis.

        Returns:
            RawAccWindow if a full window is available; otherwise None.
        """
        self.samples.append((ax, ay, az))
        if len(self.samples) < self.window_size:
            return None

        # Build window
        ax_arr, ay_arr, az_arr = (np.array(vals) for vals in zip(*list(self.samples)[: self.window_size]))


        # Slide forward by STRIDE
        for _ in range(SensorConfig.STRIDE):
            self.samples.popleft()
        return RawAccWindow(acc_x=ax_arr, acc_y=ay_arr, acc_z=az_arr)


def parse_sample(line: str) -> Optional[Tuple[float, float, float]]:
    """Parse a comma separate values (CSV) line 'ax, ay, az' into floats.

    Args:
        line: CSV text line from serial.

    Returns:
        Tuple (ax, ay, az) as floats, or None on failure.
    """
    # Handle log-prefixed lines like ".... Skipping unparsable line: S 123 1.0 2.0 3.0"
    if "Skipping unparsable line:" in line:
        line = line.split("Skipping unparsable line:", 1)[1].strip()

    parts = line.strip().split(",")
    if len(parts) < 3:
        # Try to parse verbose log-style lines like
        # "x 1.96 y 2.04 z 2.12"
        pattern = r"x\s+([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s+y\s+([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s+z\s+([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
        match = re.search(pattern, line)
        if match:
            try:
                ax, ay, az = (float(match.group(1)), float(match.group(2)), float(match.group(3)))
            except ValueError:
                return None
            return ax, ay, az

        # Parse new format lines like "S 11840048 2.354 -16.083 18.907"
        pattern_s = r"S\s+[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?\s+([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s+([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s+([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
        match_s = re.search(pattern_s, line)
        if match_s:
            try:
                ax = float(match_s.group(1))
                ay = float(match_s.group(2))
                az = float(match_s.group(3))
            except ValueError:
                return None
            return ax, ay, az
        return None
    try:
        ax, ay, az = (float(parts[0]), float(parts[1]), float(parts[2]))
    except ValueError:
        return None
    return ax, ay, az


def load_model(model_path: str):
    """Load a trained sklearn model from disk.

    Args:
        model_path: Filesystem path to a joblib/pkl model.

    Returns:
        Deserialized model object.
    """
    if not joblib:
        raise ImportError("joblib is required to load the model; install it or adjust load_model.")
    return joblib.load(model_path)


def build_pipeline(model_path: str) -> ClassificationPipeline:
    """Construct the end-to-end classification pipeline.

    Args:
        model_path: Filesystem path to the trained model.

    Returns:
        ClassificationPipeline wired with preprocessor, embedder, and inferrer.
    """
    model = load_model(model_path)
    pre = DummyPreprocessor()
    emb = MLEmbedder1()
    inf = SklearnMLInferrer(model)
    return ClassificationPipeline(pre, emb, inf)


def main():
    """CLI entry for the broker: read serial, infer, and log."""
    parser = argparse.ArgumentParser(description="Minimal Broker CLI (serial -> AI -> log)")
    parser.add_argument("--port", type=str, default="COM3",
                        help="Serial port (e.g., COM3, /dev/ttyACM0), or socket://127.0.0.1:9999")
    parser.add_argument("--baudrate", type=int, default=9600, help="Serial baudrate")
    parser.add_argument("--timeout", type=float, default=1.0, help="Serial timeout (seconds)")
    parser.add_argument("--model-path", type=str, required=True, help="Path to trained model (joblib/pkl)")
    parser.add_argument("--loop-delay", type=float, default=0.05, help="Sleep between loop iterations (seconds)")
    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level")
    parser.add_argument("--run-seconds", type=float, default=None,
                        help="How long to run before exiting. If not set, runs indefinitely.")
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO),
                         format="%(asctime)s %(levelname)s %(message)s")
    log = logging.getLogger("broker")

    # Shared buffer populated by SerialReader
    buffer: Deque[str] = deque()
    reader = SerialReader(port=args.port, baudrate=args.baudrate, timeout=args.timeout, buffer=buffer)
    window_builder = WindowBuilder(SensorConfig.WINDOW_SIZE)
    pipeline = build_pipeline(args.model_path)
    prediction_counts: Counter[int] = Counter()

    end_time = time.time() + args.run_seconds if args.run_seconds else None

    log.info("Broker started. Reading from %s @ %s baud", args.port, args.baudrate)
    try:
        while True:
            if end_time and time.time() >= end_time:
                log.info("Run duration reached; stopping.")
                break
            if buffer:
                line = buffer.popleft()     # read from the buffered data sent from microcontroller
                sample = parse_sample(line) # read the raw buffered data, to 3 floats 
                if not sample:
                    log.info("Skipping unparsable line: %s", line.strip())
                    continue
                
                # build "window", which is our representation of singleton data understood by ML
                # We get RawAccWindow once the window is full, otherwise None
                window = window_builder.add(*sample)
                length = len(window_builder.samples)
                if window:
                    preds = pipeline.predict([window])
                    # log the prediction
                    conditions = [OperatingCondition(n).name for n in preds]
                    log.info("Prediction: %s, %s", preds.tolist(), conditions)
                    for p in preds:
                        prediction_counts[int(p)] += 1

                    # TODO: 
                    # 1. Send prediction to interface's API
                    #   - Need to define what data is sent
                    #   - Definitely need some confidence score
                    #  
            else:
                time.sleep(args.loop_delay)
    except KeyboardInterrupt:
        log.info("Broker stopping (Ctrl+C).")
    finally:
        if prediction_counts:
            log.info("Prediction counts:")
            for cls_id, count in prediction_counts.items():
                try:
                    cls_name = OperatingCondition(cls_id).name
                except ValueError:
                    cls_name = f"Unknown({cls_id})"
                log.info("  %s: %s", cls_name, count)
        reader.stop()


if __name__ == "__main__":
    main()
