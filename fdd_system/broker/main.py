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
from collections import deque
from typing import Deque, Optional, Tuple
import joblib

import numpy as np

from fdd_system.broker.SerialReader import SerialReader
from fdd_system.ML.common.config.system import SensorConfig
from fdd_system.ML.common.config.data import RawAccWindow
from fdd_system.ML.common.config.operating_types import OperatingCondition

from fdd_system.ML.common.classification.embedder import MLEmbedder1
from fdd_system.ML.common.classification.inferrer import SklearnMLInferrer
from fdd_system.ML.common.classification.preprocessor import DummyPreprocessor
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
    parts = line.strip().split(",")
    if len(parts) < 3:
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
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO),
                         format="%(asctime)s %(levelname)s %(message)s")
    log = logging.getLogger("broker")

    # Shared buffer populated by SerialReader
    buffer: Deque[str] = deque()
    reader = SerialReader(port=args.port, baudrate=args.baudrate, timeout=args.timeout, buffer=buffer)
    window_builder = WindowBuilder(SensorConfig.WINDOW_SIZE)
    pipeline = build_pipeline(args.model_path)

    log.info("Broker started. Reading from %s @ %s baud", args.port, args.baudrate)
    try:
        while True:
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
        reader.stop()


if __name__ == "__main__":
    main()
