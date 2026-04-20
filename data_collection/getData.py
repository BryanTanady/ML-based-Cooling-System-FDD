"""Record labeled accelerometer data from a serial stream into CSV.

Example:
    python data_collection/getData.py \
        --time 180 \
        --label normal_1.csv \
        --port /dev/ttyUSB0 \
        --baud 115200 \
        --fs 800
"""

from __future__ import annotations

import argparse
import csv
import logging
import threading
import time
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol

import serial
from serial import SerialException

try:
    from tqdm import tqdm
except ModuleNotFoundError:
    tqdm = None

try:
    from .binary_protocol import ADXLBinaryParser, Sample
except ImportError:
    from binary_protocol import ADXLBinaryParser, Sample

LOGGER = logging.getLogger(__name__)
DATA_DIR = Path(__file__).parent
CSV_COLUMNS = ("idx", "t_us", "X", "Y", "Z")

BufferEntry = tuple[int, int, int, int, int]


class _ProgressBar(Protocol):
    n: float

    def update(self, n: float = 1.0) -> None:
        ...


class _NoopProgress:
    """Minimal tqdm-compatible fallback when tqdm is unavailable."""

    def __init__(self, total: float) -> None:
        self.total = max(0.0, total)
        self.n = 0.0

    def __enter__(self) -> "_NoopProgress":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # type: ignore[override]
        return None

    def update(self, n: float = 1.0) -> None:
        if n <= 0:
            return
        self.n = min(self.total, self.n + n)


@dataclass(frozen=True, slots=True)
class RecordConfig:
    duration_s: float
    label: str
    port: str
    baudrate: int
    timeout_s: float
    fs_hz: float
    status_interval_s: float = 5.0

    def validate(self) -> None:
        if self.duration_s <= 0:
            raise ValueError("--time must be positive")
        if not self.label.strip():
            raise ValueError("--label must be non-empty")
        if not self.port.strip():
            raise ValueError("--port must be non-empty")
        if self.baudrate <= 0:
            raise ValueError("--baud must be positive")
        if self.timeout_s <= 0:
            raise ValueError("--timeout must be positive")
        if self.fs_hz <= 0:
            raise ValueError("--fs must be positive")
        if self.status_interval_s <= 0:
            raise ValueError("--status-interval must be positive")


class _SampleStore:
    """Thread-safe store for decoded samples."""

    def __init__(self) -> None:
        self._items: list[BufferEntry] = []
        self._lock = threading.Lock()

    def append(self, sample: Sample) -> None:
        entry = (sample.idx, sample.t_us, sample.x, sample.y, sample.z)
        with self._lock:
            self._items.append(entry)

    def snapshot(self) -> list[BufferEntry]:
        with self._lock:
            return list(self._items)

    def latest_xyz(self) -> tuple[int, int, int] | None:
        with self._lock:
            if not self._items:
                return None
            _, _, x, y, z = self._items[-1]
            return x, y, z


class BinarySerialReader:
    """Background serial reader that parses binary frames into a sample sink."""

    def __init__(
        self,
        *,
        port: str,
        baudrate: int,
        timeout_s: float,
        fs_hz: float,
        on_sample: Callable[[Sample], None],
    ) -> None:
        self._ser = serial.serial_for_url(port, baudrate=baudrate, timeout=timeout_s)
        self._on_sample = on_sample
        self._parser = ADXLBinaryParser(fs_hz=fs_hz)

        with suppress(Exception):
            self._ser.reset_input_buffer()

        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            name=f"BinarySerialReader[{port}]",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)

        with suppress(Exception):
            self._ser.close()

    def __enter__(self) -> "BinarySerialReader":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # type: ignore[override]
        self.stop()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                chunk = self._ser.read(4096)
            except (SerialException, OSError) as exc:
                if not self._stop.is_set():
                    LOGGER.error("serial read failed on %s: %s", self._ser.port, exc)
                return

            if not chunk:
                continue

            for sample in self._parser.feed(chunk):
                self._on_sample(sample)


def _progress_bar(total: float) -> _ProgressBar:
    if tqdm is None:
        return _NoopProgress(total)
    return tqdm(total=total, unit="s", desc="Recording", leave=False)


def _output_path(label: str) -> Path:
    path = Path(label)
    if path.suffix.lower() != ".csv":
        path = path.with_suffix(".csv")

    if not path.parent or str(path.parent) == ".":
        path = DATA_DIR / path.name

    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _write_rows(output_path: Path, rows: list[BufferEntry]) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([*CSV_COLUMNS, "t_s"])
        for idx, t_us, x, y, z in rows:
            writer.writerow([idx, t_us, x, y, z, t_us / 1_000_000.0])


def record(config: RecordConfig) -> tuple[Path, int]:
    """Collect samples for `config.duration_s` and write them to CSV."""
    config.validate()

    store = _SampleStore()
    with BinarySerialReader(
        port=config.port,
        baudrate=config.baudrate,
        timeout_s=config.timeout_s,
        fs_hz=config.fs_hz,
        on_sample=store.append,
    ):
        start = time.monotonic()
        next_status = start + config.status_interval_s

        with _progress_bar(config.duration_s) as bar:
            while True:
                now = time.monotonic()
                elapsed = now - start
                progress = min(config.duration_s, elapsed)

                if progress > bar.n:
                    bar.update(progress - bar.n)

                if elapsed >= config.duration_s:
                    break

                if now >= next_status:
                    latest = store.latest_xyz()
                    if latest is None:
                        LOGGER.info("%6.2fs waiting for data...", elapsed)
                    else:
                        x, y, z = latest
                        LOGGER.info(
                            "%6.2fs latest sample X=%d, Y=%d, Z=%d",
                            elapsed,
                            x,
                            y,
                            z,
                        )
                    next_status = now + config.status_interval_s

                time.sleep(0.1)

    rows = store.snapshot()
    output_path = _output_path(config.label)
    _write_rows(output_path, rows)
    return output_path, len(rows)


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be > 0")
    return parsed


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be > 0")
    return parsed


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Record labeled accelerometer data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--time",
        type=_positive_float,
        required=True,
        help="Duration to record in seconds.",
    )
    parser.add_argument(
        "--label",
        type=str,
        required=True,
        help="Output filename or path. '.csv' is added if omitted.",
    )
    parser.add_argument(
        "--port",
        type=str,
        default="/dev/ttyUSB0",
        help="Serial port path or pyserial URL.",
    )
    parser.add_argument(
        "--baud",
        type=_positive_int,
        default=115_200,
        help="Serial baudrate.",
    )
    parser.add_argument(
        "--fs",
        type=_positive_float,
        default=800.0,
        help="Sampling rate in Hz used for synthetic timestamps.",
    )
    parser.add_argument(
        "--timeout",
        type=_positive_float,
        default=1.0,
        help="Serial read timeout in seconds.",
    )
    parser.add_argument(
        "--status-interval",
        type=_positive_float,
        default=5.0,
        help="Status log interval in seconds.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
        help="Logging verbosity.",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s: %(message)s",
    )

    config = RecordConfig(
        duration_s=args.time,
        label=args.label.strip(),
        port=args.port.strip(),
        baudrate=args.baud,
        timeout_s=args.timeout,
        fs_hz=args.fs,
        status_interval_s=args.status_interval,
    )

    try:
        output_path, count = record(config)
    except KeyboardInterrupt:
        raise SystemExit("Recording cancelled by user.")
    except (ValueError, SerialException, OSError) as exc:
        raise SystemExit(str(exc))

    print(f"Wrote {count} rows to {output_path}")


if __name__ == "__main__":
    main()
