"""Record labeled accelerometer data from a serial device into a CSV file.

Usage example:
    python getData2.py --time 180 --label NORMAL_1.csv --port /dev/ttyUSB0 --baud 115200 --fs 800
"""

import argparse
import threading
import time
from pathlib import Path
from typing import Optional, Union

import pandas as pd
import serial
from tqdm import tqdm

DATA_DIR = Path(__file__).parent

from binary_protocol import ADXLBinaryParser, Protocol


class BinarySerialReader:
    """Background serial reader that parses binary frames into a shared buffer.

    Buffer entries are tuples: (idx, t_us, x, y, z)

    Supports two on-wire formats:
    - Compact 9-byte frames with sync + XYZ + CRC (no idx/timestamp; generated on host).
    - Legacy 18-byte frames with idx/t_us + CRC.
    """

    def __init__(
        self,
        *,
        port: str,
        baudrate: int,
        timeout: float,
        fs_hz: float,
        protocol: Protocol = "9",
        buffer: list[tuple[int, int, int, int, int]],
    ):
        self._ser = serial.Serial(port=port, baudrate=baudrate, timeout=timeout)
        self._ser.reset_input_buffer()

        self._buf = buffer
        self._parser = ADXLBinaryParser(protocol=protocol, fs_hz=fs_hz)

        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        try:
            self._thread.join(timeout=1.0)
        finally:
            try:
                self._ser.close()
            except Exception:
                pass

    def _run(self):
        while not self._stop.is_set():
            try:
                chunk = self._ser.read(4096)
            except Exception:
                # If the port dies, exit thread.
                return

            if not chunk:
                continue

            for s in self._parser.feed(chunk):
                self._buf.append((s.idx, s.t_us, s.x, s.y, s.z))


# ---------------- Legacy text parsing (kept for backward compatibility) ----------------
def _parse_line(line: str) -> list[float] | None:
    """Parse a single legacy text line into [seq, t_us, x, y, z]."""
    line = line.strip()
    if not line:
        return None

    if line.startswith("//"):
        line = line[2:].strip()

    parts = line.split()
    if not parts:
        return None

    if parts[0] != "S" or len(parts) < 5:
        return None

    try:
        t_us = float(parts[1])
        ax = float(parts[2])
        ay = float(parts[3])
        az = float(parts[4])
        return [None, t_us, ax, ay, az]
    except ValueError:
        return None


BufferEntry = Union[str, tuple[int, int, int, int, int]]


def _parse_buffer(raw: list[BufferEntry]) -> list[list[float]]:
    """Convert buffer entries to numeric rows [Seq, t_us, X, Y, Z].

    Supports:
    - New binary tuples: (idx, t_us, x, y, z)
    - Old legacy text lines
    """
    rows: list[list[float]] = []

    for item in raw:
        if isinstance(item, tuple) and len(item) == 5:
            idx, t_us, x, y, z = item
            rows.append([float(idx), float(t_us), float(x), float(y), float(z)])
        elif isinstance(item, str):
            r = _parse_line(item)
            if r is not None:
                rows.append([float(r[0]) if r[0] is not None else float("nan"), float(r[1]), float(r[2]), float(r[3]), float(r[4])])

    return rows


def _latest_xyz(raw: list[BufferEntry]) -> Optional[list[float]]:
    """Return the most recent XYZ triple from the buffer."""
    for item in reversed(raw):
        if isinstance(item, tuple) and len(item) == 5:
            _, _, x, y, z = item
            return [float(x), float(y), float(z)]
        if isinstance(item, str):
            row = _parse_line(item)
            if row:
                return [float(row[-3]), float(row[-2]), float(row[-1])]
    return None


def _output_path(label: str) -> Path:
    path = Path(label)
    if path.suffix.lower() != ".csv":
        path = path.with_suffix(".csv")
    if not path.parent or str(path.parent) == ".":
        path = DATA_DIR / path.name
    return path


def record(duration: float, label: str, port: str, baudrate: int, timeout: float, fs_hz: float, protocol: str = "9"):
    """Collect data for the given duration and write it to the specified CSV file."""
    buffer: list[BufferEntry] = []

    # Arduino uses binary frames; we support both 9-byte and legacy 18-byte variants.
    reader = BinarySerialReader(
        port=port,
        baudrate=baudrate,
        timeout=timeout,
        fs_hz=fs_hz,
        protocol=protocol,
        buffer=buffer,  # type: ignore[arg-type]
    )

    try:
        start = time.monotonic()
        last_print = start
        with tqdm(total=duration, unit="s", desc="Recording", leave=False) as bar:
            while True:
                elapsed = time.monotonic() - start
                if elapsed >= duration:
                    bar.update(max(0.0, duration - bar.n))
                    break

                if time.monotonic() - last_print >= 5.0:
                    latest = _latest_xyz(buffer)
                    if latest:
                        print(f"{elapsed:6.2f}s latest sample X={latest[0]:.0f}, Y={latest[1]:.0f}, Z={latest[2]:.0f}")
                    else:
                        print(f"{elapsed:6.2f}s waiting for data...")
                    last_print = time.monotonic()

                time.sleep(0.1)
                bar.update(max(0.0, elapsed - bar.n))
    finally:
        reader.stop()

    rows = _parse_buffer(buffer)
    df = pd.DataFrame(rows, columns=["idx", "t_us", "X", "Y", "Z"])

    # If legacy text format produced NaN idx, fill with range
    if df["idx"].isnull().any():
        seq = pd.to_numeric(df["idx"], errors="coerce")
        seq_filled = seq.where(~seq.isna(), pd.Series(range(len(df)), index=df.index))
        df["idx"] = seq_filled

    # Keep idx and t_us as integers if possible (nice for storage)
    df["idx"] = pd.to_numeric(df["idx"], errors="coerce").astype("Int64")
    df["t_us"] = pd.to_numeric(df["t_us"], errors="coerce").astype("Int64")
    # Human-friendly seconds column (matches experiment/data_11/*.csv convention)
    df["t_s"] = (df["t_us"].astype("float64") / 1_000_000.0)

    output_path = _output_path(label)
    df.to_csv(output_path, index=False)
    return output_path, len(df)


def main():
    parser = argparse.ArgumentParser(description="Record labeled accelerometer data.")
    parser.add_argument("--time", type=float, required=True, help="Duration to record (seconds).")
    parser.add_argument(
        "--label",
        type=str,
        required=True,
        help="Output filename (e.g., NORMAL_1.csv). '.csv' is added if omitted.",
    )
    parser.add_argument("--port", type=str, default="/dev/ttyUSB0", help="Serial port (e.g., /dev/ttyUSB0, COM3).")
    parser.add_argument("--baud", type=int, default=115_200, help="Serial baudrate.")
    parser.add_argument("--fs", type=float, default=800.0, help="Sampling rate used for synthetic timestamps (Hz).")
    parser.add_argument("--proto", choices=["9", "18", "auto"], default="9", help="Binary protocol version to decode.")
    parser.add_argument("--timeout", type=float, default=1.0, help="Serial read timeout (seconds).")

    args = parser.parse_args()
    label = args.label.strip()
    if not label:
        raise SystemExit("Label must be non-empty.")

    try:
        output_path, count = record(
            duration=args.time,
            label=label,
            port=args.port,
            baudrate=args.baud,
            timeout=args.timeout,
            fs_hz=args.fs,
            protocol=args.proto,
        )
    except KeyboardInterrupt:
        raise SystemExit("Recording cancelled by user.")

    print(f"Wrote {count} rows to {output_path}")


if __name__ == "__main__":
    main()
