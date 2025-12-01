"""Record labeled accelerometer data from a serial device into a CSV file.

Usage example:
    python getData.py --time 180 --label NORMAL_1.csv --port /dev/ttyUSB0 --baud 115200
"""

import argparse
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from SerialReader import SerialReader

DATA_DIR = Path(__file__).parent


def _parse_line(line: str) -> list[float] | None:
    """Parse a single line into [seq, t_us, x, y, z].

    Expected format (current):
        // S <t_us_since_first_sample> <ax> <ay> <az>

    Legacy lines are ignored to avoid empty timestamp columns.
    """
    line = line.strip()
    if not line:
        return None

    # Strip leading comment slashes, then split
    if line.startswith("//"):
        line = line[2:].strip()

    parts = line.split()
    if not parts:
        return None

    # Expected format: S t_us ax ay az
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

    return None


def _parse_buffer(raw_lines: list[str]) -> list[list[float]]:
    """Extract numeric rows from buffered serial lines."""
    parsed: list[list[float]] = []
    for line in raw_lines:
        row = _parse_line(line)
        if row is not None:
            parsed.append(row)
    return parsed


def _latest_xyz(raw_lines: list[str]) -> list[float] | None:
    """Return the most recent parsed XYZ triple from the raw buffer."""
    for line in reversed(raw_lines):
        row = _parse_line(line)
        if row:
            return row[-3:]
    return None


def _output_path(label: str) -> Path:
    """Return the output path inside DATA_DIR, ensuring a .csv suffix."""
    path = Path(label)
    if path.suffix.lower() != ".csv":
        path = path.with_suffix(".csv")
    # If only a bare filename was given, place it in DATA_DIR
    if not path.parent or str(path.parent) == ".":
        path = DATA_DIR / path.name
    return path


def record(duration: float, label: str, port: str, baudrate: int, timeout: float):
    """Collect data for the given duration and write it to the specified CSV file."""
    buffer: list[str] = []
    reader = SerialReader(port=port, baudrate=baudrate, timeout=timeout, buffer=buffer)

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
                        print(f"{elapsed:6.2f}s latest sample X={latest[0]:.3f}, Y={latest[1]:.3f}, Z={latest[2]:.3f}")
                    else:
                        print(f"{elapsed:6.2f}s waiting for data...")
                    last_print = time.monotonic()

                time.sleep(0.1)
                bar.update(max(0.0, elapsed - bar.n))
    finally:
        reader.stop()

    rows = _parse_buffer(buffer)
    df = pd.DataFrame(rows, columns=["Seq", "t_us", "X", "Y", "Z"])

    # Fill missing sequence values (legacy format) with a simple range
    if df["Seq"].isnull().any():
        seq = pd.to_numeric(df["Seq"], errors="coerce")
        seq_filled = seq.where(~seq.isna(), pd.Series(range(len(df)), index=df.index))
        df["Seq"] = seq_filled

    output_path = _output_path(label)
    df.to_csv(output_path, index=False)
    return output_path, len(rows)


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
    parser.add_argument("--baud", type=int, default=250_000, help="Serial baudrate.")
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
        )
    except KeyboardInterrupt:
        raise SystemExit("Recording cancelled by user.")

    print(f"Wrote {count} rows to {output_path}")


if __name__ == "__main__":
    main()
