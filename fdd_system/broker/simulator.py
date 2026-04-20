"""ADXL345 Arduino binary stream simulator.

Default mode creates a virtual serial port (PTY) and streams protocol-9 frames
to it so existing scripts can use a normal `--port /tmp/ttyARDUINO` style path.

Optional mode can stream over TCP for `socket://...` consumers.
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import os
import pty
import socket
import struct
import time
from pathlib import Path

SYNC0 = 0xAA
SYNC1 = 0x55
I16_MIN = -(2**15)
I16_MAX = 2**15 - 1


def crc8_maxim(data: bytes) -> int:
    """Dallas/Maxim CRC-8 (poly 0x31), reversed (0x8C)."""
    crc = 0
    for byte in data:
        crc ^= byte
        for _ in range(8):
            if crc & 1:
                crc = (crc >> 1) ^ 0x8C
            else:
                crc >>= 1
    return crc & 0xFF


def _clip_i16(value: float) -> int:
    ivalue = int(round(value))
    if ivalue < I16_MIN:
        return I16_MIN
    if ivalue > I16_MAX:
        return I16_MAX
    return ivalue


def _encode_frame(x: float, y: float, z: float) -> bytes:
    payload = struct.pack("<hhh", _clip_i16(x), _clip_i16(y), _clip_i16(z))
    return bytes((SYNC0, SYNC1)) + payload + bytes((crc8_maxim(payload),))


def _resolve_columns(fieldnames: list[str]) -> tuple[str | None, str | None, str | None]:
    lowered = {name.strip().lower(): name for name in fieldnames}

    def pick(*candidates: str) -> str | None:
        for candidate in candidates:
            if candidate in lowered:
                return lowered[candidate]
        return None

    x_col = pick("x", "acc_x", "ax")
    y_col = pick("y", "acc_y", "ay")
    z_col = pick("z", "acc_z", "az")
    return x_col, y_col, z_col


def load_xyz_samples(csv_path: Path) -> list[tuple[float, float, float]]:
    """Load xyz samples from a CSV file."""
    with csv_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames:
            x_col, y_col, z_col = _resolve_columns(reader.fieldnames)
            if x_col and y_col and z_col:
                samples: list[tuple[float, float, float]] = []
                for row in reader:
                    try:
                        samples.append((float(row[x_col]), float(row[y_col]), float(row[z_col])))
                    except (TypeError, ValueError):
                        continue
                if samples:
                    return samples

    # Fallback for plain CSV rows with at least 3 columns.
    with csv_path.open("r", encoding="utf-8", newline="") as fh:
        rows = csv.reader(fh)
        samples = []
        for row in rows:
            if len(row) < 3:
                continue
            try:
                samples.append((float(row[0]), float(row[1]), float(row[2])))
            except ValueError:
                continue
        return samples


def synthetic_xyz(idx: int, fs_hz: float) -> tuple[float, float, float]:
    """Generate a stable vibration-like synthetic waveform."""
    t = idx / fs_hz
    carrier = 2.0 * math.pi * 52.0 * t
    drift = 2.0 * math.pi * 1.5 * t
    x = 45.0 * math.sin(carrier) + 6.0 * math.sin(drift)
    y = -410.0 + 35.0 * math.sin(carrier + 1.8) + 4.0 * math.sin(drift + 0.3)
    z = 510.0 + 18.0 * math.sin(carrier + 0.7)
    return x, y, z


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Simulate Arduino ADXL345 protocol-9 binary frames.")
    parser.add_argument(
        "--transport",
        choices=["pty", "tcp"],
        default="pty",
        help="Transport mode: virtual serial port (pty) or socket server (tcp).",
    )
    parser.add_argument("--fs-hz", type=float, default=800.0, help="Streaming sampling rate.")
    parser.add_argument(
        "--source-csv",
        type=Path,
        default=None,
        help="Optional CSV with X/Y/Z columns to replay instead of synthetic wave.",
    )
    parser.add_argument(
        "--loop",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When replaying CSV data, loop forever after the end.",
    )

    # PTY mode options
    parser.add_argument(
        "--pty-link",
        type=Path,
        default=Path("/tmp/ttyARDUINO"),
        help="Symlink path to expose the created PTY (used in --transport pty).",
    )

    # TCP mode options
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Bind address for TCP mode.")
    parser.add_argument("--port", type=int, default=9999, help="Bind port for TCP mode.")

    parser.add_argument("--log-level", type=str, default="INFO", help="Logging level.")
    return parser


def _build_sample_source(
    *,
    samples: list[tuple[float, float, float]] | None,
    fs_hz: float,
    loop: bool,
):
    sample_idx = 0
    while True:
        if samples:
            if sample_idx >= len(samples):
                if not loop:
                    return
                sample_idx = 0
            xyz = samples[sample_idx]
        else:
            xyz = synthetic_xyz(sample_idx, fs_hz)
        sample_idx += 1
        yield xyz


def _stream_arduino_like_loop(
    writer,
    *,
    fs_hz: float,
    samples: list[tuple[float, float, float]] | None,
    loop: bool,
    log: logging.Logger,
) -> None:
    # Mirror fdd_system/microcontroller/arduino.ino scheduler semantics:
    # - next sample timestamp initialized on first loop
    # - if late by >= 1 interval, resync to now + interval
    # - emit at most one frame per loop iteration
    interval_us = int(round(1_000_000.0 / fs_hz))
    sent = 0
    next_sample_us: int | None = None
    source = _build_sample_source(samples=samples, fs_hz=fs_hz, loop=loop)

    while True:
        now_us = time.monotonic_ns() // 1_000
        if next_sample_us is None:
            next_sample_us = now_us + interval_us

        if now_us - next_sample_us >= interval_us:
            next_sample_us = now_us + interval_us

        if now_us - next_sample_us >= 0:
            next_sample_us += interval_us
            try:
                x, y, z = next(source)
            except StopIteration:
                return

            writer(_encode_frame(x, y, z))
            sent += 1
            if sent > 0 and sent % int(max(1.0, fs_hz * 5.0)) == 0:
                log.info("Streamed %d samples.", sent)
            continue

        sleep_us = next_sample_us - now_us
        if sleep_us > 0:
            # Keep sleep short to preserve scheduler responsiveness.
            time.sleep(min(sleep_us / 1_000_000.0, 0.005))


def _run_pty_mode(
    *,
    pty_link: Path,
    fs_hz: float,
    samples: list[tuple[float, float, float]] | None,
    loop: bool,
    log: logging.Logger,
) -> None:
    master_fd, slave_fd = pty.openpty()
    slave_path = Path(os.ttyname(slave_fd))

    pty_link.parent.mkdir(parents=True, exist_ok=True)
    if pty_link.exists() or pty_link.is_symlink():
        if pty_link.is_dir():
            raise SystemExit(f"--pty-link points to a directory: {pty_link}")
        pty_link.unlink()
    os.symlink(str(slave_path), str(pty_link))

    log.info("PTY created: %s", slave_path)
    log.info("Serial path to use: %s", pty_link)
    log.info("Example: sh data_collection/run_getData.sh --time 10 --label test.csv --port %s --baud 115200 --fs 800", pty_link)

    def writer(frame: bytes) -> None:
        while True:
            try:
                os.write(master_fd, frame)
                return
            except InterruptedError:
                continue
            except OSError as exc:
                log.info("PTY write error: %s", exc)
                time.sleep(0.01)
                return

    try:
        _stream_arduino_like_loop(writer, fs_hz=fs_hz, samples=samples, loop=loop, log=log)
    finally:
        try:
            os.close(master_fd)
        except OSError:
            pass
        try:
            os.close(slave_fd)
        except OSError:
            pass
        if pty_link.is_symlink() and pty_link.resolve(strict=False) == slave_path:
            pty_link.unlink(missing_ok=True)


def _run_tcp_mode(
    *,
    host: str,
    port: int,
    fs_hz: float,
    samples: list[tuple[float, float, float]] | None,
    loop: bool,
    log: logging.Logger,
) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((host, port))
        server.listen(1)

        log.info("TCP simulator listening on %s:%d", host, port)
        log.info("Serial URL to use: socket://%s:%d", host, port)

        while True:
            conn, addr = server.accept()
            with conn:
                conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                log.info("Client connected from %s:%s", addr[0], addr[1])

                def writer(frame: bytes) -> None:
                    conn.sendall(frame)

                try:
                    _stream_arduino_like_loop(writer, fs_hz=fs_hz, samples=samples, loop=loop, log=log)
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError) as exc:
                    log.info("Client disconnected: %s", exc)


def main() -> int:
    args = build_arg_parser().parse_args()

    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    log = logging.getLogger("arduino-sim")

    if args.fs_hz <= 0:
        raise SystemExit("--fs-hz must be > 0.")

    samples: list[tuple[float, float, float]] | None = None
    if args.source_csv is not None:
        if not args.source_csv.exists():
            raise SystemExit(f"CSV not found: {args.source_csv}")
        samples = load_xyz_samples(args.source_csv)
        if not samples:
            raise SystemExit(f"No valid XYZ rows found in {args.source_csv}")
        log.info("Loaded %d samples from %s", len(samples), args.source_csv)
    else:
        log.info("Using synthetic waveform source.")

    if args.transport == "pty":
        _run_pty_mode(
            pty_link=args.pty_link,
            fs_hz=float(args.fs_hz),
            samples=samples,
            loop=bool(args.loop),
            log=log,
        )
        return 0

    _run_tcp_mode(
        host=args.host,
        port=int(args.port),
        fs_hz=float(args.fs_hz),
        samples=samples,
        loop=bool(args.loop),
        log=log,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
