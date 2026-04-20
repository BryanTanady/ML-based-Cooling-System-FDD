"""Threaded newline-based serial reader.

This helper continuously reads bytes from a serial device (or a pyserial URL),
reconstructs newline-delimited UTF-8 text lines, and appends decoded lines into
a caller-provided mutable buffer.
"""

from __future__ import annotations

import ipaddress
import logging
import threading
from collections.abc import MutableSequence
from contextlib import suppress

import serial
from serial import SerialException
from serial.tools import list_ports

LOGGER = logging.getLogger(__name__)
DEFAULT_NETWORK_PORT = 3240


class SerialReader:
    """Continuously read lines from a serial endpoint on a background thread."""

    def __init__(
        self,
        port: str = "COM3",
        baudrate: int = 9600,
        timeout: float = 1.0,
        buffer: MutableSequence[str] | None = None,
        network_port: int | None = None,
        network_protocol: str = "socket",
    ) -> None:
        if not port:
            raise ValueError("port must be non-empty")
        if baudrate <= 0:
            raise ValueError("baudrate must be positive")
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        if network_port is not None and network_port <= 0:
            raise ValueError("network_port must be positive when provided")

        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.network_port = network_port
        self.network_protocol = network_protocol

        self.buffer: MutableSequence[str] = buffer if buffer is not None else []
        self._buffer_lock = threading.Lock()

        self._stop_flag = threading.Event()
        self._thread: threading.Thread | None = None
        self._partial = bytearray()

        target = self._resolve_port_target()
        self.ser = serial.serial_for_url(
            target,
            baudrate=self.baudrate,
            timeout=self.timeout,
        )

        self._thread = threading.Thread(
            target=self._read_loop,
            name=f"SerialReader[{target}]",
            daemon=True,
        )
        self._thread.start()

    def _resolve_port_target(self) -> str:
        """Resolve a port string into a serial path or pyserial URL."""
        if "://" in self.port:
            return self.port

        if self._looks_like_host(self.port):
            host, explicit_port = self._split_host_and_port(self.port)
            tcp_port = explicit_port or self.network_port or DEFAULT_NETWORK_PORT
            return f"{self.network_protocol}://{host}:{tcp_port}"

        return self.port

    @staticmethod
    def _split_host_and_port(value: str) -> tuple[str, int | None]:
        """Split IPv4 hostname:port forms; return (host, None) when no explicit port."""
        if value.count(":") == 1:
            host, maybe_port = value.rsplit(":", 1)
            if maybe_port.isdigit():
                return host, int(maybe_port)
        return value, None

    @staticmethod
    def _looks_like_host(value: str) -> bool:
        """Best-effort host/IP detection for non-URL serial targets."""
        if not value:
            return False

        upper = value.upper()
        if "/" in value or upper.startswith("COM") or value.startswith("tty"):
            return False

        host, _ = SerialReader._split_host_and_port(value)
        if host.lower() == "localhost":
            return True

        try:
            ipaddress.ip_address(host)
            return True
        except ValueError:
            return "." in host

    def _append_line(self, line: str) -> None:
        with self._buffer_lock:
            self.buffer.append(line)

    def _read_loop(self) -> None:
        while not self._stop_flag.is_set():
            try:
                waiting = getattr(self.ser, "in_waiting", 0)
                chunk = self.ser.read(waiting or 1)
            except (SerialException, OSError) as exc:
                if not self._stop_flag.is_set():
                    LOGGER.error("serial read failed on %s: %s", self.port, exc)
                break

            if not chunk:
                continue
            self._process_chunk(chunk)

    def _process_chunk(self, chunk: bytes) -> None:
        """Split incoming bytes on newline and append decoded lines to the buffer."""
        self._partial.extend(chunk)

        while True:
            newline_index = self._partial.find(b"\n")
            if newline_index < 0:
                return

            line = bytes(self._partial[:newline_index]).rstrip(b"\r")
            del self._partial[: newline_index + 1]
            text = line.decode("utf-8", errors="replace")
            self._append_line(text)

    def read_buffer_snapshot(self) -> list[str]:
        """Return a point-in-time copy of the current line buffer."""
        with self._buffer_lock:
            return list(self.buffer)

    def stop(self) -> None:
        """Stop the reader thread and close the serial handle."""
        self._stop_flag.set()

        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1.0)

        with suppress(Exception):
            if self.ser and self.ser.is_open:
                self.ser.close()

    def __enter__(self) -> "SerialReader":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # type: ignore[override]
        self.stop()

    def __del__(self) -> None:
        with suppress(Exception):
            self.stop()

    @staticmethod
    def list_devices() -> list[str]:
        """Return available local serial ports with descriptions."""
        ports = sorted(list_ports.comports(), key=lambda p: p.device)
        return [f"{port.device} - {port.description}" for port in ports]
