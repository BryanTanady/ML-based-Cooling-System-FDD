"""Simple serial line reader that pushes newline-delimited strings into a buffer.

Designed to run on a background thread, reading from a microcontroller and
appending complete lines to a user-provided buffer (e.g., list or deque).
"""

import ipaddress
import serial
import threading
import time
from serial.tools import list_ports


class SerialReader:
    """Continuously reads bytes from a serial port and emits newline-terminated lines."""

    def __init__(
        self,
        port: str = "COM3",  # adjust as needed
        baudrate: int = 9600,  # set the same as microcontroller
        timeout: float = 1.0,
        buffer=None,
        network_port: int | None = None,
        network_protocol: str = "socket",
    ):
        """Start a background reader thread.

        Args:
            port: Serial device path (e.g., COM3, /dev/ttyACM0) or host/IP.
            baudrate: Baud rate matching the microcontroller.
            timeout: Serial read timeout in seconds.
            buffer: Mutable sequence with .append(str); receives decoded lines.
            network_port: TCP port to use when `port` is a host/IP (e.g., when
                connecting through usbipd/WSL or ser2net). Defaults to 3240 when
                not provided.
            network_protocol: pyserial URL scheme to use for host/IP targets,
                e.g., "socket" or "rfc2217".
        """
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.buffer = buffer
        self.network_port = network_port
        self.network_protocol = network_protocol

        self._stop_flag = threading.Event()
        self._thread = None
        self._partial = b""  # store incomplete data between reads

        target = self._resolve_port_target()
        self.ser = serial.serial_for_url(
            target, baudrate=self.baudrate, timeout=self.timeout
        )

        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

    def _resolve_port_target(self) -> str:
        """Return a pyserial URL for local ports or host/IP endpoints."""
        if "://" in self.port:
            return self.port  # already a pyserial URL

        if self._looks_like_host(self.port):
            tcp_port = self.network_port or 3240  # usbipd default port
            return f"{self.network_protocol}://{self.port}:{tcp_port}"

        return self.port

    @staticmethod
    def _looks_like_host(value: str) -> bool:
        """Heuristically decide if the given port string is a host/IP."""
        if "/" in value or value.upper().startswith("COM"):
            return False
        host = value.split(":", 1)[0]
        try:
            ipaddress.ip_address(host)
            return True
        except ValueError:
            return False

    def _read_loop(self):
        """Poll the serial port and accumulate bytes until stopped."""
        while not self._stop_flag.is_set():
            try:
                chunk = self.ser.read(self.ser.in_waiting or 1)
                if chunk:
                    self._process_chunk(chunk)
                else:
                    print("No data received.")
            except Exception:
                print("Error reading from serial port.")

    def _process_chunk(self, chunk: bytes):
        """Split incoming bytes on newline and append decoded lines to buffer."""
        self._partial += chunk

        while b"\n" in self._partial:
            line, self._partial = self._partial.split(b"\n", 1)
            line = line.rstrip(b"\r")
            text = line.decode("utf-8", errors="ignore")
            self.buffer.append(text)

    def stop(self):
        """Stop the reader thread and close the serial port."""
        self._stop_flag.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)

        if self.ser and self.ser.is_open:
            self.ser.close()

    def __del__(self):
        self.stop()

    @staticmethod
    def list_devices():
        """Return a list of available serial ports with descriptions."""
        devices = []
        baudrates = list_ports.comports()
        for b in baudrates:
            devices.append(f"{b.device} - {b.description}")
        return devices
