"""Simple serial line reader that pushes newline-delimited strings into a buffer.

Designed to run on a background thread, reading from a microcontroller and
appending complete lines to a user-provided buffer (e.g., list or deque).
"""

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
    ):
        """Start a background reader thread.

        Args:
            port: Serial device path (e.g., COM3, /dev/ttyACM0).
            baudrate: Baud rate matching the microcontroller.
            timeout: Serial read timeout in seconds.
            buffer: Mutable sequence with .append(str); receives decoded lines.
        """
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.buffer = buffer

        self._stop_flag = threading.Event()
        self._thread = None
        self._partial = b""  # store incomplete data between reads

        self.ser = serial.Serial(self.port, self.baudrate, timeout=self.timeout)

        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

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
