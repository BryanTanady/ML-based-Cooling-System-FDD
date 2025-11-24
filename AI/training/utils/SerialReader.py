import serial
import threading
import time
from serial.tools import list_ports

class SerialReader:
    def __init__(
        self,
        port: str = "COM3", # adjust as needed
        baudrate: int = 9600, # set the same as microcontroller
        timeout: float = 1.0,
        buffer=None
    ):

        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.buffer = buffer

        self._stop_flag = threading.Event()
        self._thread = None
        self._partial = b"" # store incomplete data between reads

        self.ser = serial.Serial(self.port, self.baudrate, timeout=self.timeout)

        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

    def _read_loop(self):
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
        self._partial += chunk

        while b'\n' in self._partial:
            line, self._partial = self._partial.split(b'\n', 1)
            line = line.rstrip(b'\r')
            text = line.decode('utf-8', errors='ignore')
            self.buffer.append(text)

    def stop(self):
        self._stop_flag.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)

        if self.ser and self.ser.is_open:
            self.ser.close()

    def __del__(self):
        self.stop()

    @staticmethod
    def list_devices():
        devices = []
        baudrates = list_ports.comports()
        for b in baudrates:
            devices.append(f"{b.device} - {b.description}")
        return devices