"""Binary frame decoder for ADXL345 protocol-9 streams.

Frame layout (9 bytes):
  [0]   0xAA
  [1]   0x55
  [2:4] X (int16, little-endian)
  [4:6] Y (int16, little-endian)
  [6:8] Z (int16, little-endian)
  [8]   CRC-8 (Dallas/Maxim) over bytes [2:8)
"""

from __future__ import annotations

from dataclasses import dataclass

SYNC0 = 0xAA
SYNC1 = 0x55
FRAME9_LEN = 9
PAYLOAD_SLICE = slice(2, 8)
CRC_INDEX = 8


def _i16_le(data: bytes) -> int:
    return int.from_bytes(data, byteorder="little", signed=True)


def crc8_maxim(data: bytes) -> int:
    """Return Dallas/Maxim CRC-8 for ``data``."""
    crc = 0
    for byte in data:
        crc ^= byte
        for _ in range(8):
            crc = (crc >> 1) ^ 0x8C if (crc & 1) else (crc >> 1)
    return crc & 0xFF


@dataclass(frozen=True, slots=True)
class Sample:
    idx: int
    t_us: int
    x: int
    y: int
    z: int


class ADXLBinaryParser:
    """Incremental parser for protocol-9 ADXL binary frames."""

    def __init__(self, *, fs_hz: float = 800.0) -> None:
        fs_hz = float(fs_hz)
        if fs_hz <= 0:
            raise ValueError("fs_hz must be positive")

        self._interval_us = int(round(1_000_000.0 / fs_hz))
        self._stash = bytearray()
        self._soft_idx = 0

    def reset(self) -> None:
        self._stash.clear()
        self._soft_idx = 0

    def feed(self, chunk: bytes) -> list[Sample]:
        """Feed raw bytes and return all fully decoded samples."""
        if not chunk:
            return []

        self._stash.extend(chunk)
        out: list[Sample] = []

        i = 0
        n = len(self._stash)
        while i + 1 < n:
            if self._stash[i] != SYNC0 or self._stash[i + 1] != SYNC1:
                i += 1
                continue

            frame_end = i + FRAME9_LEN
            if frame_end > n:
                break

            frame = bytes(self._stash[i:frame_end])
            payload = frame[PAYLOAD_SLICE]
            if frame[CRC_INDEX] == crc8_maxim(payload):
                x = _i16_le(frame[2:4])
                y = _i16_le(frame[4:6])
                z = _i16_le(frame[6:8])

                idx = self._soft_idx
                self._soft_idx += 1
                out.append(Sample(idx=idx, t_us=idx * self._interval_us, x=x, y=y, z=z))

                i = frame_end
                continue

            # Bad CRC after sync: shift by one byte to resync quickly.
            i += 1

        drop_until = i
        if i == n - 1 and self._stash[i] != SYNC0:
            drop_until = n
        if drop_until:
            del self._stash[:drop_until]

        return out
