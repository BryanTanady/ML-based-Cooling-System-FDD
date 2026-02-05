"""Binary frame decoder for ADXL345 streams.

Supports the two wire formats used in this repo:

1) Compact 9-byte frames (used for 800 Hz @ 115200):
   [0] 0xAA
   [1] 0x55
   [2..3] x (int16 LE)
   [4..5] y (int16 LE)
   [6..7] z (int16 LE)
   [8] crc8_maxim over bytes [2..7]

2) Legacy 18-byte frames (older binary protocol):
   [0] 0xAA
   [1] 0x55
   [2] type (=0x01)
   [3..6]  idx  (uint32 LE)
   [7..10] t_us (uint32 LE)
   [11..12] x   (int16 LE)
   [13..14] y   (int16 LE)
   [15..16] z   (int16 LE)
   [17] crc8_maxim over bytes [2..16]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

SYNC0 = 0xAA
SYNC1 = 0x55

FRAME9_LEN = 9
FRAME18_LEN = 18
TYPE_SAMPLE = 0x01

Protocol = Literal["9", "18", "auto"]


def _u32_le(b: bytes) -> int:
    return int.from_bytes(b, byteorder="little", signed=False)


def _i16_le(b: bytes) -> int:
    return int.from_bytes(b, byteorder="little", signed=True)


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


@dataclass(frozen=True)
class Sample:
    idx: int
    t_us: int
    x: int
    y: int
    z: int


class ADXLBinaryParser:
    """Incremental binary stream parser (feed bytes -> get decoded samples)."""

    def __init__(self, *, protocol: Protocol = "9", fs_hz: float = 800.0) -> None:
        if protocol not in {"9", "18", "auto"}:
            raise ValueError("protocol must be one of: '9', '18', 'auto'")
        self._protocol: Protocol = protocol
        self._fs_hz = float(fs_hz)
        self._interval_us = int(round(1_000_000.0 / self._fs_hz)) if self._fs_hz > 0 else 0

        self._stash = bytearray()
        self._soft_idx = 0

        # Legacy 18-byte frame helpers
        self._t0_us: int | None = None
        self._last_idx18: int | None = None

    def reset(self) -> None:
        self._stash.clear()
        self._soft_idx = 0
        self._t0_us = None
        self._last_idx18 = None

    def feed(self, chunk: bytes) -> list[Sample]:
        """Feed a raw bytes chunk and return any fully-decoded samples."""
        if not chunk:
            return []

        self._stash.extend(chunk)
        out: list[Sample] = []

        i = 0
        n = len(self._stash)
        while i + 2 <= n:
            if self._stash[i] != SYNC0 or self._stash[i + 1] != SYNC1:
                i += 1
                continue

            do18 = self._protocol in {"18", "auto"}
            do9 = self._protocol in {"9", "auto"}

            # Prefer 18-byte in auto mode to avoid confusing it with 9-byte prefixes.
            if do18 and i + FRAME18_LEN <= n:
                frame18 = bytes(self._stash[i : i + FRAME18_LEN])
                if frame18[2] == TYPE_SAMPLE:
                    payload18 = frame18[2:17]
                    if frame18[17] == crc8_maxim(payload18):
                        idx = _u32_le(frame18[3:7])
                        t_us_raw = _u32_le(frame18[7:11])

                        legacy_ok = True
                        if self._protocol == "auto":
                            if self._last_idx18 is None:
                                legacy_ok = idx <= 50_000_000
                            else:
                                if idx == 0:
                                    legacy_ok = True
                                else:
                                    legacy_ok = (idx > self._last_idx18) and ((idx - self._last_idx18) <= 100_000)

                        if legacy_ok:
                            if idx == 0:
                                self._t0_us = None
                            if self._t0_us is None:
                                self._t0_us = t_us_raw

                            t_us = (t_us_raw - self._t0_us) & 0xFFFFFFFF
                            x = _i16_le(frame18[11:13])
                            y = _i16_le(frame18[13:15])
                            z = _i16_le(frame18[15:17])

                            out.append(Sample(idx=int(idx), t_us=int(t_us), x=x, y=y, z=z))
                            self._last_idx18 = idx
                            i += FRAME18_LEN
                            continue

            if do9 and i + FRAME9_LEN <= n:
                frame9 = bytes(self._stash[i : i + FRAME9_LEN])
                payload9 = frame9[2:8]
                if frame9[8] == crc8_maxim(payload9):
                    x = _i16_le(frame9[2:4])
                    y = _i16_le(frame9[4:6])
                    z = _i16_le(frame9[6:8])
                    idx = self._soft_idx
                    self._soft_idx += 1
                    t_us = idx * self._interval_us
                    out.append(Sample(idx=int(idx), t_us=int(t_us), x=x, y=y, z=z))
                    i += FRAME9_LEN
                    continue

            # Sync found but not enough bytes yet or CRC mismatch: rescan.
            i += 1

        if i:
            del self._stash[:i]
        return out

