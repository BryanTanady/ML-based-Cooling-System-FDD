"""Text parsing helpers for broker serial input."""

from __future__ import annotations

import re
from typing import Optional, Tuple

_XYZ_LOG_PATTERN = re.compile(
    r"x\s+([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s+"
    r"y\s+([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s+"
    r"z\s+([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
)

_S_LINE_PATTERN = re.compile(
    r"S\s+[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?\s+"
    r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s+"
    r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s+"
    r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
)


def parse_sample(line: str) -> Optional[Tuple[float, float, float]]:
    """Parse a serial text line into accelerometer xyz values."""
    if "Skipping unparsable line:" in line:
        line = line.split("Skipping unparsable line:", 1)[1].strip()

    parts = line.strip().split(",")
    if len(parts) < 3:
        match = _XYZ_LOG_PATTERN.search(line)
        if match:
            try:
                return float(match.group(1)), float(match.group(2)), float(match.group(3))
            except ValueError:
                return None

        match_s = _S_LINE_PATTERN.search(line)
        if match_s:
            try:
                return float(match_s.group(1)), float(match_s.group(2)), float(match_s.group(3))
            except ValueError:
                return None

        return None

    try:
        return float(parts[0]), float(parts[1]), float(parts[2])
    except ValueError:
        return None
