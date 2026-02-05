#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# Example:
#   sh run_getData.sh --time 180 --label NORMAL_1.csv --port /dev/ttyUSB0 --baud 115200 --fs 800 --proto 9

python3 "$SCRIPT_DIR/getData2.py" "$@"
