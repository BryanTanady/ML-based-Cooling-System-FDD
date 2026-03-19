#!/usr/bin/env sh
set -eu
# Enable pipefail when supported by the current shell.
(set -o pipefail) >/dev/null 2>&1 && set -o pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"

# Example:
#   sh run_getData.sh --time 30 --label off_1.csv --port /dev/ttyACM0 --baud 115200 --fs 800 --proto 9

python3 "$SCRIPT_DIR/getData2.py" "$@"
