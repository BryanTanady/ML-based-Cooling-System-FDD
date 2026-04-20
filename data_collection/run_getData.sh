#!/usr/bin/env sh
set -eu
# Enable pipefail when supported by the current shell.
(set -o pipefail) >/dev/null 2>&1 && set -o pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"

python3 "$SCRIPT_DIR/getData.py" "$@"
