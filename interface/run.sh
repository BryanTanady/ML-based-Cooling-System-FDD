#!/usr/bin/env bash

set -e

# Always work relative to this script's location
ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"

# --- FRONTEND ---
echo "[run.sh] Starting frontend (npm run dev)..."
cd "$ROOT_DIR/frontend"
npm run dev &
FRONTEND_PID=$!

# --- BACKEND ---
echo "[run.sh] Starting backend (uvicorn)..."
cd "$ROOT_DIR/backend"
uvicorn server:app --reload --port 8000 --host 127.0.0.1 &
BACKEND_PID=$!

echo "[run.sh] Frontend PID: $FRONTEND_PID"
echo "[run.sh] Backend PID (launcher): $BACKEND_PID"
echo "[run.sh] Press Ctrl+C to stop both."

cleanup() {
  echo "[run.sh] Stopping servers..."

  # Kill frontend dev server
  kill "$FRONTEND_PID" 2>/dev/null || true

  # Kill backend server
  lsof -t -i :8000 | xargs kill -9

  exit 0
}

trap cleanup INT TERM

# Wait for children so Ctrl+C works nicely
wait