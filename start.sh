#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FRONTEND_DIR="$ROOT_DIR/frontend"
BACKEND_DIR="$ROOT_DIR/backend"

API_HOST="${API_HOST:-127.0.0.1}"
API_PORT="${API_PORT:-8000}"
WEB_HOST="${WEB_HOST:-127.0.0.1}"
WEB_PORT="${WEB_PORT:-5173}"

BACKEND_PID=""
FRONTEND_PID=""

cleanup() {
  local exit_code=$?

  if [[ -n "$FRONTEND_PID" ]] && kill -0 "$FRONTEND_PID" 2>/dev/null; then
    kill "$FRONTEND_PID" 2>/dev/null || true
    wait "$FRONTEND_PID" 2>/dev/null || true
  fi

  if [[ -n "$BACKEND_PID" ]] && kill -0 "$BACKEND_PID" 2>/dev/null; then
    kill "$BACKEND_PID" 2>/dev/null || true
    wait "$BACKEND_PID" 2>/dev/null || true
  fi

  exit "$exit_code"
}

trap cleanup EXIT INT TERM

require_cmd() {
  local cmd="$1"
  local hint="$2"
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "Missing required command: $cmd"
    echo "$hint"
    exit 1
  fi
}

require_cmd uv "Install uv first, then re-run ./start.sh."
require_cmd bun "Install bun first, then re-run ./start.sh."
require_cmd python3 "Install Python 3 first, then re-run ./start.sh."

echo "Preparing backend environment..."
if [[ ! -d "$BACKEND_DIR/.venv" ]]; then
  (
    cd "$BACKEND_DIR"
    uv venv
  )
fi

(
  cd "$BACKEND_DIR"
  uv pip install --python .venv/bin/python -e .
)

echo "Preparing frontend dependencies..."
(
  cd "$FRONTEND_DIR"
  if [[ ! -d node_modules ]]; then
    bun install
  fi
)

echo "Starting backend on http://$API_HOST:$API_PORT ..."
(
  cd "$BACKEND_DIR"
  ./.venv/bin/python -m uvicorn app.main:app --host "$API_HOST" --port "$API_PORT" --reload
) &
BACKEND_PID=$!

echo "Starting frontend on http://$WEB_HOST:$WEB_PORT ..."
(
  cd "$FRONTEND_DIR"
  VITE_API_URL="http://$API_HOST:$API_PORT" bun run dev --host "$WEB_HOST" --port "$WEB_PORT"
) &
FRONTEND_PID=$!

echo
echo "Lexis is starting up:"
echo "  Frontend: http://$WEB_HOST:$WEB_PORT"
echo "  Backend:  http://$API_HOST:$API_PORT"
echo
echo "Press Ctrl+C to stop both servers."

wait "$BACKEND_PID" "$FRONTEND_PID"
