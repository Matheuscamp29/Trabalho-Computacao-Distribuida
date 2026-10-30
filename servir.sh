#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
PORTA="${PORTA:-8000}"
PY="venv/bin/python"
[ -x "$PY" ] || { echo "venv não encontrada. Rode: uv venv --python 3.12 venv && uv pip install --python venv/bin/python -r requirements-api.txt"; exit 1; }
command -v cloudflared >/dev/null || { echo "cloudflared não encontrado. Rode: brew install cloudflared"; exit 1; }

HOST=127.0.0.1 PORT="$PORTA" "$PY" api.py &
API_PID=$!
cloudflared tunnel --url "http://localhost:$PORTA" --no-autoupdate &
TUN_PID=$!
trap 'kill $API_PID $TUN_PID 2>/dev/null' EXIT INT TERM
wait
