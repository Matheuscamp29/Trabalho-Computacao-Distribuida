#!/usr/bin/env bash
# Sobe a API local e abre um túnel público (cloudflared). Ctrl+C derruba os dois.
# Uso: ./servir.sh            (porta 8000)
#      PORTA=9000 ./servir.sh
# A URL pública aparece no log do cloudflared, no formato https://xxxx.trycloudflare.com
# (muda a cada vez que o túnel sobe).
set -e
cd "$(dirname "$0")"
PORTA="${PORTA:-8000}"
PY="venv/bin/python"
[ -x "$PY" ] || { echo "venv não encontrada. Rode: uv venv --python 3.12 venv && uv pip install --python venv/bin/python -r requirements-api.txt"; exit 1; }
command -v cloudflared >/dev/null || { echo "cloudflared não encontrado. Rode: brew install cloudflared"; exit 1; }

"$PY" -m uvicorn api:app --host 127.0.0.1 --port "$PORTA" &
API_PID=$!
cloudflared tunnel --url "http://localhost:$PORTA" --no-autoupdate &
TUN_PID=$!
trap 'kill $API_PID $TUN_PID 2>/dev/null' EXIT INT TERM
wait
