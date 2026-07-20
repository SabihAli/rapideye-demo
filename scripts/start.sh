#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ ! -f .env ]]; then
  cp .env.example .env
  echo "Created .env from .env.example"
fi

set -a
source .env
set +a

API_PORT="${API_PORT:-8001}"

if [[ ! -d .venv ]]; then
  echo "Missing .venv — run: make install"
  exit 1
fi

if [[ ! -d web/node_modules ]]; then
  echo "Missing web/node_modules — run: cd web && npm install"
  exit 1
fi

echo "Starting backend on :${API_PORT} and frontend on :5173"
echo "Open http://localhost:5173"
echo "Press Ctrl+C to stop both."

source .venv/bin/activate
trap 'kill 0' EXIT

uvicorn server.main:app --host 0.0.0.0 --port "${API_PORT}" --reload &
(cd web && npm run dev)
