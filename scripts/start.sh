#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ ! -f .env ]]; then
  cp .env.example .env
  echo "Created .env from .env.example"
fi

if [[ ! -d .venv ]]; then
  echo "Missing .venv — run: python -m venv .venv && source .venv/bin/activate && pip install -e ."
  exit 1
fi

if [[ ! -d web/node_modules ]]; then
  echo "Missing web/node_modules — run: cd web && npm install"
  exit 1
fi

echo "Starting backend on :8000 and frontend on :5173"
echo "Open http://localhost:5173"
echo "Press Ctrl+C to stop both."

source .venv/bin/activate
trap 'kill 0' EXIT

uvicorn server.main:app --host 0.0.0.0 --port 8000 --reload &
(cd web && npm run dev)
