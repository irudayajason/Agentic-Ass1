#!/usr/bin/env bash
# Start SpendSense at http://localhost:8000 (Ctrl+C to stop)
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -x .venv/bin/uvicorn ]; then
  echo "Virtual env missing. Run: uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r requirements.txt"
  exit 1
fi
if [ ! -f .env ]; then
  echo "No .env file. Run: cp .env.example .env  and add your GROQ_API_KEY"
  exit 1
fi

PORT="${PORT:-8000}"
echo "SpendSense → http://localhost:${PORT}"
exec .venv/bin/uvicorn server:app --port "${PORT}"
