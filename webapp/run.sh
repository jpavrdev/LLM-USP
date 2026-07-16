#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"

if [ ! -d ".venv" ]; then
    echo "[run.sh] criando venv em webapp/.venv ..."
    python3 -m venv .venv
    ./.venv/bin/pip install --upgrade pip
    ./.venv/bin/pip install -r requirements.txt
fi

echo "[run.sh] subindo servidor em http://127.0.0.1:8000"
exec ./.venv/bin/uvicorn backend:app --host 127.0.0.1 --port 8000 --reload
