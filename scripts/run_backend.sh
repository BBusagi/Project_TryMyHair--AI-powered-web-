#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

cd "${PROJECT_DIR}"

UVICORN_BIN="${PROJECT_DIR}/.venv/bin/uvicorn"

if [[ -x "${UVICORN_BIN}" ]]; then
  exec "${UVICORN_BIN}" backend.server:app --host 127.0.0.1 --port 8000 --reload
fi

exec uvicorn backend.server:app --host 127.0.0.1 --port 8000 --reload
