#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
STABLE_HAIR_DIR="${PROJECT_DIR}/external_models/Stable-Hair"
PYTHON_BIN="${STABLE_HAIR_PYTHON:-${STABLE_HAIR_DIR}/.venv/bin/python}"

echo "[TryMyHair] project: ${PROJECT_DIR}"
echo "[TryMyHair] repo: ${STABLE_HAIR_DIR}"

if [[ ! -d "${STABLE_HAIR_DIR}" ]]; then
  echo "FAIL repo missing"
  exit 1
fi

required_weights=(
  "${STABLE_HAIR_DIR}/models/stage1/pytorch_model.bin"
  "${STABLE_HAIR_DIR}/models/stage2/pytorch_model.bin"
  "${STABLE_HAIR_DIR}/models/stage2/pytorch_model_1.bin"
  "${STABLE_HAIR_DIR}/models/stage2/pytorch_model_2.bin"
)

missing=0
for path in "${required_weights[@]}"; do
  if [[ -f "${path}" ]]; then
    bytes="$(stat -c '%s' "${path}")"
    echo "PASS weight ${path} (${bytes} bytes)"
  else
    echo "FAIL weight missing ${path}"
    missing=1
  fi
done

if [[ -x "${PYTHON_BIN}" ]]; then
  echo "[TryMyHair] checking python: ${PYTHON_BIN}"
  "${PYTHON_BIN}" - <<'PY'
import torch
import diffusers
import omegaconf

print("PASS torch", torch.__version__)
print("PASS cuda_available", torch.cuda.is_available())
print("PASS diffusers", diffusers.__version__)
print("PASS omegaconf")
PY
else
  echo "FAIL python missing ${PYTHON_BIN}"
  missing=1
fi

if command -v nvidia-smi >/dev/null; then
  echo "[TryMyHair] nvidia-smi:"
  nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
else
  echo "WARN nvidia-smi not found"
fi

if [[ "${missing}" -ne 0 ]]; then
  exit 1
fi

echo "[TryMyHair] Stable-Hair local prerequisites look ready."
