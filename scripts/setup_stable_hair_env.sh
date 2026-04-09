#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
STABLE_HAIR_DIR="${PROJECT_DIR}/external_models/Stable-Hair"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${STABLE_HAIR_VENV:-${STABLE_HAIR_DIR}/.venv}"

if [[ ! -d "${STABLE_HAIR_DIR}" ]]; then
  echo "[TryMyHair] Stable-Hair repo is missing. Run: bash scripts/clone_model_repos.sh" >&2
  exit 1
fi

echo "[TryMyHair] using python: $(${PYTHON_BIN} --version)"
echo "[TryMyHair] creating Stable-Hair venv: ${VENV_DIR}"
if ! "${PYTHON_BIN}" -m venv "${VENV_DIR}"; then
  cat >&2 <<'EOF'

[TryMyHair] Python venv creation failed.

On Ubuntu / WSL, install the matching venv package first:

  sudo apt update
  sudo apt install python3.12-venv

Then rerun:

  bash scripts/setup_stable_hair_env.sh

EOF
  exit 1
fi

PIP="${VENV_DIR}/bin/pip"
PYTHON="${VENV_DIR}/bin/python"

"${PYTHON}" -m pip install --upgrade pip setuptools wheel

echo "[TryMyHair] installing PyTorch CUDA 11.8 stack"
# Upstream requirements currently pin torch 2.2.2 together with torchvision 0.16.2.
# Use the matching torchvision 0.17.2 for the inference environment.
"${PIP}" install \
  --extra-index-url https://download.pytorch.org/whl/cu118 \
  "torch==2.2.2+cu118" \
  "torchvision==0.17.2+cu118"

echo "[TryMyHair] installing xformers"
# There is no public 0.0.25.post1+cu118 package on the indexes commonly used by pip.
# Install the upstream-pinned xformers release without the local CUDA suffix and keep the torch stack unchanged.
"${PIP}" install --no-deps "xformers==0.0.25.post1"

echo "[TryMyHair] installing Stable-Hair requirements without re-installing torch stack"
grep -v -E '^(torch|torchvision|xformers)==' "${STABLE_HAIR_DIR}/requirements.txt" \
  | "${PIP}" install -r /dev/stdin

echo "[TryMyHair] verifying imports"
"${PYTHON}" - <<'PY'
import torch
import diffusers
import omegaconf

print("torch =", torch.__version__)
print("cuda_available =", torch.cuda.is_available())
print("diffusers =", diffusers.__version__)
print("omegaconf = OK")
PY

echo "[TryMyHair] Stable-Hair env is ready: ${PYTHON}"
echo "[TryMyHair] If FastAPI is already running, restart it so /stable-hair/status uses this venv."
