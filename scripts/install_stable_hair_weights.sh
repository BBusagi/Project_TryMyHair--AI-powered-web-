#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
STABLE_HAIR_DIR="${PROJECT_DIR}/external_models/Stable-Hair"
SOURCE_DIR="${1:-}"

if [[ -z "${SOURCE_DIR}" ]]; then
  cat >&2 <<'EOF'
Usage:
  bash scripts/install_stable_hair_weights.sh /path/to/downloaded/Stable-Hair-weights

The source directory must contain either:
  stage1/pytorch_model.bin
  stage2/pytorch_model.bin
  stage2/pytorch_model_1.bin
  stage2/pytorch_model_2.bin

or those four files in its top-level directory.
EOF
  exit 1
fi

if [[ ! -d "${SOURCE_DIR}" ]]; then
  echo "[TryMyHair] source directory does not exist: ${SOURCE_DIR}" >&2
  exit 1
fi

mkdir -p "${STABLE_HAIR_DIR}/models/stage1" "${STABLE_HAIR_DIR}/models/stage2"

copy_weight() {
  local src="$1"
  local dst="$2"

  if [[ ! -f "${src}" ]]; then
    echo "[TryMyHair] missing source weight: ${src}" >&2
    exit 1
  fi

  echo "[TryMyHair] copying ${src} -> ${dst}"
  cp -f "${src}" "${dst}"
}

stage1="${SOURCE_DIR}/stage1/pytorch_model.bin"
stage2_a="${SOURCE_DIR}/stage2/pytorch_model.bin"
stage2_b="${SOURCE_DIR}/stage2/pytorch_model_1.bin"
stage2_c="${SOURCE_DIR}/stage2/pytorch_model_2.bin"

if [[ ! -f "${stage1}" ]]; then
  stage1="${SOURCE_DIR}/pytorch_model.bin"
  stage2_a="${SOURCE_DIR}/pytorch_model.bin"
  stage2_b="${SOURCE_DIR}/pytorch_model_1.bin"
  stage2_c="${SOURCE_DIR}/pytorch_model_2.bin"
fi

copy_weight "${stage1}" "${STABLE_HAIR_DIR}/models/stage1/pytorch_model.bin"
copy_weight "${stage2_a}" "${STABLE_HAIR_DIR}/models/stage2/pytorch_model.bin"
copy_weight "${stage2_b}" "${STABLE_HAIR_DIR}/models/stage2/pytorch_model_1.bin"
copy_weight "${stage2_c}" "${STABLE_HAIR_DIR}/models/stage2/pytorch_model_2.bin"

echo "[TryMyHair] weights installed under ${STABLE_HAIR_DIR}/models"
