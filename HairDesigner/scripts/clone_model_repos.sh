#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
EXTERNAL_DIR="${PROJECT_DIR}/external_models"

mkdir -p "${EXTERNAL_DIR}"

clone_or_update() {
  local name="$1"
  local repo="$2"
  local dest="${EXTERNAL_DIR}/${name}"

  if [[ -d "${dest}/.git" ]]; then
    echo "[HairDesigner] ${name} exists, fetching latest shallow state"
    git -C "${dest}" fetch --depth 1 origin
    git -C "${dest}" pull --ff-only
    return
  fi

  echo "[HairDesigner] cloning ${name} -> ${dest}"
  git clone --depth 1 "${repo}" "${dest}"
}

clone_or_update "Barbershop" "https://github.com/ZPdesu/Barbershop.git"
clone_or_update "HairFastGAN" "https://github.com/FusionBrainLab/HairFastGAN.git"
clone_or_update "Stable-Hair" "https://github.com/Xiaojiu-z/Stable-Hair.git"

echo "[HairDesigner] model repos are ready under ${EXTERNAL_DIR}"
