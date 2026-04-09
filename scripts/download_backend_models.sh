#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
MODEL_DIR="${PROJECT_DIR}/backend/models"
MODEL_PATH="${MODEL_DIR}/blaze_face_short_range.tflite"

mkdir -p "${MODEL_DIR}"

if [[ -f "${MODEL_PATH}" ]]; then
  echo "[TryMyHair] MediaPipe face detector model already exists: ${MODEL_PATH}"
  exit 0
fi

echo "[TryMyHair] downloading MediaPipe face detector model"
curl -L \
  "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/latest/blaze_face_short_range.tflite" \
  -o "${MODEL_PATH}"

echo "[TryMyHair] saved ${MODEL_PATH}"
