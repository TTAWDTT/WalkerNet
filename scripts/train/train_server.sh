#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${WALKERNET_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${ROOT_DIR}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

python -m src.train \
  --config configs/server_3090.yaml \
  --device cuda \
  --num-workers 8 \
  "$@"
