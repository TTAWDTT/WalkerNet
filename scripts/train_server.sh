#!/usr/bin/env bash
set -euo pipefail

cd /mnt/sda/WalkerNet

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

python -m src.train \
  --config configs/server_3090.yaml \
  --device cuda \
  --num-workers 8 \
  "$@"
