#!/usr/bin/env bash
set -euo pipefail

cd /mnt/sda/WalkerNet

# 默认使用服务器 8 张 3090；短测时可覆盖：
#   NPROC_PER_NODE=2 CUDA_VISIBLE_DEVICES=0,1 bash scripts/train_ddp.sh
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

PYTHON_BIN="${PYTHON_BIN:-/home/cpji/wwb/torch/bin/python}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
CONFIG_PATH="${CONFIG_PATH:-configs/server_ddp_smoke.yaml}"
NUM_WORKERS="${NUM_WORKERS:-4}"

"${PYTHON_BIN}" -m torch.distributed.run \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --master_port="${MASTER_PORT:-29500}" \
  --module src.train \
  --config "${CONFIG_PATH}" \
  --num-workers "${NUM_WORKERS}" \
  "$@"
