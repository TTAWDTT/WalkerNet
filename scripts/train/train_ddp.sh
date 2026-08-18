#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$(pwd)}"
cd "${ROOT_DIR}"

# 默认只启动一个进程做公开配置检查；多卡时由调用者显式覆盖：
#   NPROC_PER_NODE=2 CUDA_VISIBLE_DEVICES=0,1 bash scripts/train/train_ddp.sh
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

# 这台机器是单机 3090，多数情况下不需要 InfiniBand；关闭 IB 探测能减少 NCCL 初始化卡住的概率。
# 如果后续确认有稳定 NVLink/P2P 拓扑，可把 NCCL_P2P_DISABLE 覆盖回 0。
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

PYTHON_BIN="${PYTHON_BIN:-python}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
CONFIG_PATH="${CONFIG_PATH:-configs/examples/smoke.yaml}"
NUM_WORKERS="${NUM_WORKERS:-4}"

"${PYTHON_BIN}" -m torch.distributed.run \
  --nproc_per_node="${NPROC_PER_NODE}" \
  --master_port="${MASTER_PORT:-29500}" \
  --module src.train \
  --config "${CONFIG_PATH}" \
  --num-workers "${NUM_WORKERS}" \
  "$@"
