#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="${WALKERNET_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-/home/cpji/wwb/torch/bin/python}"
MASTER_PORT="${MASTER_PORT:-29541}"
NUM_WORKERS="${NUM_WORKERS:-2}"
TRAIN_GPUS="${TRAIN_GPUS:-0,1,2,3,4,5,6,7}"
NPROC_PER_NODE="${NPROC_PER_NODE:-$(awk -F, '{print NF}' <<< "${TRAIN_GPUS}")}"
CONFIG="${ROOT_DIR}/configs/server_3090_ssp126_sourcewise_stage1.yaml"
CHECKPOINT_DIR="${ROOT_DIR}/checkpoints_ssp126_sourcewise_stage1"
LOG_DIR="${ROOT_DIR}/outputs/logs"
TRAIN_LOG="${LOG_DIR}/ssp126_sourcewise_stage1.log"

mkdir -p "${CHECKPOINT_DIR}" "${LOG_DIR}"

export PYTHONUNBUFFERED=1
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

train_args=()
if [[ -s "${CHECKPOINT_DIR}/latest.pt" ]]; then
  train_args=(--resume "${CHECKPOINT_DIR}/latest.pt")
  echo "[ssp126-sourcewise] resume from latest.pt" | tee -a "${TRAIN_LOG}"
else
  echo "[ssp126-sourcewise] train from scratch" | tee -a "${TRAIN_LOG}"
fi

echo "[ssp126-sourcewise] gpus=${TRAIN_GPUS} world_size=${NPROC_PER_NODE} begin=$(date -Is)" \
  | tee -a "${TRAIN_LOG}"

CUDA_VISIBLE_DEVICES="${TRAIN_GPUS}" \
  "${PYTHON_BIN}" -m torch.distributed.run \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --master_port="${MASTER_PORT}" \
    --module src.train \
    --config "${CONFIG}" \
    --num-workers "${NUM_WORKERS}" \
    "${train_args[@]}" \
    2>&1 | tee -a "${TRAIN_LOG}"
