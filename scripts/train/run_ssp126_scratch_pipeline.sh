#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${WALKERNET_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-/home/cpji/wwb/torch/bin/python}"
MASTER_PORT="${MASTER_PORT:-29531}"
NUM_WORKERS="${NUM_WORKERS:-2}"
LOG_DIR="${ROOT_DIR}/outputs/logs"
RUNTIME_DIR="${ROOT_DIR}/runtime/ssp126_scratch_pipeline"
STAGE1_CONFIG="${ROOT_DIR}/configs/server_3090_ssp126_scratch_stage1_ddp8.yaml"
STAGE2_CONFIG="${ROOT_DIR}/configs/server_3090_ssp126_scratch_stage2_ddp8.yaml"
STAGE1_DIR="${ROOT_DIR}/checkpoints_ssp126_scratch_stage1_ddp8"
STAGE2_DIR="${ROOT_DIR}/checkpoints_ssp126_scratch_stage2_ddp8"

mkdir -p "${LOG_DIR}" "${RUNTIME_DIR}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export PYTHONUNBUFFERED=1
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

run_ddp() {
  local config="$1"
  local log_path="$2"
  shift 2
  "${PYTHON_BIN}" -m torch.distributed.run \
    --nproc_per_node=8 \
    --master_port="${MASTER_PORT}" \
    --module src.train \
    --config "${config}" \
    --num-workers "${NUM_WORKERS}" \
    "$@" \
    2>&1 | tee -a "${log_path}"
}

if [[ ! -f "${RUNTIME_DIR}/stage1.complete" ]]; then
  stage1_args=()
  if [[ -s "${STAGE1_DIR}/latest.pt" ]]; then
    stage1_args=(--resume "${STAGE1_DIR}/latest.pt")
  fi
  echo "[ssp126] stage 1 begin: $(date -Is)" | tee -a "${LOG_DIR}/ssp126_scratch_pipeline.log"
  run_ddp "${STAGE1_CONFIG}" "${LOG_DIR}/ssp126_scratch_stage1_ddp8.log" "${stage1_args[@]}"
  touch "${RUNTIME_DIR}/stage1.complete"
fi

if [[ ! -s "${STAGE1_DIR}/best_skill.pt" ]]; then
  echo "stage 1 did not produce best_skill.pt" >&2
  exit 1
fi

if [[ ! -f "${RUNTIME_DIR}/stage2.complete" ]]; then
  stage2_args=(--init-checkpoint "${STAGE1_DIR}/best_skill.pt")
  if [[ -s "${STAGE2_DIR}/latest.pt" ]]; then
    stage2_args=(--resume "${STAGE2_DIR}/latest.pt")
  fi
  echo "[ssp126] stage 2 begin: $(date -Is)" | tee -a "${LOG_DIR}/ssp126_scratch_pipeline.log"
  run_ddp "${STAGE2_CONFIG}" "${LOG_DIR}/ssp126_scratch_stage2_ddp8.log" "${stage2_args[@]}"
  touch "${RUNTIME_DIR}/stage2.complete"
fi

echo "[ssp126] pipeline complete: $(date -Is)" | tee -a "${LOG_DIR}/ssp126_scratch_pipeline.log"
