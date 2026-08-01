#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="${WALKERNET_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-/home/cpji/wwb/torch/bin/python}"
TRAIN_GPUS="${TRAIN_GPUS:-0,1,2,3,4,5,6,7}"
NPROC_PER_NODE="${NPROC_PER_NODE:-$(awk -F, '{print NF}' <<< "${TRAIN_GPUS}")}"
MASTER_PORT="${MASTER_PORT:-29545}"
NUM_WORKERS="${NUM_WORKERS:-2}"
GPU_MEMORY_LIMIT_MIB="${GPU_MEMORY_LIMIT_MIB:-1000}"
LOG_DIR="${ROOT_DIR}/outputs/logs"
RUNTIME_DIR="${ROOT_DIR}/runtime/ssp_sourcewise_stage2_sequence"
PIPELINE_LOG="${LOG_DIR}/ssp_sourcewise_stage2_sequence.log"

mkdir -p "${LOG_DIR}" "${RUNTIME_DIR}"
exec > >(tee -a "${PIPELINE_LOG}") 2>&1

export PYTHONUNBUFFERED=1
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

log() {
  echo "[ssp-stage2] $*"
}

wait_for_gpus() {
  log "wait for GPUs: ${TRAIN_GPUS}"
  while true; do
    local busy=0
    IFS=',' read -ra gpu_ids <<< "${TRAIN_GPUS}"
    for gpu_id in "${gpu_ids[@]}"; do
      local used
      used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "${gpu_id}" | tr -d ' ')"
      if (( used > GPU_MEMORY_LIMIT_MIB )); then
        busy=1
        break
      fi
    done
    if (( busy == 0 )); then
      return 0
    fi
    sleep 300
  done
}

run_scenario() {
  local scenario="$1"
  local config="${ROOT_DIR}/configs/server_3090_${scenario}_sourcewise_stage2.yaml"
  local stage1_best="${ROOT_DIR}/checkpoints_${scenario}_sourcewise_stage1/best_skill.pt"
  local stage2_dir="${ROOT_DIR}/checkpoints_${scenario}_sourcewise_stage2"
  local train_log="${LOG_DIR}/${scenario}_sourcewise_stage2.log"
  local complete_marker="${RUNTIME_DIR}/${scenario}.complete"

  if [[ -f "${complete_marker}" ]]; then
    log "skip completed scenario: ${scenario}"
    return 0
  fi
  if [[ ! -s "${config}" ]]; then
    log "missing config: ${config}"
    return 1
  fi
  if [[ ! -s "${stage1_best}" ]]; then
    log "missing Stage 1 best checkpoint: ${stage1_best}"
    return 1
  fi

  mkdir -p "${stage2_dir}"
  wait_for_gpus

  local train_args
  if [[ -s "${stage2_dir}/latest.pt" ]]; then
    train_args=(--resume "${stage2_dir}/latest.pt")
    log "${scenario}: resume Stage 2 latest.pt"
  else
    train_args=(--init-checkpoint "${stage1_best}")
    log "${scenario}: initialize from Stage 1 best_skill.pt"
  fi

  log "${scenario}: Stage 2 begin, gpus=${TRAIN_GPUS}, time=$(date -Is)"
  CUDA_VISIBLE_DEVICES="${TRAIN_GPUS}" \
    "${PYTHON_BIN}" -m torch.distributed.run \
      --nproc_per_node="${NPROC_PER_NODE}" \
      --master_port="${MASTER_PORT}" \
      --module src.train \
      --config "${config}" \
      --num-workers "${NUM_WORKERS}" \
      "${train_args[@]}" \
      2>&1 | tee -a "${train_log}"

  if [[ ! -s "${stage2_dir}/best_skill.pt" || ! -s "${stage2_dir}/latest.pt" ]]; then
    log "${scenario}: training ended without complete checkpoints"
    return 1
  fi
  touch "${complete_marker}"
  log "${scenario}: Stage 2 complete, time=$(date -Is)"
}

log "pipeline begin: $(date -Is)"
for scenario in ssp245 ssp370 ssp585; do
  run_scenario "${scenario}"
done
log "pipeline complete: $(date -Is)"
