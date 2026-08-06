#!/usr/bin/env bash
set -Eeuo pipefail

# 四个 SSP 情景的新训练流水线：Stage 1 先学 6 个月，再用 Stage 2 学 12 个月。
# 该脚本只使用新 checkpoint 目录，不会覆盖此前的 sourcewise 实验。

ROOT_DIR="${WALKERNET_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-/home/cpji/wwb/torch/bin/python}"
TRAIN_GPUS="${TRAIN_GPUS:-0,1,2,3,4,5,6,7}"
NPROC_PER_NODE="${NPROC_PER_NODE:-$(awk -F, '{print NF}' <<< "${TRAIN_GPUS}")}"
MASTER_PORT="${MASTER_PORT:-29552}"
NUM_WORKERS="${NUM_WORKERS:-2}"
GPU_MEMORY_LIMIT_MIB="${GPU_MEMORY_LIMIT_MIB:-1000}"
LOG_DIR="${ROOT_DIR}/outputs/logs"
RUNTIME_DIR="${ROOT_DIR}/runtime/sourcewise_s6_s12_sequence"
PIPELINE_LOG="${LOG_DIR}/sourcewise_s6_s12_sequence.log"

mkdir -p "${LOG_DIR}" "${RUNTIME_DIR}"
exec > >(tee -a "${PIPELINE_LOG}") 2>&1

export PYTHONUNBUFFERED=1
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

log() {
  echo "[sourcewise-s6-s12] $*"
}

wait_for_gpus() {
  log "等待 GPU 空闲：${TRAIN_GPUS}"
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

run_ddp() {
  local config="$1"
  local log_file="$2"
  shift 2
  wait_for_gpus
  log "启动训练：config=$(basename "${config}") gpus=${TRAIN_GPUS}"
  CUDA_VISIBLE_DEVICES="${TRAIN_GPUS}" \
    "${PYTHON_BIN}" -m torch.distributed.run \
      --nproc_per_node="${NPROC_PER_NODE}" \
      --master_port="${MASTER_PORT}" \
      --module src.train \
      --config "${config}" \
      --num-workers "${NUM_WORKERS}" \
      "$@" 2>&1 | tee -a "${log_file}"
}

run_scenario() {
  local scenario="$1"
  local stage1_config="${ROOT_DIR}/configs/server_3090_${scenario}_s6_stage1.yaml"
  local stage2_config="${ROOT_DIR}/configs/server_3090_${scenario}_s12_stage2.yaml"
  local stage1_dir="${ROOT_DIR}/checkpoints_${scenario}_s6_stage1"
  local stage2_dir="${ROOT_DIR}/checkpoints_${scenario}_s12_stage2"
  local stage1_log="${LOG_DIR}/${scenario}_s6_stage1.log"
  local stage2_log="${LOG_DIR}/${scenario}_s12_stage2.log"
  local complete_marker="${RUNTIME_DIR}/${scenario}.complete"

  if [[ -f "${complete_marker}" ]]; then
    log "跳过已完成情景：${scenario}"
    return 0
  fi

  mkdir -p "${stage1_dir}" "${stage2_dir}"
  if [[ -s "${stage1_dir}/latest.pt" ]]; then
    run_ddp "${stage1_config}" "${stage1_log}" --resume "${stage1_dir}/latest.pt"
  else
    run_ddp "${stage1_config}" "${stage1_log}"
  fi
  [[ -s "${stage1_dir}/best_skill.pt" ]] || { log "${scenario}: Stage 1 没有 best_skill.pt"; return 1; }

  if [[ -s "${stage2_dir}/latest.pt" ]]; then
    run_ddp "${stage2_config}" "${stage2_log}" --resume "${stage2_dir}/latest.pt"
  else
    run_ddp "${stage2_config}" "${stage2_log}" --init-checkpoint "${stage1_dir}/best_skill.pt"
  fi
  [[ -s "${stage2_dir}/best_skill.pt" ]] || { log "${scenario}: Stage 2 没有 best_skill.pt"; return 1; }

  touch "${complete_marker}"
  log "完成情景：${scenario}"
}

log "流水线开始：$(date -Is)"
for scenario in ssp126 ssp245 ssp370 ssp585; do
  run_scenario "${scenario}"
done
log "全部训练完成：$(date -Is)"
