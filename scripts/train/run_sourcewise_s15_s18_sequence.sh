#!/usr/bin/env bash
set -Eeuo pipefail

# 在 6 -> 12 月流水线完成后，依次训练 SSP245、SSP370、SSP585 的 15 -> 18 月 rollout。
ROOT_DIR="${WALKERNET_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-/home/cpji/wwb/torch/bin/python}"
TRAIN_GPUS="${TRAIN_GPUS:-0,1,2,3,4,5,6,7}"
NPROC_PER_NODE="${NPROC_PER_NODE:-$(awk -F, '{print NF}' <<< "${TRAIN_GPUS}")}"
MASTER_PORT="${MASTER_PORT:-29553}"
NUM_WORKERS="${NUM_WORKERS:-2}"
GPU_MEMORY_LIMIT_MIB="${GPU_MEMORY_LIMIT_MIB:-1000}"
LOG_DIR="${ROOT_DIR}/outputs/logs"
RUNTIME_DIR="${ROOT_DIR}/runtime/sourcewise_s15_s18_sequence"
PIPELINE_LOG="${LOG_DIR}/sourcewise_s15_s18_sequence.log"

mkdir -p "${LOG_DIR}" "${RUNTIME_DIR}"
exec > >(tee -a "${PIPELINE_LOG}") 2>&1

export PYTHONUNBUFFERED=1
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

log() {
  echo "[sourcewise-s15-s18] $*"
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

run_scenario() {
  local scenario="$1"
  local config="${ROOT_DIR}/configs/server_3090_${scenario}_s15_s18_stage3.yaml"
  local stage2_best="${ROOT_DIR}/checkpoints_${scenario}_s12_stage2/best_skill.pt"
  local stage3_dir="${ROOT_DIR}/checkpoints_${scenario}_s15_s18_stage3"
  local train_log="${LOG_DIR}/${scenario}_s15_s18_stage3.log"
  local complete_marker="${RUNTIME_DIR}/${scenario}.complete"

  if [[ -f "${complete_marker}" ]]; then
    log "跳过已完成情景：${scenario}"
    return 0
  fi
  [[ -s "${config}" ]] || { log "缺少配置：${config}"; return 1; }
  [[ -s "${stage2_best}" ]] || { log "缺少 12 月 best checkpoint：${stage2_best}"; return 1; }

  mkdir -p "${stage3_dir}"
  wait_for_gpus

  local train_args
  if [[ -s "${stage3_dir}/latest.pt" ]]; then
    train_args=(--resume "${stage3_dir}/latest.pt")
    log "${scenario}: 从 Stage 3 latest.pt 恢复"
  else
    train_args=(--init-checkpoint "${stage2_best}")
    log "${scenario}: 从 12 月 best_skill.pt 初始化"
  fi

  log "${scenario}: 15 -> 18 月训练开始，gpus=${TRAIN_GPUS}，time=$(date -Is)"
  CUDA_VISIBLE_DEVICES="${TRAIN_GPUS}" \
    "${PYTHON_BIN}" -m torch.distributed.run \
      --nproc_per_node="${NPROC_PER_NODE}" \
      --master_port="${MASTER_PORT}" \
      --module src.train \
      --config "${config}" \
      --num-workers "${NUM_WORKERS}" \
      "${train_args[@]}" 2>&1 | tee -a "${train_log}"

  [[ -s "${stage3_dir}/best_skill.pt" ]] || { log "${scenario}: 缺少 best_skill.pt"; return 1; }
  touch "${complete_marker}"
  log "${scenario}: 15 -> 18 月训练完成，time=$(date -Is)"
}

log "流水线开始：$(date -Is)"
for scenario in ssp245 ssp370 ssp585; do
  run_scenario "${scenario}"
done
log "全部训练完成：$(date -Is)"
