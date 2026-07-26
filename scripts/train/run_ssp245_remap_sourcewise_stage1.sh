#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="${WALKERNET_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-/home/cpji/wwb/torch/bin/python}"
CDO_BIN="${CDO_BIN:-/home/cpji/cdo/bin/cdo}"
TRAIN_GPUS="${TRAIN_GPUS:-0,1,2,3,4,5,6,7}"
NPROC_PER_NODE="${NPROC_PER_NODE:-$(awk -F, '{print NF}' <<< "${TRAIN_GPUS}")}"
MASTER_PORT="${MASTER_PORT:-29542}"
NUM_WORKERS="${NUM_WORKERS:-2}"
CONFIG="${ROOT_DIR}/configs/server_3090_ssp245_sourcewise_stage1.yaml"
RAW_ROOT="${ROOT_DIR}/cmip6-ssp"
REMAP_ROOT="${ROOT_DIR}/cmip6_ssp_1x1"
CHECKPOINT_DIR="${ROOT_DIR}/checkpoints_ssp245_sourcewise_stage1"
RUNTIME_DIR="${ROOT_DIR}/runtime/ssp245_sourcewise_stage1"
LOG_DIR="${ROOT_DIR}/outputs/logs"
PIPELINE_LOG="${LOG_DIR}/ssp245_sourcewise_pipeline.log"
TRAIN_LOG="${LOG_DIR}/ssp245_sourcewise_stage1.log"

mkdir -p "${CHECKPOINT_DIR}" "${RUNTIME_DIR}" "${LOG_DIR}"
exec > >(tee -a "${PIPELINE_LOG}") 2>&1

export PYTHONUNBUFFERED=1
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

log() {
  echo "[ssp245-sourcewise] $*"
}

if [[ ! -f "${RUNTIME_DIR}/remap.complete" ]]; then
  log "remap begin: $(date -Is)"
  "${PYTHON_BIN}" -u scripts/data/remap_cmip6_ssp_to_1x1.py \
    --input-root "${RAW_ROOT}" \
    --output-root "${REMAP_ROOT}" \
    --cdo-bin "${CDO_BIN}" \
    --threads 8 \
    --scenarios ssp245
  touch "${RUNTIME_DIR}/remap.complete"
  log "remap complete: $(date -Is)"
fi

log "validate 20 remapped files"
"${PYTHON_BIN}" -u scripts/data/check_remapped_data.py \
  --data-dir "${REMAP_ROOT}/ssp245" \
  --multi-source \
  --expected-months 1032

log "compute and verify source-wise statistics"
"${PYTHON_BIN}" -u scripts/data/inspect_norm_stats.py --config "${CONFIG}"

log "wait for selected GPUs: ${TRAIN_GPUS}"
while true; do
  busy=0
  IFS=',' read -ra gpu_ids <<< "${TRAIN_GPUS}"
  for gpu_id in "${gpu_ids[@]}"; do
    used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "${gpu_id}" | tr -d ' ')"
    if (( used > 1000 )); then
      busy=1
      break
    fi
  done
  if (( busy == 0 )); then
    break
  fi
  sleep 300
done

train_args=()
if [[ -s "${CHECKPOINT_DIR}/latest.pt" ]]; then
  train_args=(--resume "${CHECKPOINT_DIR}/latest.pt")
  log "resume from latest.pt"
else
  log "train from scratch"
fi

log "training begin: gpus=${TRAIN_GPUS} world_size=${NPROC_PER_NODE} time=$(date -Is)"
CUDA_VISIBLE_DEVICES="${TRAIN_GPUS}" \
  "${PYTHON_BIN}" -m torch.distributed.run \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --master_port="${MASTER_PORT}" \
    --module src.train \
    --config "${CONFIG}" \
    --num-workers "${NUM_WORKERS}" \
    "${train_args[@]}" \
    2>&1 | tee -a "${TRAIN_LOG}"

log "training complete: $(date -Is)"
