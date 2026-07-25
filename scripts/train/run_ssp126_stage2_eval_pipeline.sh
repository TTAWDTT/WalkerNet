#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="${WALKERNET_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-/home/cpji/wwb/torch/bin/python}"
MASTER_PORT="${MASTER_PORT:-29532}"
NUM_WORKERS="${NUM_WORKERS:-2}"
TRAIN_GPUS="${TRAIN_GPUS:-0,1,2,3,4,5,6,7}"
NPROC_PER_NODE="${NPROC_PER_NODE:-$(awk -F, '{print NF}' <<< "${TRAIN_GPUS}")}"
CONFIG="${ROOT_DIR}/configs/server_3090_ssp126_scratch_stage2_ddp8.yaml"
STAGE1_BEST="${ROOT_DIR}/checkpoints_ssp126_scratch_stage1_ddp8/best_skill.pt"
STAGE2_DIR="${ROOT_DIR}/checkpoints_ssp126_scratch_stage2_ddp8"
LOG_DIR="${ROOT_DIR}/outputs/logs"
EVAL_DIR="${ROOT_DIR}/outputs/eval_ssp126_scratch_stage2_ddp8"
RUNTIME_DIR="${ROOT_DIR}/runtime/ssp126_stage2_eval_pipeline"
PIPELINE_LOG="${LOG_DIR}/ssp126_scratch_stage2_eval_pipeline.log"
TRAIN_LOG="${LOG_DIR}/ssp126_scratch_stage2_ddp8.log"

mkdir -p "${LOG_DIR}" "${EVAL_DIR}" "${RUNTIME_DIR}" "${STAGE2_DIR}"

export PYTHONUNBUFFERED=1
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

log() {
  echo "[ssp126-stage2] $*" | tee -a "${PIPELINE_LOG}"
}

on_error() {
  local exit_code=$?
  log "pipeline failed: exit_code=${exit_code}, time=$(date -Is)"
  exit "${exit_code}"
}
trap on_error ERR

run_stage2() {
  local train_args=()
  if [[ -s "${STAGE2_DIR}/latest.pt" ]]; then
    train_args=(--resume "${STAGE2_DIR}/latest.pt")
    log "resume Stage 2 from ${STAGE2_DIR}/latest.pt"
  else
    if [[ ! -s "${STAGE1_BEST}" ]]; then
      log "missing Stage 1 checkpoint: ${STAGE1_BEST}"
      return 1
    fi
    train_args=(--init-checkpoint "${STAGE1_BEST}")
    log "initialize Stage 2 from Stage 1 best_skill: ${STAGE1_BEST}"
  fi

  log "Stage 2 training begin: gpus=${TRAIN_GPUS}, world_size=${NPROC_PER_NODE}, time=$(date -Is)"
  CUDA_VISIBLE_DEVICES="${TRAIN_GPUS}" \
    "${PYTHON_BIN}" -m torch.distributed.run \
      --nproc_per_node="${NPROC_PER_NODE}" \
      --master_port="${MASTER_PORT}" \
      --module src.train \
      --config "${CONFIG}" \
      --num-workers "${NUM_WORKERS}" \
      "${train_args[@]}" \
      2>&1 | tee -a "${TRAIN_LOG}"

  if [[ ! -s "${STAGE2_DIR}/best_skill.pt" || ! -s "${STAGE2_DIR}/latest.pt" ]]; then
    log "Stage 2 ended without complete best_skill/latest checkpoints"
    return 1
  fi
  touch "${RUNTIME_DIR}/stage2.complete"
  log "Stage 2 training complete: $(date -Is)"
}

run_evaluation() {
  local name="$1"
  local gpu="$2"
  local checkpoint="$3"
  local source_name="${4:-}"
  local marker="${RUNTIME_DIR}/eval_${name}.complete"
  local output_dir="${EVAL_DIR}/${name}"
  local eval_log="${LOG_DIR}/ssp126_scratch_stage2_eval_${name}.log"

  if [[ -f "${marker}" ]]; then
    log "skip completed evaluation: ${name}"
    return 0
  fi

  local source_args=()
  if [[ -n "${source_name}" ]]; then
    source_args=(--source-names "${source_name}")
  fi

  mkdir -p "${output_dir}"
  log "evaluation begin: name=${name}, gpu=${gpu}, source=${source_name:-ALL}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" -u -m src.evaluate_rollout \
    --config "${CONFIG}" \
    --checkpoint "${checkpoint}" \
    --split test \
    --device cuda \
    --batch-size 4 \
    --num-workers 2 \
    --max-lead 18 \
    --leads 1,3,6,9,12,18 \
    --output-dir "${output_dir}" \
    "${source_args[@]}" \
    > "${eval_log}" 2>&1
  touch "${marker}"
  log "evaluation complete: name=${name}, time=$(date -Is)"
}

run_all_evaluations() {
  local best="${STAGE2_DIR}/best_skill.pt"
  local latest="${STAGE2_DIR}/latest.pt"
  local failed=0
  local pids=()
  local names=()

  run_evaluation "best_skill" 0 "${best}" &
  pids+=("$!"); names+=("best_skill")
  run_evaluation "latest" 1 "${latest}" &
  pids+=("$!"); names+=("latest")
  run_evaluation "best_skill_CESM2" 2 "${best}" "CESM2" &
  pids+=("$!"); names+=("best_skill_CESM2")
  run_evaluation "best_skill_EC-Earth3" 3 "${best}" "EC-Earth3" &
  pids+=("$!"); names+=("best_skill_EC-Earth3")
  run_evaluation "best_skill_GFDL-ESM4" 4 "${best}" "GFDL-ESM4" &
  pids+=("$!"); names+=("best_skill_GFDL-ESM4")
  run_evaluation "best_skill_IPSL-CM6A-LR" 5 "${best}" "IPSL-CM6A-LR" &
  pids+=("$!"); names+=("best_skill_IPSL-CM6A-LR")
  run_evaluation "best_skill_MPI-ESM1-2-HR" 7 "${best}" "MPI-ESM1-2-HR" &
  pids+=("$!"); names+=("best_skill_MPI-ESM1-2-HR")

  for idx in "${!pids[@]}"; do
    if ! wait "${pids[$idx]}"; then
      log "evaluation failed: ${names[$idx]}"
      failed=1
    fi
  done
  if (( failed != 0 )); then
    return 1
  fi

  touch "${RUNTIME_DIR}/evaluation.complete"
  log "all evaluations complete: $(date -Is)"
}

log "pipeline begin: $(date -Is)"
if [[ ! -f "${RUNTIME_DIR}/stage2.complete" ]]; then
  run_stage2
else
  log "skip completed Stage 2 training"
fi

if [[ ! -f "${RUNTIME_DIR}/evaluation.complete" ]]; then
  run_all_evaluations
else
  log "skip completed evaluations"
fi
log "pipeline complete: $(date -Is)"
