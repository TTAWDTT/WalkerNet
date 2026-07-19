#!/usr/bin/env bash
set -euo pipefail

cd /mnt/sda/WalkerNet

LOG_DIR="/mnt/sda/WalkerNet/outputs/logs"
OUT_DIR="/mnt/sda/WalkerNet/outputs/cnop_event_l2_constraint_scale04_64_0707"
SAMPLE_DIR="${OUT_DIR}/sampling"
SHARD_DIR="${OUT_DIR}/shards"
CONSTRAINT_JSON="/mnt/sda/WalkerNet/outputs/cnop_constraint_0705/cnop_constraint_summary.json"
FORECAST_CLIM="/mnt/sda/WalkerNet/outputs/cnop_tos_zos_patch_0703/forecast_tos_climatology_train_h12.npz"
CONFIG="configs/server_3090_mixed5_ddp8.yaml"
CHECKPOINT="/mnt/sda/WalkerNet/checkpoints_mixed5_enso18_simple_loss_corr_ddp8/best_skill.pt"
PYTHON="/home/cpji/wwb/torch/bin/python"

mkdir -p "${LOG_DIR}" "${SAMPLE_DIR}" "${SHARD_DIR}" "${OUT_DIR}/figures"

wait_for_gpu_memory() {
  local max_used_mb="$1"
  echo "[cnop64] wait for GPUs memory.used <= ${max_used_mb} MiB: $(date)"
  while true; do
    local used_values
    used_values="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)"
    local ok=1
    local max_seen=0
    while read -r used; do
      used="${used// /}"
      if [[ -n "${used}" && "${used}" -gt "${max_seen}" ]]; then
        max_seen="${used}"
      fi
      if [[ -n "${used}" && "${used}" -gt "${max_used_mb}" ]]; then
        ok=0
      fi
    done <<< "${used_values}"
    echo "[cnop64] gpu max memory.used=${max_seen} MiB"
    if [[ "${ok}" -eq 1 ]]; then
      break
    fi
    sleep 600
  done
}

run_case() {
  local idx="$1"
  local source="$2"
  local year="$3"
  local gpu="$4"
  local case_dir="${SHARD_DIR}/case_${idx}_${source}_${year}"
  if [[ -s "${case_dir}/cnop_summary.csv" ]]; then
    echo "[cnop64] skip finished case=${idx} ${source} ${year}"
    return 0
  fi
  rm -rf "${case_dir}"
  mkdir -p "${case_dir}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" -u scripts/cnop/compute_tos_zos_cnop.py \
    --config "${CONFIG}" \
    --checkpoint "${CHECKPOINT}" \
    --split test \
    --case-source-name "${source}" \
    --case-target-year "${year}" \
    --device cuda \
    --num-cases 1 \
    --horizon 12 \
    --steps 80 \
    --lr 0.08 \
    --num-starts 16 \
    --top-k 5 \
    --random-init-scale 0.02 \
    --constraint-mode event_l2 \
    --constraint-file "${CONSTRAINT_JSON}" \
    --constraint-scale 0.4 \
    --max-abs 2.0 \
    --neutral-threshold 0.5 \
    --domain tropical_pacific \
    --perturb-grid patch \
    --perturb-patch-size 4 \
    --objective-mode lead_delta \
    --objective-lead 12 \
    --smoothness-weight 0.001 \
    --output-dir "${case_dir}" \
    > "${LOG_DIR}/cnop64_scale04_case_${idx}_${source}_${year}.log" 2>&1
}

merge_outputs() {
  echo "[cnop64] merge outputs: $(date)"
  cp "${SHARD_DIR}"/case_*_*/case_*.npz "${OUT_DIR}/"
  cp "${SHARD_DIR}"/case_*_*/*_history.json "${OUT_DIR}/"
  cp "${SHARD_DIR}"/case_*_*/*_candidates.json "${OUT_DIR}/"
  cp "${SHARD_DIR}/case_0_"*/method.json "${OUT_DIR}/method.json"
  awk 'FNR==1 && NR!=1 {next} {print}' "${SHARD_DIR}"/case_*_*/cnop_summary.csv > "${OUT_DIR}/cnop_summary.csv"
  awk 'FNR==1 && NR!=1 {next} {print}' "${SHARD_DIR}"/case_*_*/cnop_candidate_summary.csv > "${OUT_DIR}/cnop_candidate_summary.csv"
}

echo "[cnop64] begin: $(date)"
echo "[cnop64] sample cases"
CUDA_VISIBLE_DEVICES=0 "${PYTHON}" -u scripts/cnop/sample_cnop_cases_by_baseline.py \
  --config "${CONFIG}" \
  --checkpoint "${CHECKPOINT}" \
  --split test \
  --device cuda \
  --output-dir "${SAMPLE_DIR}" \
  --num-cases 64 \
  --horizon 12 \
  --neutral-threshold 0.5 \
  --case-year-range 1851,2014 \
  --max-per-source 16 \
  --forecast-climatology-cache "${FORECAST_CLIM}" \
  --climatology-batch-size 2

echo "[cnop64] run 64 CNOP cases"
mapfile -t CASES < "${SAMPLE_DIR}/selected_cases_for_bash.txt"
total="${#CASES[@]}"
batch_size=8
for ((start=0; start<total; start+=batch_size)); do
  wait_for_gpu_memory 13000
  pids=()
  for ((offset=0; offset<batch_size && start+offset<total; offset++)); do
    idx=$((start + offset))
    read -r source year gpu <<< "${CASES[$idx]}"
    echo "[cnop64] launch case=${idx}/${total} source=${source} year=${year} gpu=${gpu}"
    run_case "${idx}" "${source}" "${year}" "${gpu}" &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do
    wait "${pid}"
  done
done

merge_outputs

echo "[cnop64] recompute forecast-clim summary"
CUDA_VISIBLE_DEVICES=0 "${PYTHON}" -u scripts/cnop/recompute_cnop_summary_forecast_clim.py \
  --config "${CONFIG}" \
  --checkpoint "${CHECKPOINT}" \
  --cnop-dir "${OUT_DIR}" \
  --forecast-climatology-cache "${FORECAST_CLIM}" \
  --split test \
  --device cuda \
  --horizon 12 \
  --lead-month 12

echo "[cnop64] cluster 64 cases and build representative subset"
"${PYTHON}" -u scripts/cnop/cluster_cnop_cases.py \
  --cnop-dir "${OUT_DIR}" \
  --summary-name cnop_summary_forecast_clim.csv \
  --representative-count 10 \
  --cluster-count 4

echo "[cnop64] plot representative 10 cases"
CUDA_VISIBLE_DEVICES=0 "${PYTHON}" -u scripts/cnop/plot_cnop_ten_case_lead12.py \
  --config "${CONFIG}" \
  --checkpoint "${CHECKPOINT}" \
  --cnop-dir "${OUT_DIR}/representative_10" \
  --split test \
  --device cuda \
  --candidate-rank 1 \
  --horizon 12 \
  --lead-month 12 \
  --second-column truth \
  --tos-mode anomaly \
  --forecast-climatology train \
  --forecast-climatology-cache "${FORECAST_CLIM}" \
  --climatology-batch-size 2 \
  --require-cases 10 \
  --max-cases 10 \
  --smooth-sigma 1.2 \
  --dpi 360 \
  --output "${OUT_DIR}/figures/cnop_representative10_scale04_forecast_clim_fixed.png"

echo "[cnop64] all done: $(date)"
