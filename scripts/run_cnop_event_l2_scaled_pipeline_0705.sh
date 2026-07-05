#!/usr/bin/env bash
set -euo pipefail

cd /mnt/sda/WalkerNet

LOG_DIR="/mnt/sda/WalkerNet/outputs/logs"
BASE_DIR="/mnt/sda/WalkerNet/outputs/cnop_event_l2_constraint_0705"
CONSTRAINT_JSON="/mnt/sda/WalkerNet/outputs/cnop_constraint_0705/cnop_constraint_summary.json"
FORECAST_CLIM="/mnt/sda/WalkerNet/outputs/cnop_tos_zos_patch_0703/forecast_tos_climatology_train_h12.npz"
CONFIG="configs/server_3090_mixed5_ddp8.yaml"
CHECKPOINT="/mnt/sda/WalkerNet/checkpoints_mixed5_enso18_simple_loss_corr_ddp8/best_skill.pt"
mkdir -p "${LOG_DIR}"

CASES=(
  "IPSL-CM6A-LR 1880 0"
  "EC-Earth3 1959 1"
  "EC-Earth3 1942 2"
  "EC-Earth3 1975 3"
  "MPI-ESM1-2-HR 1876 4"
  "GFDL-ESM4 1995 5"
  "MPI-ESM1-2-HR 2003 6"
  "CESM2 1856 7"
  "EC-Earth3 1878 0"
  "GFDL-ESM4 1930 1"
)

plot_and_recompute() {
  local out_dir="$1"
  local tag="$2"
  local figure="${out_dir}/figures/cnop_ten_case_lead12_${tag}_forecast_clim_fixed.png"
  mkdir -p "${out_dir}/figures"

  echo "[pipeline] recompute summary ${tag}: $(date)"
  CUDA_VISIBLE_DEVICES=0 /home/cpji/wwb/torch/bin/python -u scripts/recompute_cnop_summary_forecast_clim.py \
    --config "${CONFIG}" \
    --checkpoint "${CHECKPOINT}" \
    --cnop-dir "${out_dir}" \
    --forecast-climatology-cache "${FORECAST_CLIM}" \
    --split test \
    --device cuda \
    --horizon 12 \
    --lead-month 12

  echo "[pipeline] plot ${tag}: $(date)"
  CUDA_VISIBLE_DEVICES=0 /home/cpji/wwb/torch/bin/python -u scripts/plot_cnop_ten_case_lead12.py \
    --config "${CONFIG}" \
    --checkpoint "${CHECKPOINT}" \
    --cnop-dir "${out_dir}" \
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
    --output "${figure}"
}

run_case() {
  local idx="$1"
  local source="$2"
  local year="$3"
  local gpu="$4"
  local scale="$5"
  local out_dir="$6"
  local scale_tag="$7"
  local shard_dir="${out_dir}/shards"
  local case_dir="${shard_dir}/case_${idx}_${source}_${year}"
  rm -rf "${case_dir}"
  mkdir -p "${case_dir}"

  CUDA_VISIBLE_DEVICES="${gpu}" /home/cpji/wwb/torch/bin/python -u scripts/compute_tos_zos_cnop.py \
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
    --constraint-scale "${scale}" \
    --max-abs 2.0 \
    --neutral-threshold 0.5 \
    --domain tropical_pacific \
    --perturb-grid patch \
    --perturb-patch-size 4 \
    --objective-mode lead_delta \
    --objective-lead 12 \
    --smoothness-weight 0.001 \
    --output-dir "${case_dir}" \
    > "${LOG_DIR}/cnop_event_l2_${scale_tag}_case_${idx}_${source}_${year}.log" 2>&1
}

merge_scale_outputs() {
  local out_dir="$1"
  local shard_dir="${out_dir}/shards"
  mkdir -p "${out_dir}/figures"
  cp "${shard_dir}"/case_*_*/case_*.npz "${out_dir}/"
  cp "${shard_dir}"/case_*_*/*_history.json "${out_dir}/"
  cp "${shard_dir}"/case_*_*/*_candidates.json "${out_dir}/"
  cp "${shard_dir}/case_0_"*/method.json "${out_dir}/method.json"
  awk 'FNR==1 && NR!=1 {next} {print}' "${shard_dir}"/case_*_*/cnop_summary.csv > "${out_dir}/cnop_summary.csv"
  awk 'FNR==1 && NR!=1 {next} {print}' "${shard_dir}"/case_*_*/cnop_candidate_summary.csv > "${out_dir}/cnop_candidate_summary.csv"
}

run_scale() {
  local scale="$1"
  local scale_tag="$2"
  local out_dir="/mnt/sda/WalkerNet/outputs/cnop_event_l2_constraint_${scale_tag}_0705"
  local shard_dir="${out_dir}/shards"
  rm -rf "${out_dir}"
  mkdir -p "${shard_dir}"

  echo "[pipeline] start scale=${scale} tag=${scale_tag}: $(date)"
  local pids=()
  for idx in 0 1 2 3 4 5 6 7; do
    read -r source year gpu <<< "${CASES[$idx]}"
    echo "[pipeline] launch ${scale_tag} case=${idx} source=${source} year=${year} gpu=${gpu}"
    run_case "${idx}" "${source}" "${year}" "${gpu}" "${scale}" "${out_dir}" "${scale_tag}" &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do
    wait "${pid}"
  done

  pids=()
  for idx in 8 9; do
    read -r source year gpu <<< "${CASES[$idx]}"
    echo "[pipeline] delayed ${scale_tag} case=${idx} source=${source} year=${year} gpu=${gpu}"
    run_case "${idx}" "${source}" "${year}" "${gpu}" "${scale}" "${out_dir}" "${scale_tag}" &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do
    wait "${pid}"
  done

  echo "[pipeline] merge scale=${scale_tag}: $(date)"
  merge_scale_outputs "${out_dir}"
  plot_and_recompute "${out_dir}" "${scale_tag}"
  echo "[pipeline] done scale=${scale_tag}: $(date)"
}

echo "[pipeline] begin all requested tasks: $(date)"
plot_and_recompute "${BASE_DIR}" "scale10"
run_scale "0.3" "scale03"
run_scale "0.2" "scale02"
run_scale "0.4" "scale04"
echo "[pipeline] all done: $(date)"
