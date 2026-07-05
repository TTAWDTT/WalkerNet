#!/usr/bin/env bash
set -euo pipefail

cd /mnt/sda/WalkerNet

LOG_DIR="/mnt/sda/WalkerNet/outputs/logs"
OUT_DIR="/mnt/sda/WalkerNet/outputs/cnop_event_l2_constraint_0705"
SHARD_DIR="${OUT_DIR}/shards"
CONSTRAINT_JSON="/mnt/sda/WalkerNet/outputs/cnop_constraint_0705/cnop_constraint_summary.json"
FORECAST_CLIM="/mnt/sda/WalkerNet/outputs/cnop_tos_zos_patch_0703/forecast_tos_climatology_train_h12.npz"
FIG_PATH="${OUT_DIR}/figures/cnop_ten_case_lead12_event_l2_forecast_clim_ssta.png"
mkdir -p "${LOG_DIR}"

rm -rf "${OUT_DIR}"
mkdir -p "${SHARD_DIR}" "${OUT_DIR}/figures"

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

echo "[parallel] start 10 CNOP cases: $(date)"
PIDS=()
launch_case() {
  local idx="$1"
  local source="$2"
  local year="$3"
  local gpu="$4"
  local case_dir="${SHARD_DIR}/case_${idx}_${source}_${year}"
  mkdir -p "${case_dir}"
  echo "[parallel] launch case=${idx} source=${source} year=${year} gpu=${gpu}"
  (
    CUDA_VISIBLE_DEVICES="${gpu}" /home/cpji/wwb/torch/bin/python -u scripts/compute_tos_zos_cnop.py \
      --config configs/server_3090_mixed5_ddp8.yaml \
      --checkpoint /mnt/sda/WalkerNet/checkpoints_mixed5_enso18_simple_loss_corr_ddp8/best_skill.pt \
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
      --constraint-scale 1.0 \
      --max-abs 2.0 \
      --neutral-threshold 0.5 \
      --domain tropical_pacific \
      --perturb-grid patch \
      --perturb-patch-size 4 \
      --objective-mode lead_delta \
      --objective-lead 12 \
      --smoothness-weight 0.001 \
      --output-dir "${case_dir}"
  ) > "${LOG_DIR}/cnop_event_l2_case_${idx}_${source}_${year}.log" 2>&1 &
  PIDS+=("$!")
}

for idx in 0 1 2 3 4 5 6 7; do
  read -r SOURCE YEAR GPU <<< "${CASES[$idx]}"
  launch_case "${idx}" "${SOURCE}" "${YEAR}" "${GPU}"
done

FAILED=0
for pid in "${PIDS[@]}"; do
  if ! wait "${pid}"; then
    FAILED=1
  fi
done
if [[ "${FAILED}" -ne 0 ]]; then
  echo "[parallel] at least one case failed; check ${LOG_DIR}/cnop_event_l2_case_*.log" >&2
  exit 1
fi

PIDS=()
for idx in 8 9; do
  read -r SOURCE YEAR GPU <<< "${CASES[$idx]}"
  launch_case "${idx}" "${SOURCE}" "${YEAR}" "${GPU}"
done

for pid in "${PIDS[@]}"; do
  if ! wait "${pid}"; then
    FAILED=1
  fi
done
if [[ "${FAILED}" -ne 0 ]]; then
  echo "[parallel] delayed cases failed; check ${LOG_DIR}/cnop_event_l2_case_*.log" >&2
  exit 1
fi

echo "[parallel] merge outputs: $(date)"
cp "${SHARD_DIR}"/case_*_*/case_*.npz "${OUT_DIR}/"
cp "${SHARD_DIR}"/case_*_*/*_history.json "${OUT_DIR}/"
cp "${SHARD_DIR}"/case_*_*/*_candidates.json "${OUT_DIR}/"
cp "${SHARD_DIR}/case_0_"*/method.json "${OUT_DIR}/method.json"

awk 'FNR==1 && NR!=1 {next} {print}' "${SHARD_DIR}"/case_*_*/cnop_summary.csv > "${OUT_DIR}/cnop_summary.csv"
awk 'FNR==1 && NR!=1 {next} {print}' "${SHARD_DIR}"/case_*_*/cnop_candidate_summary.csv > "${OUT_DIR}/cnop_candidate_summary.csv"

echo "[parallel] plot merged result: $(date)"
CUDA_VISIBLE_DEVICES=0 /home/cpji/wwb/torch/bin/python -u scripts/plot_cnop_ten_case_lead12.py \
  --config configs/server_3090_mixed5_ddp8.yaml \
  --checkpoint /mnt/sda/WalkerNet/checkpoints_mixed5_enso18_simple_loss_corr_ddp8/best_skill.pt \
  --cnop-dir "${OUT_DIR}" \
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
  --output "${FIG_PATH}"

echo "[parallel] done: $(date)"
echo "[parallel] figure: ${FIG_PATH}"
