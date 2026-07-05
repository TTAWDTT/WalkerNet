#!/usr/bin/env bash
set -euo pipefail

cd /mnt/sda/WalkerNet

LOG_DIR="/mnt/sda/WalkerNet/outputs/logs"
OUT_DIR="/mnt/sda/WalkerNet/outputs/cnop_event_l2_constraint_0705"
SHARD_DIR="${OUT_DIR}/shards"
CONSTRAINT_JSON="/mnt/sda/WalkerNet/outputs/cnop_constraint_0705/cnop_constraint_summary.json"
FORECAST_CLIM="/mnt/sda/WalkerNet/outputs/cnop_tos_zos_patch_0703/forecast_tos_climatology_train_h12.npz"
FIG_PATH="${OUT_DIR}/figures/cnop_ten_case_lead12_event_l2_forecast_clim_ssta.png"

run_case() {
  local idx="$1"
  local source="$2"
  local year="$3"
  local gpu="$4"
  local case_dir="${SHARD_DIR}/case_${idx}_${source}_${year}"
  rm -rf "${case_dir}"
  mkdir -p "${case_dir}"
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
}

echo "[rescue] wait for cases 0-7: $(date)"
while true; do
  ready=0
  for idx in 0 1 2 3 4 5 6 7; do
    if compgen -G "${SHARD_DIR}/case_${idx}_*/cnop_summary.csv" > /dev/null; then
      ready=$((ready + 1))
    fi
  done
  echo "[rescue] ready=${ready}/8 $(date)"
  [[ "${ready}" -eq 8 ]] && break
  sleep 300
done

echo "[rescue] run delayed cases 8 and 9: $(date)"
run_case 8 EC-Earth3 1878 0 > "${LOG_DIR}/cnop_event_l2_case_8_EC-Earth3_1878.log" 2>&1 &
pid8="$!"
run_case 9 GFDL-ESM4 1930 1 > "${LOG_DIR}/cnop_event_l2_case_9_GFDL-ESM4_1930.log" 2>&1 &
pid9="$!"
wait "${pid8}"
wait "${pid9}"

echo "[rescue] merge outputs: $(date)"
mkdir -p "${OUT_DIR}/figures"
cp "${SHARD_DIR}"/case_*_*/case_*.npz "${OUT_DIR}/"
cp "${SHARD_DIR}"/case_*_*/*_history.json "${OUT_DIR}/"
cp "${SHARD_DIR}"/case_*_*/*_candidates.json "${OUT_DIR}/"
cp "${SHARD_DIR}/case_0_"*/method.json "${OUT_DIR}/method.json"
awk 'FNR==1 && NR!=1 {next} {print}' "${SHARD_DIR}"/case_*_*/cnop_summary.csv > "${OUT_DIR}/cnop_summary.csv"
awk 'FNR==1 && NR!=1 {next} {print}' "${SHARD_DIR}"/case_*_*/cnop_candidate_summary.csv > "${OUT_DIR}/cnop_candidate_summary.csv"

echo "[rescue] plot merged result: $(date)"
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

echo "[rescue] done: $(date)"
echo "[rescue] figure: ${FIG_PATH}"
