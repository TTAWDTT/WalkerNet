#!/usr/bin/env bash
set -euo pipefail

cd /mnt/sda/WalkerNet

LOG_DIR="/mnt/sda/WalkerNet/outputs/logs"
mkdir -p "${LOG_DIR}"

CONSTRAINT_JSON="/mnt/sda/WalkerNet/outputs/cnop_constraint_0705/cnop_constraint_summary.json"
FORECAST_CLIM="/mnt/sda/WalkerNet/outputs/cnop_tos_zos_patch_0703/forecast_tos_climatology_train_h12.npz"
CNOP_DIR="/mnt/sda/WalkerNet/outputs/cnop_event_l2_constraint_0705"
FIG_PATH="/mnt/sda/WalkerNet/outputs/cnop_event_l2_constraint_0705/figures/cnop_ten_case_lead12_event_l2_forecast_clim_ssta.png"

echo "[run] wait for constraint + forecast climatology: $(date)"
while [[ ! -s "${CONSTRAINT_JSON}" || ! -s "${FORECAST_CLIM}" ]]; do
  echo "[wait] $(date) constraint=$([[ -s "${CONSTRAINT_JSON}" ]] && echo yes || echo no) forecast_clim=$([[ -s "${FORECAST_CLIM}" ]] && echo yes || echo no)"
  sleep 300
done

echo "[run] start event_l2 CNOP: $(date)"
CUDA_VISIBLE_DEVICES=0 /home/cpji/wwb/torch/bin/python -u scripts/cnop/compute_tos_zos_cnop.py \
  --config configs/server_3090_mixed5_ddp8.yaml \
  --checkpoint /mnt/sda/WalkerNet/checkpoints_mixed5_enso18_simple_loss_corr_ddp8/best_skill.pt \
  --split test \
  --case-year-range 1851,2014 \
  --device cuda \
  --num-cases 10 \
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
  --output-dir "${CNOP_DIR}"

echo "[run] plot event_l2 CNOP with forecast climatology: $(date)"
mkdir -p "$(dirname "${FIG_PATH}")"
CUDA_VISIBLE_DEVICES=0 /home/cpji/wwb/torch/bin/python -u scripts/cnop/plot_cnop_ten_case_lead12.py \
  --config configs/server_3090_mixed5_ddp8.yaml \
  --checkpoint /mnt/sda/WalkerNet/checkpoints_mixed5_enso18_simple_loss_corr_ddp8/best_skill.pt \
  --cnop-dir "${CNOP_DIR}" \
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

echo "[run] done event_l2 CNOP + plot: $(date)"
