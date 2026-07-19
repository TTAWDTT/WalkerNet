#!/usr/bin/env bash
set -euo pipefail

cd /mnt/sda/WalkerNet

LOG_DIR="/mnt/sda/WalkerNet/outputs/logs"
mkdir -p "${LOG_DIR}"

echo "[run] start CNOP forecast climatology + constraint: $(date)"
echo "[run] host: $(hostname)"

CUDA_VISIBLE_DEVICES=0 /home/cpji/wwb/torch/bin/python -u scripts/cnop/plot_cnop_ten_case_lead12.py \
  --config configs/server_3090_mixed5_ddp8.yaml \
  --checkpoint /mnt/sda/WalkerNet/checkpoints_mixed5_enso18_simple_loss_corr_ddp8/best_skill.pt \
  --cnop-dir outputs/cnop_tos_zos_patch_0703 \
  --split test \
  --device cuda \
  --candidate-rank 1 \
  --horizon 12 \
  --lead-month 12 \
  --second-column truth \
  --tos-mode anomaly \
  --forecast-climatology train \
  --forecast-climatology-cache outputs/cnop_tos_zos_patch_0703/forecast_tos_climatology_train_h12.npz \
  --climatology-batch-size 2 \
  --require-cases 10 \
  --max-cases 10 \
  --smooth-sigma 1.2 \
  --dpi 360 \
  --output outputs/cnop_tos_zos_patch_0703/figures/cnop_ten_case_lead12_forecast_clim_ssta.png

/home/cpji/wwb/torch/bin/python -u scripts/cnop/compute_cnop_constraint.py \
  --config configs/server_3090_mixed5_ddp8.yaml \
  --split train \
  --event-year-range train \
  --event-threshold 0.5 \
  --normalization december_anomaly_train_std_equal_rms \
  --output-dir outputs/cnop_constraint_0705

echo "[run] done: $(date)"
