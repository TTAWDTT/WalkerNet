#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="/mnt/sda/WalkerNet"
PYTHON="/home/cpji/wwb/torch/bin/python"
CONFIG="${ROOT}/configs/server_3090_mixed5_ddp8.yaml"
CHECKPOINT="${ROOT}/checkpoints_mixed5_enso18_simple_loss_corr_ddp8/best_skill.pt"
CONSTRAINT="${ROOT}/outputs/cnop_constraint_0705/cnop_constraint_summary.json"
FORECAST_CLIM="${ROOT}/outputs/cnop_tos_zos_patch_0703/forecast_tos_climatology_train_h12.npz"
OUT="${ROOT}/outputs/cnop_basin_gfdl1995_scale01_steps1000_0816"
LOG_DIR="${ROOT}/outputs/logs/cnop_basin_gfdl1995_scale01_steps1000_0816"

mkdir -p "${OUT}/shards" "${OUT}/combined" "${LOG_DIR}"
cd "${ROOT}"

run_shard() {
  local domain="$1"
  local shard="$2"
  local offset="$3"
  local gpu="$4"
  local shard_dir="${OUT}/shards/${domain}/shard_${shard}"
  local log_file="${LOG_DIR}/${domain}_shard_${shard}.log"
  mkdir -p "${shard_dir}"
  if [[ -s "${shard_dir}/cnop_summary.csv" ]]; then
    echo "[basin-cnop] skip completed ${domain} shard=${shard}"
    return 0
  fi
  echo "[basin-cnop] start ${domain} shard=${shard} gpu=${gpu} offset=${offset} time=$(date -Is)"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" -u scripts/cnop/compute_tos_zos_cnop.py \
    --config "${CONFIG}" \
    --checkpoint "${CHECKPOINT}" \
    --split test \
    --case-source-name GFDL-ESM4 \
    --case-target-year 1995 \
    --device cuda \
    --num-cases 1 \
    --horizon 12 \
    --steps 1000 \
    --lr 0.08 \
    --num-starts 8 \
    --start-index-offset "${offset}" \
    --top-k 5 \
    --candidate-max-cosine-similarity 0.98 \
    --random-init-scale 0.02 \
    --constraint-mode event_l2 \
    --constraint-file "${CONSTRAINT}" \
    --constraint-scale 0.1 \
    --max-abs 2.0 \
    --domain "${domain}" \
    --basin-lat-bounds=-60,60 \
    --perturb-grid patch \
    --perturb-patch-size 4 \
    --objective-mode late_3m_delta \
    --smoothness-weight 0.001 \
    --seed 42 \
    --output-dir "${shard_dir}" \
    > "${log_file}" 2>&1
  echo "[basin-cnop] finish ${domain} shard=${shard} time=$(date -Is)"
}

run_random_controls() {
  local domain="$1"
  local gpu="$2"
  local output="${OUT}/random_controls/${domain}.csv"
  local log_file="${LOG_DIR}/${domain}_random_controls.log"
  if [[ -s "${output}" ]]; then
    echo "[basin-cnop] skip completed random controls ${domain}"
    return 0
  fi
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" -u scripts/cnop/evaluate_basin_random_controls.py \
    --config "${CONFIG}" \
    --checkpoint "${CHECKPOINT}" \
    --constraint-file "${CONSTRAINT}" \
    --constraint-scale 0.1 \
    --domain "${domain}" \
    --case-source-name GFDL-ESM4 \
    --case-target-year 1995 \
    --num-controls 128 \
    --device cuda \
    --output "${output}" \
    > "${log_file}" 2>&1
}

echo "[basin-cnop] experiment begin $(date -Is)"
pids=()
run_shard pacific a 0 0 & pids+=("$!")
run_shard pacific b 8 1 & pids+=("$!")
run_shard atlantic_indian a 0 2 & pids+=("$!")
run_shard atlantic_indian b 8 3 & pids+=("$!")
run_shard global a 0 4 & pids+=("$!")
run_shard global b 8 7 & pids+=("$!")
for pid in "${pids[@]}"; do
  wait "${pid}"
done

# 优化分片完成后再做随机对照，避免与同卡上的 CNOP 优化争抢显存。
run_random_controls pacific 0
run_random_controls atlantic_indian 0
run_random_controls global 0

"${PYTHON}" -u scripts/cnop/aggregate_basin_cnop_shards.py \
  --input-dir "${OUT}/shards" \
  --output-dir "${OUT}/combined" \
  --top-k 5 \
  --max-cosine-similarity 0.98

for domain in pacific atlantic_indian global; do
  CUDA_VISIBLE_DEVICES=0 "${PYTHON}" -u scripts/cnop/recompute_cnop_summary_forecast_clim.py \
    --config "${CONFIG}" \
    --checkpoint "${CHECKPOINT}" \
    --cnop-dir "${OUT}/combined/${domain}" \
    --forecast-climatology-cache "${FORECAST_CLIM}" \
    --split test \
    --device cuda \
    --horizon 12 \
    --lead-month 12
done

echo "[basin-cnop] experiment complete $(date -Is)"
