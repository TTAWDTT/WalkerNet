#!/usr/bin/env bash
set -Eeuo pipefail

# Defaults preserve the original training host.  A portable deployment may
# override all location-dependent paths through WALKERNET_* environment vars.
ROOT="${WALKERNET_ROOT:-/mnt/sda/WalkerNet}"
PYTHON="${WALKERNET_PYTHON:-/home/cpji/wwb/torch/bin/python}"
CONFIG="${WALKERNET_CONFIG:-${ROOT}/configs/server_3090_mixed5_ddp8.yaml}"
CHECKPOINT="${WALKERNET_CHECKPOINT:-${ROOT}/checkpoints_mixed5_enso18_simple_loss_corr_ddp8/best_skill.pt}"
FORECAST_CLIM="${WALKERNET_FORECAST_CLIM:-${ROOT}/outputs/cnop_tos_zos_patch_0703/forecast_tos_climatology_train_h12.npz}"
CONSTRAINT_ROOT="${WALKERNET_CONSTRAINT_ROOT:-${ROOT}/outputs/cnop_basin_constraints_0817}"
OUT="${WALKERNET_OUTPUT_DIR:-${ROOT}/outputs/cnop_basin_gfdl1995_clim_scale01_steps1000_0817}"
LOG_DIR="${WALKERNET_LOG_DIR:-${ROOT}/outputs/logs/cnop_basin_gfdl1995_clim_scale01_steps1000_0817}"

mkdir -p "${CONSTRAINT_ROOT}" "${OUT}/shards" "${OUT}/combined" "${OUT}/random_controls" "${OUT}/gradient_baseline" "${LOG_DIR}"
cd "${ROOT}"

constraint_file() {
  printf '%s/%s/cnop_constraint_summary.json' "${CONSTRAINT_ROOT}" "$1"
}

compute_constraint() {
  local domain="$1"
  local output_dir="${CONSTRAINT_ROOT}/${domain}"
  local summary="${output_dir}/cnop_constraint_summary.json"
  if [[ -s "${summary}" ]]; then
    echo "[basin-clim] skip constraint ${domain}"
    return
  fi
  mkdir -p "${output_dir}"
  "${PYTHON}" -u scripts/cnop/compute_cnop_constraint.py \
    --config "${CONFIG}" \
    --split train \
    --event-threshold 0.5 \
    --event-year-range train \
    --normalization december_anomaly_train_std_equal_rms \
    --domain "${domain}" \
    --basin-lat-bounds=-60,60 \
    --output-dir "${output_dir}" \
    > "${LOG_DIR}/constraint_${domain}.log" 2>&1
  echo "[basin-clim] constraint ${domain} ready"
}

all_gpus_idle() {
  local busy
  busy=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | sed '/^[[:space:]]*$/d' | wc -l)
  [[ "${busy}" -eq 0 ]]
}

wait_for_all_gpus() {
  local stable=0
  while (( stable < 3 )); do
    if all_gpus_idle; then
      stable=$((stable + 1))
      echo "[basin-clim] all GPUs idle check ${stable}/3 $(date -Is)"
    else
      stable=0
      echo "[basin-clim] waiting for all GPUs $(date -Is)"
    fi
    sleep 60
  done
}

run_shard() {
  local domain="$1"
  local shard="$2"
  local offset="$3"
  local gpu="$4"
  shift 4
  local shard_dir="${OUT}/shards/${domain}/shard_${shard}"
  local log_file="${LOG_DIR}/${domain}_shard_${shard}.log"
  mkdir -p "${shard_dir}"
  if [[ -s "${shard_dir}/cnop_summary.csv" ]]; then
    echo "[basin-clim] skip completed ${domain} shard=${shard}"
    return
  fi
  echo "[basin-clim] start ${domain} shard=${shard} gpu=${gpu} offset=${offset} $(date -Is)"
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
    --constraint-file "$(constraint_file "${domain}")" \
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
    "$@" \
    > "${log_file}" 2>&1
  echo "[basin-clim] finish ${domain} shard=${shard} $(date -Is)"
}

aggregate_domains() {
  local domains="$1"
  "${PYTHON}" -u scripts/cnop/aggregate_basin_cnop_shards.py \
    --input-dir "${OUT}/shards" \
    --output-dir "${OUT}/combined" \
    --domains "${domains}" \
    --top-k 5 \
    --max-cosine-similarity 0.98
}

run_random_controls() {
  local domain="$1"
  local gpu="$2"
  local output="${OUT}/random_controls/${domain}.csv"
  if [[ -s "${output}" ]]; then
    return
  fi
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" -u scripts/cnop/evaluate_basin_random_controls.py \
    --config "${CONFIG}" \
    --checkpoint "${CHECKPOINT}" \
    --constraint-file "$(constraint_file "${domain}")" \
    --constraint-scale 0.1 \
    --domain "${domain}" \
    --basin-lat-bounds=-60,60 \
    --case-source-name GFDL-ESM4 \
    --case-target-year 1995 \
    --num-controls 128 \
    --device cuda \
    --output "${output}" \
    > "${LOG_DIR}/${domain}_random_controls.log" 2>&1
}

run_zero_state_gradient() {
  local domain="$1"
  local gpu="$2"
  local output_dir="${OUT}/gradient_baseline/${domain}"
  if [[ -s "${output_dir}/gradient_summary.csv" ]]; then
    return
  fi
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" -u scripts/cnop/evaluate_basin_zero_state_gradient.py \
    --config "${CONFIG}" \
    --checkpoint "${CHECKPOINT}" \
    --constraint-file "$(constraint_file "${domain}")" \
    --constraint-scale 0.1 \
    --domain "${domain}" \
    --basin-lat-bounds=-60,60 \
    --case-source-name GFDL-ESM4 \
    --case-target-year 1995 \
    --device cuda \
    --perturb-grid patch \
    --perturb-patch-size 4 \
    --max-abs 2.0 \
    --output-dir "${output_dir}" \
    > "${LOG_DIR}/${domain}_zero_state_gradient.log" 2>&1
}

echo "[basin-clim] pipeline begin $(date -Is)"
for domain in pacific atlantic_indian global; do
  compute_constraint "${domain}"
done

wait_for_all_gpus

pids=()
run_shard pacific a 0 0 & pids+=("$!")
run_shard pacific b 8 1 & pids+=("$!")
run_shard atlantic_indian a 0 2 & pids+=("$!")
run_shard atlantic_indian b 8 3 & pids+=("$!")
for pid in "${pids[@]}"; do wait "${pid}"; done

aggregate_domains "pacific,atlantic_indian"
PACIFIC_WARM="${OUT}/combined/pacific/case_GFDL-ESM4_1995.npz"
REMOTE_WARM="${OUT}/combined/atlantic_indian/case_GFDL-ESM4_1995.npz"

pids=()
for gpu in 0 1 2 3 4 5 6 7; do
  offset=$((gpu * 8))
  run_shard global "${gpu}" "${offset}" "${gpu}" \
    --warm-start-npz "${PACIFIC_WARM}" \
    --warm-start-npz "${REMOTE_WARM}" &
  pids+=("$!")
done
for pid in "${pids[@]}"; do wait "${pid}"; done

pids=()
run_random_controls pacific 0 & pids+=("$!")
run_random_controls atlantic_indian 1 & pids+=("$!")
run_random_controls global 2 & pids+=("$!")
run_zero_state_gradient pacific 3 & pids+=("$!")
run_zero_state_gradient atlantic_indian 4 & pids+=("$!")
run_zero_state_gradient global 5 & pids+=("$!")
for pid in "${pids[@]}"; do wait "${pid}"; done

aggregate_domains "pacific,atlantic_indian,global"
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

CUDA_VISIBLE_DEVICES=0 "${PYTHON}" -u scripts/cnop/plot_basin_cnop_experiment.py \
  --config "${CONFIG}" \
  --checkpoint "${CHECKPOINT}" \
  --experiment-dir "${OUT}" \
  --forecast-climatology-cache "${FORECAST_CLIM}" \
  --output-dir "${OUT}/figures_paper" \
  --split test \
  --device cuda \
  --horizon 12 \
  --lead-month 12 \
  --dpi 320 \
  > "${LOG_DIR}/plot_paper.log" 2>&1

"${PYTHON}" -u scripts/cnop/plot_basin_cnop_gradient_comparison.py \
  --experiment-dir "${OUT}" \
  --output-dir "${OUT}/figures_paper" \
  --metric objective \
  --dpi 320 \
  > "${LOG_DIR}/plot_gradient_comparison.log" 2>&1

echo "[basin-clim] pipeline complete $(date -Is)"
