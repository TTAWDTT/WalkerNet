#!/usr/bin/env bash
set -Eeuo pipefail

# Pre-registered global-domain pilot.  It intentionally requires an explicit
# held-out case manifest and never chooses cases at run time.
ROOT="${WALKERNET_ROOT:-/mnt/sda/WalkerNet}"
PYTHON="${WALKERNET_PYTHON:-/home/cpji/wwb/torch/bin/python}"
CONFIG="${WALKERNET_CONFIG:-${ROOT}/configs/server_3090_mixed5_ddp8.yaml}"
CHECKPOINT="${WALKERNET_CHECKPOINT:-${ROOT}/checkpoints_mixed5_enso18_simple_loss_corr_ddp8/best_skill.pt}"
CASE_MANIFEST="${WALKERNET_CASE_MANIFEST:?Set WALKERNET_CASE_MANIFEST to the approved CSV case list.}"
CONSTRAINT_ROOT="${WALKERNET_CONSTRAINT_ROOT:-${ROOT}/outputs/cnop_basin_constraints_0817}"
OUT="${WALKERNET_OUTPUT_DIR:?Set WALKERNET_OUTPUT_DIR for this pilot.}"
LOG_DIR="${WALKERNET_LOG_DIR:-${OUT}/logs}"
SCALES="${WALKERNET_CONSTRAINT_SCALES:-0.05 0.10 0.20}"
GPU_IDS_TEXT="${WALKERNET_GPU_IDS:-0 1 2 3 5 6 7}"

mkdir -p "${OUT}" "${LOG_DIR}" "${CONSTRAINT_ROOT}/global"
cd "${ROOT}"

if [[ ! -s "${CASE_MANIFEST}" ]]; then
  echo "Case manifest is missing or empty: ${CASE_MANIFEST}" >&2
  exit 2
fi
read -r -a GPU_IDS <<< "${GPU_IDS_TEXT}"
if (( ${#GPU_IDS[@]} == 0 )); then
  echo "WALKERNET_GPU_IDS must contain at least one GPU index" >&2
  exit 2
fi
for gpu in "${GPU_IDS[@]}"; do
  if ! [[ "${gpu}" =~ ^[0-7]$ ]]; then
    echo "Invalid GPU index in WALKERNET_GPU_IDS: ${gpu}" >&2
    exit 2
  fi
done
MAX_PARALLEL="${WALKERNET_MAX_PARALLEL:-${#GPU_IDS[@]}}"
if (( MAX_PARALLEL < 1 || MAX_PARALLEL > ${#GPU_IDS[@]} )); then
  echo "WALKERNET_MAX_PARALLEL must be between 1 and ${#GPU_IDS[@]}" >&2
  exit 2
fi

CONSTRAINT_FILE="${CONSTRAINT_ROOT}/global/cnop_constraint_summary.json"
if [[ ! -s "${CONSTRAINT_FILE}" ]]; then
  "${PYTHON}" -u scripts/cnop/compute_cnop_constraint.py \
    --config "${CONFIG}" --split train --event-threshold 0.5 --event-year-range train \
    --normalization december_anomaly_train_std_equal_rms --domain global --basin-lat-bounds=-60,60 \
    --output-dir "${CONSTRAINT_ROOT}/global" > "${LOG_DIR}/constraint_global.log" 2>&1
fi

readarray -t CASES < <(awk -F, 'NR > 1 && $1 !~ /^#/ && NF >= 2 {gsub(/^[ \t]+|[ \t]+$/, "", $1); gsub(/^[ \t]+|[ \t]+$/, "", $2); print $1 "," $2}' "${CASE_MANIFEST}")
if (( ${#CASES[@]} == 0 )); then
  echo "No usable cases in ${CASE_MANIFEST}" >&2
  exit 2
fi

scale_tag() { printf 'scale_%s' "${1/./p}"; }
case_tag() { printf '%s_%s' "${1//\//_}" "$2"; }
wait_batch() { local pid; for pid in "${PIDS[@]}"; do wait "${pid}"; done; PIDS=(); }
all_selected_gpus_idle() {
  local gpu busy
  for gpu in "${GPU_IDS[@]}"; do
    busy=$(nvidia-smi -i "${gpu}" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | sed '/^[[:space:]]*$/d' | wc -l)
    if [[ "${busy}" -ne 0 ]]; then
      return 1
    fi
  done
  return 0
}
wait_for_all_gpus() {
  local stable=0
  while (( stable < 3 )); do
    if all_selected_gpus_idle; then
      stable=$((stable + 1))
      echo "[pilot] all GPUs idle check ${stable}/3 $(date -Is)"
    else
      stable=0
      echo "[pilot] waiting for all GPUs $(date -Is)"
    fi
    sleep 60
  done
}

run_gradient() {
  local scale="$1" source="$2" year="$3" gpu="$4" case_dir="$5"
  mkdir -p "${case_dir}/gradient"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" -u scripts/cnop/evaluate_basin_zero_state_gradient.py \
    --config "${CONFIG}" --checkpoint "${CHECKPOINT}" --constraint-file "${CONSTRAINT_FILE}" \
    --constraint-scale "${scale}" --domain global --basin-lat-bounds=-60,60 \
    --case-source-name "${source}" --case-target-year "${year}" --device cuda \
    --perturb-grid patch --perturb-patch-size 4 --max-abs 2.0 --output-dir "${case_dir}/gradient" \
    > "${LOG_DIR}/$(basename "${case_dir}")_$(basename "$(dirname "${case_dir}")")_gradient.log" 2>&1
}

run_random() {
  local scale="$1" source="$2" year="$3" gpu="$4" case_dir="$5"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" -u scripts/cnop/evaluate_basin_random_controls.py \
    --config "${CONFIG}" --checkpoint "${CHECKPOINT}" --constraint-file "${CONSTRAINT_FILE}" \
    --constraint-scale "${scale}" --domain global --basin-lat-bounds=-60,60 \
    --case-source-name "${source}" --case-target-year "${year}" --num-controls 128 --device cuda \
    --output "${case_dir}/random_controls.csv" \
    > "${LOG_DIR}/$(basename "${case_dir}")_$(basename "$(dirname "${case_dir}")")_random.log" 2>&1
}

run_cnop() {
  local scale="$1" source="$2" year="$3" gpu="$4" case_dir="$5"
  mkdir -p "${case_dir}/cnop"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" -u scripts/cnop/compute_tos_zos_cnop.py \
    --config "${CONFIG}" --checkpoint "${CHECKPOINT}" --split test \
    --case-source-name "${source}" --case-target-year "${year}" --device cuda --num-cases 1 \
    --horizon 12 --steps 1000 --lr 0.08 --optimizer-mode accepted_adam --acceptance-tolerance 0.0 \
    --backtrack-factor 0.5 --min-lr 0.0001 --num-starts 8 --top-k 5 \
    --candidate-max-cosine-similarity 0.98 --random-init-scale 0.02 \
    --constraint-mode event_l2 --constraint-file "${CONSTRAINT_FILE}" --constraint-scale "${scale}" \
    --max-abs 2.0 --domain global --basin-lat-bounds=-60,60 --perturb-grid patch --perturb-patch-size 4 \
    --objective-mode late_3m_delta --smoothness-weight 0.001 --seed 42 \
    --warm-start-npz "${case_dir}/gradient/case_${source}_${year}.npz" --output-dir "${case_dir}/cnop" \
    > "${LOG_DIR}/$(basename "${case_dir}")_$(basename "$(dirname "${case_dir}")")_cnop.log" 2>&1
}

run_stage() {
  local stage="$1"; shift
  local gpu_index=0 gpu
  PIDS=()
  for scale in ${SCALES}; do
    for case in "${CASES[@]}"; do
      IFS=, read -r source year <<< "${case}"
      case_dir="${OUT}/$(scale_tag "${scale}")/$(case_tag "${source}" "${year}")"
      gpu="${GPU_IDS[${gpu_index}]}"
      "run_${stage}" "${scale}" "${source}" "${year}" "${gpu}" "${case_dir}" &
      PIDS+=("$!")
      gpu_index=$(( (gpu_index + 1) % ${#GPU_IDS[@]} ))
      if (( ${#PIDS[@]} >= MAX_PARALLEL )); then wait_batch; fi
    done
  done
  wait_batch
}

echo "[pilot] begin $(date -Is) cases=${#CASES[@]} scales=${SCALES}"
wait_for_all_gpus
run_stage gradient
run_stage random
run_stage cnop
"${PYTHON}" -u scripts/cnop/summarize_constraint_scale_pilot.py --experiment-dir "${OUT}" --output-dir "${OUT}/summary" \
  > "${LOG_DIR}/summary.log" 2>&1
echo "[pilot] complete $(date -Is)"
