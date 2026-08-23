#!/usr/bin/env bash
set -u

ROOT=/data/WalkerNet
REPO="$ROOT/repo"
PY="$ROOT/venv313/bin/python"
CONFIG="$REPO/configs/server_gpu006_historical_mixed5.yaml"
CHECKPOINT="$ROOT/input/artifacts/historical_mixed5_best_skill.pt"
OUT="${OUT:-$ROOT/outputs/cnop_global_delayed_onset_24starts_steps100_v1}"
CANDIDATE_MAX_SIM="${CANDIDATE_MAX_SIM:-1.0}"
MANIFEST="$ROOT/outputs/cnop_pacific_delayed_onset_24starts_steps100_v2_top3/metadata/formal_manifest_v1.csv"
LOGDIR="$OUT/logs"
mkdir -p "$LOGDIR" "$OUT/delayed" "$OUT/metadata"

if [[ -e "$OUT/RUN_COMPLETE" ]]; then
  echo "refusing to rerun completed experiment: $OUT" >&2
  exit 2
fi

GPUS=(0 1 2 3 4 5 6 7)
declare -a PIDS JOBS
job_count=0
failed=0

slugify() { echo "$1" | tr ' /' '__' | tr -cd '[:alnum:]_.-'; }

launch_job() {
  local source="$1" year="$2" gpu="$3"
  local slug="$(slugify "$source")_${year}"
  local job_dir="$OUT/delayed/$slug"
  local log="$LOGDIR/delayed_${slug}.log"
  mkdir -p "$job_dir"
  echo "[$(date -Is)] start source=$source year=$year gpu=$gpu" >> "$log"
  CUDA_VISIBLE_DEVICES="$gpu" nohup "$PY" -u "$REPO/scripts/cnop/compute_tos_zos_cnop.py" \
    --config "$CONFIG" \
    --checkpoint "$CHECKPOINT" \
    --split train \
    --device cuda \
    --output-dir "$job_dir" \
    --case-source-name "$source" \
    --case-target-year "$year" \
    --horizon 12 \
    --steps 100 \
    --lr 0.08 \
    --num-starts 12 \
    --top-k 3 \
    --candidate-max-cosine-similarity "$CANDIDATE_MAX_SIM" \
    --seed 42 \
    --constraint-mode relative_initial_l2 \
    --relative-l2-fraction 0.03 \
    --domain global \
    --objective-mode delayed_lead_delta \
    --objective-lead 12 \
    --delay-early-leads 3 \
    --delay-early-threshold 0.2 \
    --delay-penalty-weight 2.0 \
    --checkpoint-rollout \
    > "$log" 2>&1 &
  local pid=$!
  echo "$pid" > "$job_dir/job.pid"
  echo "[$(date -Is)] pid=$pid" >> "$log"
  PIDS[$gpu]=$pid
  JOBS[$gpu]="$slug"
  job_count=$((job_count + 1))
}

reap_gpu() {
  local gpu="$1" pid="${PIDS[$gpu]:-}"
  [[ -z "$pid" ]] && return 0
  if wait "$pid"; then
    echo "[$(date -Is)] finished gpu=$gpu job=${JOBS[$gpu]} status=0" >> "$LOGDIR/launcher.log"
  else
    local status=$?
    echo "[$(date -Is)] finished gpu=$gpu job=${JOBS[$gpu]} status=$status" >> "$LOGDIR/launcher.log"
    failed=$((failed + 1))
  fi
  PIDS[$gpu]=
  JOBS[$gpu]=
}

echo "[$(date -Is)] launcher start host=$(hostname) domain=global objective=delayed_lead_delta" | tee "$LOGDIR/launcher.log"
while IFS=, read -r source year target_t observed; do
  [[ "$source" == source || -z "$source" ]] && continue
  gpu=$((job_count % ${#GPUS[@]}))
  while [[ -n "${PIDS[$gpu]:-}" ]] && kill -0 "${PIDS[$gpu]}" 2>/dev/null && ! ps -o stat= -p "${PIDS[$gpu]}" 2>/dev/null | grep -q '^Z'; do sleep 10; done
  [[ -n "${PIDS[$gpu]:-}" ]] && reap_gpu "$gpu"
  launch_job "$source" "$year" "${GPUS[$gpu]}"
done < "$MANIFEST"

for gpu in "${!PIDS[@]}"; do
  [[ -n "${PIDS[$gpu]:-}" ]] && reap_gpu "$gpu"
done

if (( failed == 0 )); then
  date -Is > "$OUT/RUN_COMPLETE"
  echo "[$(date -Is)] launcher complete jobs=$job_count failed=0" | tee -a "$LOGDIR/launcher.log"
else
  echo "[$(date -Is)] launcher complete jobs=$job_count failed=$failed" | tee -a "$LOGDIR/launcher.log"
  exit 1
fi
