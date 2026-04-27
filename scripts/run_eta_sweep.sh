#!/usr/bin/env bash
# Quick eta sweep on 8x8 to find the regime where the unmasked ICM failure mode is visible.
# Runs C (poisoned) and D (fixed) at multiple eta values, single seed, short horizon.
#
# Usage: bash scripts/run_eta_sweep.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-$REPO_ROOT/.venv/bin/python}"
PARALLEL="${PARALLEL:-4}"
THREADS_PER_JOB="${THREADS_PER_JOB:-2}"
NUM_GAMES="${NUM_GAMES:-2000}"
SEED="${SEED:-1}"
ETAS="${ETAS:-0.01 0.1 1.0}"
LOG_DIR="${LOG_DIR:-pilot_logs}"

export WANDB_MODE="${WANDB_MODE:-offline}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="$THREADS_PER_JOB"
export MKL_NUM_THREADS="$THREADS_PER_JOB"
export OPENBLAS_NUM_THREADS="$THREADS_PER_JOB"
export VECLIB_MAXIMUM_THREADS="$THREADS_PER_JOB"
export NUMEXPR_NUM_THREADS="$THREADS_PER_JOB"

mkdir -p "$LOG_DIR"

run_one() {
    local tag="$1"; shift
    local log="${LOG_DIR}/${tag}.log"
    if [[ -s "$log" ]] && grep -q "Training Complete" "$log"; then
        echo "[skip ] $tag"
        return 0
    fi
    echo "[start] $tag"
    if "$PYTHON" -u "$REPO_ROOT/scripts/train_dqn.py" "$@" --run_tag "$tag" > "$log" 2>&1; then
        local final
        final=$(grep "Final Mean Score" "$log" | tail -1 || true)
        echo "[ok   ] $tag — ${final:-<no score>}"
    else
        echo "[FAIL ] $tag (rc=$?) — see $log"
    fi
}

throttle() {
    while [[ "$(jobs -rp 2>/dev/null | wc -l | tr -d ' ')" -ge "$PARALLEL" ]]; do
        sleep 1
    done
}

COMMON="--board_size 8 --num_games $NUM_GAMES --seed $SEED --disable_priority_cap --disable_foundation_memory"

JOBS=()
for eta in $ETAS; do
    eta_tag=$(printf "%s" "$eta" | tr '.' 'p')
    JOBS+=("eta_C_unmasked_eta${eta_tag}_g${NUM_GAMES}|||$COMMON --icm_eta $eta --disable_terminal_mask")
    JOBS+=("eta_D_masked_eta${eta_tag}_g${NUM_GAMES}|||$COMMON --icm_eta $eta")
done

echo "Queued ${#JOBS[@]} sweep jobs (parallel=$PARALLEL)."
for entry in "${JOBS[@]}"; do
    tag="${entry%%|||*}"
    args="${entry#*|||}"
    throttle
    # shellcheck disable=SC2086
    ( run_one "$tag" $args ) &
done
wait

echo
echo "=== Eta sweep summary (8x8, ${NUM_GAMES} games, seed ${SEED}) ==="
printf "%-40s %s\n" "tag" "final_mean | last500_mean"
for f in "$LOG_DIR"/eta_*_g${NUM_GAMES}.log; do
    [[ -e "$f" ]] || continue
    tag=$(basename "$f" .log)
    final=$(grep "Final Mean Score" "$f" | tail -1 | sed 's/Final Mean Score: //' || true)
    # Compute last-500 window mean from cumulative-mean checkpoints
    last500=$("$PYTHON" -c "
import re, sys
pat = re.compile(r'Game (\d+) .* Mean Score: ([\d.]+)')
pts = [(int(m.group(1)), float(m.group(2))) for line in open('$f') for m in [pat.search(line)] if m]
if not pts:
    print('-'); sys.exit()
prev_g, prev_ms = 0, 0.0
wins = []
for g, ms in pts:
    if g - prev_g > 0:
        wins.append((g, (ms*g - prev_ms*prev_g) / (g-prev_g)))
    prev_g, prev_ms = g, ms
last = [w for g,w in wins if g > pts[-1][0] - 500]
print(f'{sum(last)/len(last):.2f}' if last else '-')
")
    printf "%-40s %s | %s\n" "$tag" "${final:-<incomplete>}" "$last500"
done
