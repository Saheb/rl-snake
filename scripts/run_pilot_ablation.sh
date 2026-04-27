#!/usr/bin/env bash
# 4-way DQN ablation pilot on 8x8 — 4000 games x 3 seeds per condition.
#
# Conditions:
#   A: DQN              (no PER, no ICM)
#   B: DQN + PER
#   C: DQN + PER + ICM  (terminal mask DISABLED — poisoned)
#   D: DQN + PER + ICM  (terminal mask ENABLED — fixed)
#
# Usage:
#   bash scripts/run_pilot_ablation.sh                # sequential
#   PARALLEL=4 bash scripts/run_pilot_ablation.sh     # 4 jobs at a time
#
# Env overrides:
#   BOARD_SIZE=8       NUM_GAMES=4000     SEEDS="1 2 3"
#   PARALLEL=1         (concurrent workers; 1 = sequential)
#   THREADS_PER_JOB=2  (BLAS/OMP threads per worker; multiply by PARALLEL <= cores)
#   WANDB_MODE=offline (recommended for pilots so the W&B dashboard stays clean)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

BOARD_SIZE="${BOARD_SIZE:-8}"
NUM_GAMES="${NUM_GAMES:-4000}"
SEEDS="${SEEDS:-1 2 3}"
LOG_DIR="${LOG_DIR:-pilot_logs}"
if [[ -z "${PYTHON:-}" ]]; then
    if [[ -x "$REPO_ROOT/.venv/bin/python" ]]; then
        PYTHON="$REPO_ROOT/.venv/bin/python"
    else
        PYTHON="python3"
    fi
fi

PARALLEL="${PARALLEL:-1}"
THREADS_PER_JOB="${THREADS_PER_JOB:-2}"

export WANDB_MODE="${WANDB_MODE:-offline}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="$THREADS_PER_JOB"
export MKL_NUM_THREADS="$THREADS_PER_JOB"
export OPENBLAS_NUM_THREADS="$THREADS_PER_JOB"
export VECLIB_MAXIMUM_THREADS="$THREADS_PER_JOB"
export NUMEXPR_NUM_THREADS="$THREADS_PER_JOB"

mkdir -p "$LOG_DIR"

# Synchronous single-job runner. Args after the tag are forwarded verbatim to train_dqn.py.
run_one() {
    local tag="$1"; shift
    local log="${LOG_DIR}/${tag}.log"

    if [[ -s "$log" ]] && grep -q "Training Complete" "$log"; then
        echo "[skip ] $tag (already complete)"
        return 0
    fi

    echo "[start] $tag → $log"
    if "$PYTHON" -u "$REPO_ROOT/scripts/train_dqn.py" "$@" --run_tag "$tag" > "$log" 2>&1; then
        local final
        final=$(grep "Final Mean Score" "$log" | tail -1 || true)
        echo "[ok   ] $tag — ${final:-<no score>}"
    else
        echo "[FAIL ] $tag (rc=$?) — see $log"
        return 1
    fi
}

# Concurrency pool: launches background jobs but blocks once $PARALLEL are running.
# Works in mac bash 3.2 (no `wait -n` required).
running_count() {
    jobs -rp 2>/dev/null | wc -l | tr -d ' '
}

throttle() {
    while [[ "$(running_count)" -ge "$PARALLEL" ]]; do
        sleep 1
    done
}

# Build the condition list. Each entry is "tag|||args_with_spaces".
# Using ||| as a sentinel that won't appear in tags or paths.
JOBS=()
COMMON_ARGS="--board_size $BOARD_SIZE --num_games $NUM_GAMES --disable_priority_cap --disable_foundation_memory"
for seed in $SEEDS; do
    JOBS+=("pilot_A_dqn_${BOARD_SIZE}x${BOARD_SIZE}_g${NUM_GAMES}_seed${seed}|||--seed $seed $COMMON_ARGS --disable_icm --disable_per")
    JOBS+=("pilot_B_dqn_per_${BOARD_SIZE}x${BOARD_SIZE}_g${NUM_GAMES}_seed${seed}|||--seed $seed $COMMON_ARGS --disable_icm")
    JOBS+=("pilot_C_dqn_per_icm_unmasked_${BOARD_SIZE}x${BOARD_SIZE}_g${NUM_GAMES}_seed${seed}|||--seed $seed $COMMON_ARGS --disable_terminal_mask")
    JOBS+=("pilot_D_dqn_per_icm_masked_${BOARD_SIZE}x${BOARD_SIZE}_g${NUM_GAMES}_seed${seed}|||--seed $seed $COMMON_ARGS")
done

echo "Queued ${#JOBS[@]} jobs (parallel=${PARALLEL}, threads/job=${THREADS_PER_JOB})."
echo "Logs → $LOG_DIR/"
echo

for entry in "${JOBS[@]}"; do
    tag="${entry%%|||*}"
    args_str="${entry#*|||}"
    throttle
    # shellcheck disable=SC2086
    ( run_one "$tag" $args_str ) &
done

wait

echo
echo "Pilot complete. Final mean scores:"
for f in "$LOG_DIR"/pilot_*.log; do
    [[ -e "$f" ]] || continue
    final=$(grep "Final Mean Score" "$f" | tail -1 || true)
    base=$(basename "$f" .log)
    echo "  ${base}: ${final:-<incomplete>}"
done
