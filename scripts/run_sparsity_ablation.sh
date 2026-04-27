#!/usr/bin/env bash
# 2x2 factorial: {dense, sparse} reward × {DQN, DQN+ICM}
# Tests whether ICM rescues learning under sparse rewards.
#
# Conditions (all without PER for clean isolation):
#   S1: dense  + DQN          (baseline; should learn easily)
#   S2: dense  + DQN+ICM      (does ICM help when shaping already provides dense signal?)
#   S3: sparse + DQN          (degraded baseline; expected to struggle)
#   S4: sparse + DQN+ICM      (does curiosity rescue sparse rewards?)
#
# Usage:
#   PARALLEL=4 bash scripts/run_sparsity_ablation.sh
#
# Env overrides:
#   BOARD_SIZE=8       NUM_GAMES=5000     SEEDS="1 2 3"
#   ICM_ETA=0.1        PARALLEL=1         THREADS_PER_JOB=2

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

BOARD_SIZE="${BOARD_SIZE:-8}"
NUM_GAMES="${NUM_GAMES:-5000}"
SEEDS="${SEEDS:-1 2 3}"
ICM_ETA="${ICM_ETA:-0.1}"
SPARSE_MODE="${SPARSE_MODE:-sparse}"  # 'sparse' or 'pure_sparse'
LOG_DIR="${LOG_DIR:-pilot_logs}"
PARALLEL="${PARALLEL:-1}"
THREADS_PER_JOB="${THREADS_PER_JOB:-2}"

if [[ "$SPARSE_MODE" != "sparse" && "$SPARSE_MODE" != "pure_sparse" ]]; then
    echo "Error: SPARSE_MODE must be 'sparse' or 'pure_sparse' (got '$SPARSE_MODE')" >&2
    exit 1
fi

# Tag prefix differs by mode so logs don't collide.
if [[ "$SPARSE_MODE" == "pure_sparse" ]]; then
    PREFIX="puresparse"
else
    PREFIX="sparse"
fi

if [[ -z "${PYTHON:-}" ]]; then
    PYTHON="${REPO_ROOT}/.venv/bin/python"
    [[ -x "$PYTHON" ]] || PYTHON="python3"
fi

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
    fi
}

throttle() {
    while [[ "$(jobs -rp 2>/dev/null | wc -l | tr -d ' ')" -ge "$PARALLEL" ]]; do
        sleep 1
    done
}

# Common args: PER disabled (clean isolation); priority_cap and foundation_memory inert without PER.
COMMON="--board_size $BOARD_SIZE --num_games $NUM_GAMES --disable_per"

SKIP_DENSE="${SKIP_DENSE:-0}"  # If 1, skip S1/S2 (dense conditions); useful when re-using a previous pilot's dense baseline.

JOBS=()
for seed in $SEEDS; do
    suffix="${BOARD_SIZE}x${BOARD_SIZE}_g${NUM_GAMES}_seed${seed}"
    if [[ "$SKIP_DENSE" != "1" ]]; then
        JOBS+=("${PREFIX}_S1_dense_dqn_${suffix}|||--seed $seed $COMMON --reward_mode dense        --disable_icm")
        JOBS+=("${PREFIX}_S2_dense_icm_${suffix}|||--seed $seed $COMMON --reward_mode dense        --icm_eta $ICM_ETA")
    fi
    JOBS+=("${PREFIX}_S3_${SPARSE_MODE}_dqn_${suffix}|||--seed $seed $COMMON --reward_mode $SPARSE_MODE --disable_icm")
    JOBS+=("${PREFIX}_S4_${SPARSE_MODE}_icm_${suffix}|||--seed $seed $COMMON --reward_mode $SPARSE_MODE --icm_eta $ICM_ETA")
done

echo "Queued ${#JOBS[@]} jobs (parallel=${PARALLEL}, threads/job=${THREADS_PER_JOB}, eta=${ICM_ETA}, sparse_mode=${SPARSE_MODE})."
echo "Tag prefix: ${PREFIX}_"
echo "Logs → $LOG_DIR/"
echo

for entry in "${JOBS[@]}"; do
    tag="${entry%%|||*}"
    args="${entry#*|||}"
    throttle
    # shellcheck disable=SC2086
    ( run_one "$tag" $args ) &
done

wait

echo
echo "=== Final cumulative-mean scores ==="
for f in "$LOG_DIR"/${PREFIX}_*.log; do
    [[ -e "$f" ]] || continue
    final=$(grep "Final Mean Score" "$f" | tail -1 || true)
    base=$(basename "$f" .log)
    echo "  ${base}: ${final:-<incomplete>}"
done
