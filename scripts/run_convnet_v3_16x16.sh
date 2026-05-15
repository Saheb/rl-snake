#!/usr/bin/env bash
# ConvNet v3 on 16×16, 16k games — more games needed for larger board to converge.
# Usage: bash scripts/run_convnet_v3_16x16.sh [num_games]

set -e
cd "$(dirname "$0")/.."

GAMES=${1:-16000}

run() {
  local label="$1"; shift
  echo "=== $label ==="
  uv run python scripts/"$@" 2>&1 | tee "pilot_logs/${label}.log"
  echo "--- $label done ---"
}

mkdir -p pilot_logs

for seed in 0 1 2; do
  run "ConvDQN_v3_16x16_dense_seed${seed}" \
    train_dqn_conv.py --board_size 16 --num_games $GAMES --reward_mode dense --seed $seed \
    --run_tag "ConvDQN_v3_16x16_dense_seed${seed}"
done

echo "All 16×16 v3 experiments complete."
