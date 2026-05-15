#!/usr/bin/env bash
# Runs ConvNet DQN v3 experiments (sweep-tuned hyperparameters, igzvteil winner).
# 6 total runs: 3 seeds × 2 board sizes (10x10, 16x16), 10k games each.
# Usage: bash scripts/run_convnet_v2.sh [num_games]

set -e
cd "$(dirname "$0")/.."

GAMES=${1:-10000}

run() {
  local label="$1"; shift
  echo "=== $label ==="
  uv run python scripts/"$@" 2>&1 | tee "pilot_logs/${label}.log"
  echo "--- $label done ---"
}

mkdir -p pilot_logs

for seed in 0 1 2; do
  run "ConvDQN_v3_10x10_dense_seed${seed}" \
    train_dqn_conv.py --board_size 10 --num_games $GAMES --reward_mode dense --seed $seed \
    --run_tag "ConvDQN_v3_10x10_dense_seed${seed}"
done

for seed in 0 1 2; do
  run "ConvDQN_v3_16x16_dense_seed${seed}" \
    train_dqn_conv.py --board_size 16 --num_games $GAMES --reward_mode dense --seed $seed \
    --run_tag "ConvDQN_v3_16x16_dense_seed${seed}"
done

echo "All v3 experiments complete."
