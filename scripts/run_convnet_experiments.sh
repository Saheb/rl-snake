#!/usr/bin/env bash
# Runs all ConvNet DQN experiments for the convnet.py notebook.
# 9 total runs (3 board configs × 3 seeds), each logged to wandb and stdout.
# Usage: bash scripts/run_convnet_experiments.sh

set -e
cd "$(dirname "$0")/.."

GAMES=5000

run() {
  local label="$1"; shift
  echo "=== $label ==="
  uv run python scripts/"$@" 2>&1 | tee -a "pilot_logs/${label}.log"
  echo "--- $label done ---"
}

mkdir -p pilot_logs

# ConvNet DQN on 10×10 (3 seeds)
for seed in 0 1 2; do
  run "ConvDQN_10x10_dense_seed${seed}" \
    train_dqn_conv.py --board_size 10 --num_games $GAMES --reward_mode dense --seed $seed \
    --run_tag "ConvDQN_10x10_dense_seed${seed}"
done

# Feature DQN on 16×16 (3 seeds) — uses existing train_dqn.py
for seed in 0 1 2; do
  run "FeatureDQN_16x16_dense_seed${seed}" \
    train_dqn.py --board_size 16 --num_games $GAMES --reward_mode dense \
    --disable_icm --seed $seed \
    --run_tag "FeatureDQN_16x16_dense_seed${seed}"
done

# ConvNet DQN on 16×16 (3 seeds)
for seed in 0 1 2; do
  run "ConvDQN_16x16_dense_seed${seed}" \
    train_dqn_conv.py --board_size 16 --num_games $GAMES --reward_mode dense --seed $seed \
    --run_tag "ConvDQN_16x16_dense_seed${seed}"
done

echo "All experiments complete."
