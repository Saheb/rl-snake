#!/usr/bin/env bash
# Feature DQN long runs — matched game counts to ConvNet v3 for apples-to-apples comparison.
# 10×10: 10k games (matches ConvNet v3)
# 16×16: 16k games (matches ConvNet v3)
# Usage: bash scripts/run_featuredqn_longrun.sh

set -e
cd "$(dirname "$0")/.."

mkdir -p pilot_logs

run() {
  local label="$1"; shift
  echo "=== $label ==="
  uv run python scripts/"$@" 2>&1 | tee "pilot_logs/${label}.log"
  echo "--- $label done ---"
}

for seed in 0 1 2; do
  run "FeatureDQN_v2_10x10_dense_seed${seed}" \
    train_dqn.py --board_size 10 --num_games 10000 --reward_mode dense \
    --disable_icm --seed $seed \
    --run_tag "FeatureDQN_v2_10x10_dense_seed${seed}"
done

for seed in 0 1 2; do
  run "FeatureDQN_v2_16x16_dense_seed${seed}" \
    train_dqn.py --board_size 16 --num_games 16000 --reward_mode dense \
    --disable_icm --seed $seed \
    --run_tag "FeatureDQN_v2_16x16_dense_seed${seed}"
done

echo "All Feature DQN long runs complete."
