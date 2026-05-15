#!/usr/bin/env bash
# Run all GCP experiments in sequence.
# Usage: WANDB_API_KEY=<key> bash scripts/gcp_run_all.sh
# Or:    bash scripts/gcp_run_all.sh <WANDB_API_KEY>

set -e
cd "$(dirname "$0")/.."

WANDB_API_KEY="${1:-$WANDB_API_KEY}"
if [ -z "$WANDB_API_KEY" ]; then
  echo "ERROR: set WANDB_API_KEY or pass as first argument"
  exit 1
fi
export WANDB_API_KEY

echo "Device check:"
python3 -c "import torch; d='cuda' if torch.cuda.is_available() else 'cpu'; print(f'Using: {d}'); torch.cuda.is_available() and print(torch.cuda.get_device_name(0))"

# ConvNet v3 — 16×16, 16k games (already running locally — run if not done)
echo "=== ConvNet v3 16×16 (16k games) ==="
bash scripts/run_convnet_v3_16x16.sh 16000

# ConvDQN + RND — 10×10 and 16×16, 10k games
echo "=== ConvDQN + RND 10×10 (10k games) ==="
for seed in 0 1 2; do
  uv run python scripts/train_dqn_conv_rnd.py \
    --board_size 10 --num_games 10000 --reward_mode dense --seed $seed \
    --run_tag "ConvDQN_RND_10x10_dense_seed${seed}" \
    2>&1 | tee "pilot_logs/ConvDQN_RND_10x10_dense_seed${seed}.log"
done

echo "=== ConvDQN + RND 16×16 (16k games) ==="
for seed in 0 1 2; do
  uv run python scripts/train_dqn_conv_rnd.py \
    --board_size 16 --num_games 16000 --reward_mode dense --seed $seed \
    --run_tag "ConvDQN_RND_16x16_dense_seed${seed}" \
    2>&1 | tee "pilot_logs/ConvDQN_RND_16x16_dense_seed${seed}.log"
done

echo "All GCP experiments complete."
