#!/usr/bin/env bash
# Create a W&B sweep and immediately run COUNT agents (trials) against it.
#
# Usage:
#   bash scripts/run_sweep.sh [COUNT]
#   COUNT defaults to 30
#
# W&B will randomly sample hyperparameters, Hyperband culls bad runs at game 500.
# Each trial: 2000 games on 10×10 dense.  Expect ~3-5 min each → ~2 hours total.
#
# After the sweep, open the W&B project → Sweeps tab to find the best config,
# then run that config for 10k–15k games with 3 seeds.

set -e
cd "$(dirname "$0")/.."

COUNT=${1:-30}

echo "Creating sweep from sweeps/convdqn_sweep.yaml..."
SWEEP_ID=$(uv run wandb sweep sweeps/convdqn_sweep.yaml --project rl-snake 2>&1 \
           | grep -oE '[a-z0-9]+/rl-snake/sweeps/[a-zA-Z0-9]+' \
           | tail -1)

if [ -z "$SWEEP_ID" ]; then
  echo "ERROR: could not parse sweep ID from wandb output. Run manually:"
  echo "  uv run wandb sweep sweeps/convdqn_sweep.yaml --project rl-snake"
  exit 1
fi

echo "Sweep created: $SWEEP_ID"
echo "Running $COUNT trials..."
uv run wandb agent --count "$COUNT" "$SWEEP_ID"
