#!/usr/bin/env bash
# Run this once after SSH-ing into the GCP VM.
# Usage: bash scripts/gcp_setup.sh <WANDB_API_KEY>
#
# Installs dependencies, configures wandb, and verifies GPU + wandb connectivity.

set -e

WANDB_API_KEY="${1:-$WANDB_API_KEY}"
if [ -z "$WANDB_API_KEY" ]; then
  echo "ERROR: pass your wandb API key as argument or set WANDB_API_KEY env var"
  echo "  bash scripts/gcp_setup.sh <your_key>"
  exit 1
fi

echo "=== 1. GPU check ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo "=== 2. Python / CUDA check ==="
python3 -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"

echo "=== 3. Install uv + deps ==="
pip install uv -q
uv sync --quiet

echo "=== 4. Configure wandb ==="
export WANDB_API_KEY="$WANDB_API_KEY"
uv run wandb login "$WANDB_API_KEY" --relogin

echo "=== 5. Verify wandb can reach project ==="
uv run python -c "
import wandb
run = wandb.init(project='rl-snake', name='gcp-connectivity-test', mode='online')
wandb.log({'test': 1})
run.finish()
print('wandb OK — check https://wandb.ai/saheb37/rl-snake for the test run')
"

echo ""
echo "=== Setup complete. Ready to train. ==="
echo ""
echo "Run experiments:"
echo "  # ConvNet v3 — 16x16, 16k games (3 seeds)"
echo "  WANDB_API_KEY=$WANDB_API_KEY bash scripts/run_convnet_v3_16x16.sh 16000"
echo ""
echo "  # ConvDQN + RND — 10x10 + 16x16, 10k games"
echo "  WANDB_API_KEY=$WANDB_API_KEY bash scripts/run_convnet_rnd.sh 10000"
