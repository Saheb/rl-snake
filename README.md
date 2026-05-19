# RL Snake

[![Curiosity notebook](https://marimo.io/shield.svg)](https://molab.marimo.io/github/Saheb/rl-snake/blob/main/notebooks/curiosity.py/wasm)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A full journey through modern reinforcement learning on the game of Snake — from tabular Q-learning through ConvNets — with interactive marimo notebooks, real training logs, and honest empirical results.

---

## Experiments

### 1. Getting Snake to Learn (The Hard Way)

Before anything sophisticated, we established baselines by climbing the RL stack from scratch:

| Algorithm | Notes |
|---|---|
| Tabular Q-Learning | 5×5 board; converges in ~10k episodes |
| REINFORCE | Policy gradient baseline; high variance |
| A2C | Critic stabilises training on 8×8 |
| PPO | On-policy; strong on dense reward |
| PPO + Curriculum | Progressive board growth (5×5 → 10×10) |
| Behaviour Cloning → PPO | Pretrain on human demos; mixed results |
| PPO teacher → DQN student | Knowledge distillation across algorithm families |

These experiments live in `scripts/` as a preserved historical record and are visualised in the [original learning journey demo](assets/snake_learning_journey.html).

---

### 2. Curiosity Killed the Snake

**[Open interactive notebook →](https://molab.marimo.io/github/Saheb/rl-snake/blob/main/notebooks/curiosity.py/wasm)**

An empirical investigation of the Intrinsic Curiosity Module (Pathak et al., 2017) — when does curiosity actually help, and when does it backfire?

We ran a **2 × 3 × 2 × 3** grid (algorithm × reward mode × board × seed):

| Question | Answer |
|---|---|
| Does ICM help DQN with dense reward? | **No** — \|Δ\| < 0.05 |
| Does ICM rescue DQN under sparse reward? | **No** — \|Δ\| < 0.15 |
| Does dense shaping matter for DQN? | **Yes** — ~15% score drop without it |
| Does ICM help PPO under sparse reward? | **Yes** — +24% on `pure_sparse` 10×10 |
| Does ICM increase state coverage? | **No** — both saturate at ~99% |

**The mechanistic takeaway.** ICM is not a free upgrade. DQN's replay buffer dilutes the intrinsic signal across stale transitions; PPO consumes it fresh on every rollout. And even where ICM helps, the gain comes from *reward densification*, not expanded exploration.

---

### 3. Can a ConvNet Beat Hand-Crafted Features?

**[Open interactive notebook →](https://molab.marimo.io/github/Saheb/rl-snake/blob/main/notebooks/convnet.py/wasm)**

The Feature DQN uses 24 hand-crafted features (danger flags, flood-fill, normalized distances). The ConvNet sees only raw 3-channel grid pixels. Which wins?

**Short answer: ConvNet v3 wins decisively — but only after a 30-trial hyperparameter sweep.**

| Algorithm | 10×10 (10k games) | 16×16 (16k games) |
|---|---|---|
| Feature DQN | 5.50 | 5.92 |
| ConvNet v1 (wrong hyperparams) | 1.49 | 0.20 |
| **ConvNet v3 (sweep-tuned)** | **12.32** | **10.83** |

**Key findings:**
- `n_steps=1` universally better with PER — multi-step returns hurt when combined with prioritised replay
- Feature DQN plateaus at ~5.5–6 on both boards regardless of training budget
- ConvNet v3 is still climbing at 10k games on 10×10; there is no plateau in sight
- RND (Random Network Distillation) exploration bonus: **null result** — saturation on sparse boards

The v1 ConvNet's 0.20 score on 16×16 looked like architectural failure. It was actually a budget failure: 128 gradient updates/game vs 3,000 for Feature DQN, and an epsilon schedule that forced exploitation before the network had learned anything useful. Fix those, and the ConvNet wins 2× on every metric.

---

## Results at a Glance

```
Mean Score (10×10, 10k games, 3 seeds)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ConvNet v3      ████████████████████████  12.32
Feature DQN     ██████████                 5.50
ConvNet v1      ██                         1.49
Random          ▌                          0.07
```

```
Mean Score (16×16, 16k games)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ConvNet v3      ████████████████████       10.83
Feature DQN     ██████████                  5.92
ConvNet v1      ▌                           0.20
```

---

## The Notebooks

| Notebook | What it covers | Run it |
|---|---|---|
| [`curiosity.py`](notebooks/curiosity.py) | ICM on DQN + PPO, 2×3×2×3 ablation, death-oversampling trap, terminal-mask fix | [![Open](https://marimo.io/shield.svg)](https://molab.marimo.io/github/Saheb/rl-snake/blob/main/notebooks/curiosity.py/wasm) |
| [`convnet.py`](notebooks/convnet.py) | Feature DQN vs ConvNet, hyperparameter sweep, RND ablation, architecture walkthrough | [![Open](https://marimo.io/shield.svg)](https://molab.marimo.io/github/Saheb/rl-snake/blob/main/notebooks/convnet.py/wasm) |

---

## Running Locally

```bash
git clone https://github.com/Saheb/rl-snake.git
cd rl-snake
uv sync
uv run marimo edit notebooks/curiosity.py   # ICM investigation
uv run marimo edit notebooks/convnet.py     # ConvNet investigation
```

### Key Scripts

| Script | What it does |
|---|---|
| `scripts/train_dqn.py` | Feature DQN + PER + optional ICM |
| `scripts/train_dqn_conv.py` | ConvNet DQN v3 (sweep-tuned hyperparameters) |
| `scripts/train_dqn_conv_rnd.py` | ConvNet + RND exploration bonus |
| `scripts/train_ppo.py` | PPO + optional ICM |
| `scripts/run_convnet_v3_16x16.sh` | Reproduces 16×16 ConvNet v3 results |
| `scripts/run_featuredqn_longrun.sh` | Reproduces Feature DQN 10k/16k-game baselines |
| `sweeps/convdqn_sweep.yaml` | W&B sweep config (30-trial random search) |
| `utils/rnd.py` | RND target/predictor with running reward normalisation |
| `utils/icm.py` | ICM encoder, inverse model, forward model |

Earlier experiments (tabular Q, REINFORCE, A2C, PPO curriculum, behaviour cloning, knowledge distillation) are in `scripts/` as a record of the full project history.

---

## Project Structure

```
rl-snake/
├── notebooks/
│   ├── curiosity.py        # ICM investigation (interactive)
│   └── convnet.py          # ConvNet investigation (interactive)
├── scripts/                # All training scripts
├── utils/
│   ├── icm.py              # Intrinsic Curiosity Module
│   ├── rnd.py              # Random Network Distillation
│   └── per.py              # Prioritised Experience Replay
├── pilot_logs/             # Raw training logs (all experiments)
├── sweeps/                 # W&B sweep configs
├── assets/                 # Charts, replay demos
└── envs/
    └── snake_game.py       # Snake environment
```

---

*Curiosity notebook based on "Curiosity-driven Exploration by Self-supervised Prediction", Pathak et al., ICML 2017.*  
*ConvNet experiments inspired by "Human-level control through deep reinforcement learning", Mnih et al., Nature 2015.*
