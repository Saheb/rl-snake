# Curiosity Killed the Snake

[![Open in marimo](https://marimo.io/shield.svg)](https://molab.marimo.io/github/Saheb/rl-snake/blob/main/notebooks/curiosity.py/wasm)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An empirical investigation into the Intrinsic Curiosity Module (Pathak et al., 2017 · [arXiv](https://arxiv.org/abs/1705.05363) · [alphaxiv](https://alphaxiv.org/abs/1705.05363)) on the game of Snake — when does curiosity actually help, and when does it backfire?

---

## What We Found

We ran a 2 × 3 × 2 × 3 grid of experiments (algorithm × reward mode × board × seed) and got sharper answers than the standard ICM story predicts:

| Question | Answer | Evidence |
|---|---|---|
| Does ICM help DQN with dense reward? | No | \|Δ\| < 0.05 across boards |
| Does ICM rescue DQN under sparse reward? | **No** | \|Δ\| < 0.15 across `sparse` and `pure_sparse` |
| Does dense shaping matter for DQN? | Yes | ~15% score drop without it |
| Does ICM help PPO under sparse reward? | **Yes** | +24% on `pure_sparse` 10×10 (6.63 vs 5.36, n=3) |
| Does ICM increase state coverage? | **No** | PPO and PPO+ICM both reach ~98.9% of the state space |

**The mechanistic takeaway.** ICM is not a free upgrade across algorithms. DQN's replay buffer dilutes the intrinsic signal across stale transitions; PPO consumes it fresh on every rollout. And even where ICM helps (PPO `pure_sparse`), it does not help by expanding coverage — both agents saturate at ~98.9%. ICM's actual contribution is **per-step reward densification**: turning a sparse `+1`-per-food signal into a continuously non-zero novelty signal that PPO's advantage estimator can credit-assign over. It behaves less like an exploration bonus and more like a self-supervised replacement for hand-engineered reward shaping.

**Curiosity is not universally helpful — it is highly dependent on the reward structure of the environment.**

---

## The Interactive Notebook

The full investigation lives in [`notebooks/curiosity.py`](notebooks/curiosity.py) — an interactive marimo notebook with live demos, real training logs, and the complete experimental results.

**[Open in molab →](https://molab.marimo.io/github/Saheb/rl-snake/blob/main/notebooks/curiosity.py/wasm)**

The notebook covers:
- An interactive ICM simulator (build intuition before the math)
- The death-oversampling trap: how PER + ICM compounds terminal surprise into a training pathology
- The terminal-mask fix and why the original reproduction failed
- The full 2×3×2×3 experiment: DQN vs PPO, three reward modes, two boards, three seeds
- Coverage analysis: why +24% score gain is *not* an exploration effect
- Learning curves for all four headline conditions

---

## Running Locally

```bash
git clone https://github.com/Saheb/rl-snake.git
cd rl-snake
uv sync
uv run marimo edit notebooks/curiosity.py
```

### Training Scripts

| Script | What it does |
|---|---|
| `scripts/train_dqn.py` | DQN + PER + ICM — terminal mask fix at line 547 |
| `scripts/train_ppo.py` | PPO + ICM on 10×10 Snake |
| `scripts/parse_pilot_logs.py` | Parses `pilot_logs/` into JSON (data is inlined in the notebook) |
| `utils/icm.py` | ICM module (encoder, inverse model, forward model) |

---

## Project Structure

```
rl-snake/
├── notebooks/
│   └── curiosity.py        # The main interactive investigation
├── scripts/
│   ├── train_dqn.py
│   ├── train_ppo.py
│   └── parse_pilot_logs.py
├── utils/
│   └── icm.py
├── pilot_logs/             # Raw training logs from the 2×3×2×3 experiment
├── assets/                 # Charts and figures
└── logs/                   # Miscellaneous training output
```

---

*Based on "Curiosity-driven Exploration by Self-supervised Prediction", Pathak et al., ICML 2017.*
