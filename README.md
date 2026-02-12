# 🐍 Snake RL: Zero to Hero

Welcome to the Snake Reinforcement Learning project!
This repository contains code to train an AI agent to play Snake, scaling from a small 5x5 board to mastering a 10x10 grid using Curriculum Learning and PPO.

## 🚀 Start Here

1.  **[Interactive Tutorial (Notebook)](Snake_RL_Tutorial.ipynb)**:
    - Open `Snake_RL_Tutorial.ipynb` in Jupyter/VS Code.
    - Learn the theory (REINFORCE vs DQN vs PPO).
    - Run the code snippets to see how the agents work.

2.  **[The Journey (Blog Post)](blog.md)**:
    - Read `blog.md` for the narrative story of how we solved the sparse reward problem on large boards.

3.  **[Interactive Visualization](snake_learning_journey.html)**:
    - Open `snake_learning_journey.html` in your browser.
    - Watch the agent evolve from random movements to skilled gameplay!

## 📂 Key Files

- **`train_tabular_q.py`**: The "Zero" point. Classic non-deep Q-Learning and Double Q-Learning on small boards.
- **`train_ppo_curriculum.py`**: The core logic. Implements Proximal Policy Optimization (PPO) with Curriculum Learning (5x5 -> 8x8) and Imitation Learning.
- **`train_curriculum_10x10.py`**: The final boss level. Trains the agent on the 10x10 board.
- **`snake_game.py`**: The custom Gym-like environment.
- **`visualize_journey.py`**: The script that recorded the games and generated the HTML visualization.

## 🛠️ Installation

This project uses modern Python tooling.

```bash
# If using uv (recommended)
uv sync

# Or standard pip
pip install -e .
```

## 🏆 Results

- **5x5 Board**: Perfect Score (24/24)
- **8x8 Board**: High Proficiency
- **10x10 Board**: Record Score **64** (Mean ~18)

Happy Learning! 🐍
