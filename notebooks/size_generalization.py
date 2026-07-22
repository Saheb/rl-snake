# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "torch>=2.0.0",
#     "numpy",
#     "matplotlib",
# ]
# ///
"""Try the cross-size Snake agent: trained only on boards 6-22, plays up to 100x100 zero-shot."""
import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(
        """
        # 🐍 A Snake agent that scales

        This agent was trained **only on boards 6×6 through 22×22**, yet it plays boards it has
        *never seen* — up to 100×100 — zero-shot. Pick a board size and watch it go.

        Two ingredients make this work (see [`size_transfer/`](../size_transfer/README.md)):
        a **size-invariant egocentric representation** (so a big board looks like training near the
        head) and **training coverage** across a range of sizes. The model is tiny: ~209K parameters,
        824 KB.
        """
    )
    return


@app.cell
def _():
    import os
    import sys
    import numpy as np
    import torch
    import torch.nn as nn
    import matplotlib.pyplot as plt
    import marimo as mo

    # Locate the repo root (dir containing size_transfer/) so we can import the env + load the model.
    root = os.getcwd()
    while root != "/" and not os.path.isdir(os.path.join(root, "size_transfer")):
        root = os.path.dirname(root)
    sys.path.insert(0, root)
    from envs.snake_game import SnakeGame

    model_path = os.path.join(root, "size_transfer", "curriculum_ego_best.pth")
    is_script_mode = mo.app_meta().mode == "script"
    return SnakeGame, is_script_mode, mo, model_path, np, nn, plt, torch


@app.cell
def _(nn, torch):
    class CrossSizeAgent(nn.Module):
        """Egocentric input + attention pool + dueling head. Runs on any board size."""

        def __init__(self, n_actions=4, n_heads=4):
            super().__init__()
            self.n_heads = n_heads
            self.conv = nn.Sequential(
                nn.Conv2d(5, 64, 3, padding=1), nn.ReLU(),
                nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            )
            self.attn = nn.Conv2d(128, n_heads, 1)
            self.proj = nn.Sequential(nn.Linear(128 * n_heads, 128), nn.ReLU())
            self.value_stream = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 1))
            self.advantage_stream = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, n_actions))

        def _pool(self, x):
            h = self.conv(x)
            b, c, hh, ww = h.shape
            w = torch.softmax(self.attn(h).view(b, self.n_heads, hh * ww), dim=2)
            pooled = torch.einsum("bnk,bck->bnc", w, h.view(b, c, hh * ww))
            av = self.proj(pooled.reshape(b, self.n_heads * c))
            return torch.cat([av, h.amax(dim=(2, 3))], dim=1)

        def forward(self, x):
            z = self._pool(x)
            v = self.value_stream(z); a = self.advantage_stream(z)
            return v + (a - a.mean(dim=1, keepdim=True))

    return (CrossSizeAgent,)


@app.cell
def _(np):
    # Egocentric state encoding — must match training (scripts/size_ego_attn.py).
    kappa = 3.0

    def state_ego(game):
        b = game.board_size
        s = np.zeros((3, b, b), np.float32)
        snake = list(game.snake_position)
        for p in snake[:-1]:
            s[0, p[0], p[1]] = 1.0
        head = snake[-1]
        s[1, head[0], head[1]] = 1.0
        if game.food_position:
            s[2, game.food_position[0], game.food_position[1]] = 1.0
        rel_r = np.tanh((np.arange(b) - head[0]) / kappa)[:, None] * np.ones((1, b), np.float32)
        rel_c = np.tanh((np.arange(b) - head[1]) / kappa)[None, :] * np.ones((b, 1), np.float32)
        return np.concatenate([s, rel_r[None].astype(np.float32), rel_c[None].astype(np.float32)], axis=0)

    return (state_ego,)


@app.cell
def _(CrossSizeAgent, model_path, torch):
    model = CrossSizeAgent()
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return (model,)


@app.cell
def _(mo):
    board_slider = mo.ui.slider(6, 100, value=25, step=1, label="Board size", show_value=True)
    board_slider
    return (board_slider,)


@app.cell
def _(SnakeGame, board_slider, model, np, state_ego, torch):
    def play_episode(board, max_steps):
        game = SnakeGame(board_size=board, reward_mode="dense")
        game.reset()

        def grid():
            g = np.zeros((board, board), np.int8)
            for p in list(game.snake_position)[:-1]:
                g[p[0], p[1]] = 1
            h = game.snake_position[-1]
            g[h[0], h[1]] = 2
            if game.food_position:
                g[game.food_position[0], game.food_position[1]] = 3
            return g

        frames = []
        info = {"score": 0}
        hits = tot = 0
        with torch.no_grad():
            for _ in range(max_steps):
                head = game.snake_position[-1]
                food = game.food_position
                frames.append(grid())
                if food is None:
                    break
                d0 = abs(head[0] - food[0]) + abs(head[1] - food[1])
                a = int(torch.argmax(model(torch.tensor(state_ego(game)[None]))).item())
                prev = info.get("score", 0)
                _, _, done, info = game.step(a)
                if info.get("score", prev) > prev:
                    hits += 1; tot += 1
                elif not done:
                    nh = game.snake_position[-1]
                    tot += 1
                    if food and abs(nh[0] - food[0]) + abs(nh[1] - food[1]) < d0:
                        hits += 1
                else:
                    tot += 1
                if done:
                    break
        frames.append(grid())
        return frames, info.get("score", 0), 100 * hits / max(tot, 1)

    size = board_slider.value
    frames, score, toward = play_episode(size, max_steps=min(size * size * 2, 1200))
    return frames, score, toward


@app.cell
def _(board_slider, frames, mo, score, toward):
    mo.md(
        f"""
        **Board {board_slider.value}×{board_slider.value}** · **{score} food eaten** ·
        **{toward:.0f}% of moves toward food** (50% = random) · {len(frames)} steps.
        Scrub the slider below to replay the episode.
        """
    )
    return


@app.cell
def _(frames, mo):
    frame_slider = mo.ui.slider(0, len(frames) - 1, value=len(frames) - 1, step=1, label="Step", show_value=True)
    frame_slider
    return (frame_slider,)


@app.cell
def _(frame_slider, frames, np, plt):
    g = frames[frame_slider.value]
    b = g.shape[0]
    img = np.ones((b, b, 3), np.float32)
    img[g == 1] = [0.20, 0.70, 0.35]   # body
    img[g == 2] = [0.05, 0.32, 0.15]   # head
    img[g == 3] = [0.90, 0.22, 0.22]   # food
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(img, interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"{b}×{b}  ·  step {frame_slider.value}/{len(frames) - 1}")
    fig
    return


if __name__ == "__main__":
    app.run()
