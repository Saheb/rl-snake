"""
Prediction check: does the environment get EASIER on larger boards (fewer constraints)?

Hypothesis (why performance rises with board size): larger boards have fewer wall interactions,
fewer forced turns, more room to execute the greedy policy — so once open-space behavior is learned,
bigger boards are easier. Testable: collision rate should DECREASE with board size.

Measures the curriculum ego agent across boards:
  - collisions per 1000 steps (the key metric)
  - death cause: % self-collision / % wall / % survived-to-budget
  - % steps with head on a wall edge (wall-interaction rate)
  - food per 1000 steps (eating rate) + avg/MAX score + mean survival steps
"""
import sys, random, numpy as np, torch
sys.path.insert(0, "/Users/saheb/home/rl-snake")
from scripts.size_curriculum import CoordAttnPoolQNet, state_ego, CKPT
from scripts.size_transfer import DEVICE
from envs.snake_game import SnakeGame

DELTAS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
BOARDS = [6, 10, 22, 48, 100]


@torch.no_grad()
def analyze(model, B, n, cap):
    scores = []; total_steps = 0; wall_edge = 0
    deaths = {"self": 0, "wall": 0}; survived = 0
    for e in range(n):
        random.seed(7000 + e); np.random.seed(7000 + e)
        game = SnakeGame(board_size=B, reward_mode="dense"); game.reset(); info = {"score": 0}
        died = None
        for _ in range(cap):
            head = game.snake_position[-1]
            if min(head[0], B-1-head[0], head[1], B-1-head[1]) == 0:
                wall_edge += 1
            a = int(torch.argmax(model(torch.tensor(state_ego(game)[None], device=DEVICE))).item())
            nh = (head[0]+DELTAS[a][0], head[1]+DELTAS[a][1])
            _, _, done, info = game.step(a); total_steps += 1
            if done:
                if nh[0] < 0 or nh[0] >= B or nh[1] < 0 or nh[1] >= B:
                    died = "wall"
                else:
                    died = "self"
                break
        if died:
            deaths[died] += 1
        else:
            survived += 1
        scores.append(info["score"])
    dtot = deaths["self"] + deaths["wall"]
    return {
        "coll_per_1k": 1000 * dtot / max(total_steps, 1),
        "pct_self": 100 * deaths["self"] / max(dtot, 1) if dtot else 0.0,
        "pct_wall": 100 * deaths["wall"] / max(dtot, 1) if dtot else 0.0,
        "pct_survived": 100 * survived / n,
        "pct_wall_edge": 100 * wall_edge / max(total_steps, 1),
        "food_per_1k": 1000 * sum(scores) / max(total_steps, 1),
        "avg_score": float(np.mean(scores)), "max_score": int(np.max(scores)),
        "mean_survival": total_steps / n,
    }


def main():
    model = CoordAttnPoolQNet().to(DEVICE)
    model.load_state_dict(torch.load(CKPT, map_location=DEVICE)); model.eval()
    rows = {}
    for B in BOARDS:
        n, cap = (14, B*B*2) if B <= 22 else (5, min(B*B, 8000))
        rows[B] = analyze(model, B, n, cap)
        r = rows[B]
        print(f"board {B:3d} done", flush=True)
    keys = [
        ("coll_per_1k", "collisions / 1000 steps", "{:.2f}"),
        ("pct_self", "  of deaths: % self", "{:.0f}%"),
        ("pct_wall", "  of deaths: % wall", "{:.0f}%"),
        ("pct_survived", "% episodes survived budget", "{:.0f}%"),
        ("pct_wall_edge", "% steps on a wall edge", "{:.0f}%"),
        ("food_per_1k", "food / 1000 steps", "{:.1f}"),
        ("avg_score", "avg score", "{:.1f}"),
        ("max_score", "MAX score", "{:d}"),
        ("mean_survival", "mean steps/episode", "{:.0f}"),
    ]
    print(f"\n{'statistic':>28} | " + "  ".join(f"{b:>7d}" for b in BOARDS))
    print("-" * (31 + 9 * len(BOARDS)))
    for k, label, fmt in keys:
        print(f"{label:>28} | " + "  ".join(f"{fmt.format(rows[b][k]):>7}" for b in BOARDS))


if __name__ == "__main__":
    main()
