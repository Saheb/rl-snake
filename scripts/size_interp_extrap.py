"""
Interpolation vs extrapolation in board size, with clean controls.

Reuses the already-trained 6x6- and 10x10-trained size-agnostic agents (no retraining).
Controls that isolate board size / food-distance as the only variable:
  - fixed length cap (snake <= CAP everywhere) -> self-collision negligible; navigation only
  - fixed step budget (BUDGET steps, all boards) -> "score" = food per budget; no timeout variable
Metrics per board: toward-food% (navigation), coverage=distinct cells (exploration), food count.

Hypothesis (from the confound mechanism): the TRAINING board is the pivot. Test boards
<= train size are interpolation (food distances within training range) -> works. Test boards
> train size are extrapolation -> cliff at ~train's max food distance. So the 10x10 agent's
cliff should sit to the RIGHT of the 6x6 agent's.
"""
import os, sys, numpy as np, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, "/Users/saheb/home/rl-snake")
from scripts.size_transfer import SizeAgnosticQNet, get_conv_state, DEVICE
from envs.snake_game import SnakeGame

OUT = "/Users/saheb/home/rl-snake/size_transfer"
CAP, BUDGET, N = 5, 400, 40
BOARDS = [6, 7, 8, 9, 10, 11, 12, 14, 16]


def mv(head, a):
    x, y = head
    return [(x - 1, y), (x, y + 1), (x + 1, y), (x, y - 1)][a]


@torch.no_grad()
def evaluate(model, board):
    tf, cov, food = [], [], []
    for _ in range(N):
        g = SnakeGame(board_size=board, reward_mode="dense"); g.reset()
        visited = set(); tw = tot = 0; info = {"score": 0}
        for _ in range(BUDGET):
            head = g.snake_position[-1]; fx = g.food_position
            visited.add(head)
            d0 = abs(head[0] - fx[0]) + abs(head[1] - fx[1]) if fx else 0
            a = int(torch.argmax(model(torch.tensor(get_conv_state(g)[None], device=DEVICE))).item())
            nh = mv(head, a)
            _, _, done, info = g.step(a)
            if fx:
                tot += 1; tw += (abs(nh[0] - fx[0]) + abs(nh[1] - fx[1]) < d0)
            # length cap: keep snake short so self-collision can't confound board size
            while len(g.snake_position) > CAP:
                g.snake_position.popleft()
            g._clear_board(); g._update_board_snake(); g._update_board_food()
            if done:
                break
        tf.append(tw / max(tot, 1)); cov.append(len(visited)); food.append(info["score"])
    return 100 * np.mean(tf), np.mean(cov), np.mean(food)


agents = {
    "6x6-trained":  (f"{OUT}/sizeagnostic_6x6_best.pth", 6, "#dc2626"),
    "10x10-trained": (f"{OUT}/sizeagnostic_10x10_best.pth", 10, "#2563eb"),
}
results = {}
for name, (ckpt, tsize, _) in agents.items():
    m = SizeAgnosticQNet().to(DEVICE); m.load_state_dict(torch.load(ckpt, map_location=DEVICE)); m.eval()
    print(f"\n=== {name} (pivot {tsize}) ===")
    print(f'{"board":>6} {"toward-food%":>12} {"coverage":>9} {"food/budget":>12} {"regime":>8}')
    rows = {"tf": [], "cov": [], "food": []}
    for b in BOARDS:
        tf, cov, food = evaluate(m, b)
        rows["tf"].append(tf); rows["cov"].append(cov); rows["food"].append(food)
        regime = "interp" if b <= tsize else "EXTRAP"
        print(f'{b:6d} {tf:12.0f} {cov:9.1f} {food:12.1f} {regime:>8}', flush=True)
    results[name] = rows

# figure: toward-food and coverage vs board, both agents, pivots marked
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
for name, (_, tsize, color) in agents.items():
    ax1.plot(BOARDS, results[name]["tf"], "o-", color=color, label=name)
    ax1.axvline(tsize, ls="--", color=color, alpha=0.4)
    ax2.plot(BOARDS, results[name]["cov"], "s-", color=color, label=name)
    ax2.axvline(tsize, ls="--", color=color, alpha=0.4)
ax1.axhline(50, ls=":", color="gray", alpha=0.6, label="chance (50%)")
ax1.set_xlabel("test board size"); ax1.set_ylabel("moves toward food (%)")
ax1.set_title("Navigation: interpolation (≤ pivot) vs extrapolation (> pivot)"); ax1.legend()
ax2.set_xlabel("test board size"); ax2.set_ylabel("distinct cells visited (coverage)")
ax2.set_title("Exploration: does it roam or get stuck when it fails?"); ax2.legend()
fig.tight_layout(); fig.savefig(f"{OUT}/interp_extrap.png", dpi=130)
print(f"\nWrote {OUT}/interp_extrap.png")
