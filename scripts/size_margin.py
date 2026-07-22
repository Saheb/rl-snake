"""
Experiment B' — advantage-margin dilution: WHY the policy collapses while direction stays readable.

Experiment A showed a FROZEN linear reader still recovers food-direction at 0.85 AUC on 16x16,
yet the policy is at chance. AUC is rank-based, so it survives if the direction *ordering* holds.
What it can't see is MARGIN. Hypothesis: GAP dilution shrinks the *magnitude* of the food-direction
term in the advantage vector, while head-LOCAL terms (danger / self-avoidance, carried by the
area-invariant max-pool and a single head cell) keep their magnitude. On a big board the preserved
local terms then OUTVOTE the diluted food term in argmax -> the snake avoids death but wanders at
chance w.r.t. food.

For each board size, over the agent's own states, decompose the advantage vector A(4) (== Q up to
constants that cancel in differences):
  food_margin   = max_{a toward food} Q[a]  -  max_{a away from food} Q[a]
  danger_margin = max_{a safe}        Q[a]  -  max_{a fatal}          Q[a]
  spread        = max Q - min Q
  toward-argmax = fraction of states whose argmax action moves toward food

Prediction: food_margin -> 0 with board size (relative to spread), danger_margin holds, toward-argmax
-> chance. That is the mechanism: food term diluted, local terms preserved, local wins the argmax.
"""
import os, sys, random, numpy as np, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, "/Users/saheb/home/rl-snake")
from scripts.size_transfer import SizeAgnosticQNet, get_conv_state, DEVICE
from envs.snake_game import SnakeGame

OUT = "/Users/saheb/home/rl-snake/size_transfer"
CKPT = f"{OUT}/sizeagnostic_6x6_best.pth"
SIZES = [6, 8, 10, 12, 16]
DELTAS = [(-1, 0), (0, 1), (1, 0), (0, -1)]   # 0 up, 1 right, 2 down, 3 left


@torch.no_grad()
def margins(model, board, n_states=4000, eps=0.3, seed=0):
    random.seed(seed); np.random.seed(seed)
    food_m, danger_m, spread, toward_arg = [], [], [], []
    collected = 0
    while collected < n_states:
        game = SnakeGame(board_size=board, reward_mode="dense"); game.reset()
        for _ in range(board * board * 4):
            head = game.snake_position[-1]; food = game.food_position
            if food is None:
                break
            d0 = abs(head[0] - food[0]) + abs(head[1] - food[1])
            q = model(torch.tensor(get_conv_state(game)[None], device=DEVICE)).cpu().numpy().ravel()

            toward = np.zeros(4, bool); fatal = np.zeros(4, bool)
            for a, (dx, dy) in enumerate(DELTAS):
                nh = (head[0] + dx, head[1] + dy)
                d1 = abs(nh[0] - food[0]) + abs(nh[1] - food[1])
                toward[a] = d1 < d0
                fatal[a] = game._check_collision(nh, nh == food)

            if toward.any() and (~toward).any():
                food_m.append(q[toward].max() - q[~toward].max())
                toward_arg.append(bool(toward[int(q.argmax())]))
            if fatal.any() and (~fatal).any():
                danger_m.append(q[~fatal].max() - q[fatal].max())
            spread.append(q.max() - q.min())
            collected += 1

            # eps-greedy to keep exploring the board's state space
            a = random.randint(0, 3) if random.random() < eps else int(q.argmax())
            _, _, done, _ = game.step(a)
            if done or collected >= n_states:
                break
    return (float(np.mean(food_m)), float(np.mean(danger_m)), float(np.mean(spread)),
            float(np.mean(toward_arg)))


def main():
    model = SizeAgnosticQNet().to(DEVICE)
    model.load_state_dict(torch.load(CKPT, map_location=DEVICE)); model.eval()

    print(f"{'board':>5} | {'food_margin':>11} | {'danger_margin':>13} | {'spread':>7} "
          f"| {'food/spread':>11} | {'toward-argmax':>13}")
    fm, dm, sp, ta = [], [], [], []
    for b in SIZES:
        f, d, s, t = margins(model, b)
        fm.append(f); dm.append(d); sp.append(s); ta.append(t)
        print(f"{b:5d} | {f:11.3f} | {d:13.3f} | {s:7.3f} | {f/s:11.3f} | {t*100:12.1f}%", flush=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(SIZES, fm, "o-", color="#dc2626", label="food margin  (steer-to-food signal)")
    ax.plot(SIZES, dm, "s-", color="#2563eb", label="danger margin  (avoid-death, local)")
    ax.axhline(0, ls=":", color="#9ca3af", lw=1)
    ax.axvline(6, ls=":", color="gray", alpha=0.5)
    ax.set_xlabel("board size (zero-shot)"); ax.set_ylabel("advantage margin (Q units)")
    ax.set_title("Advantage-margin dilution: food signal collapses, danger signal holds")
    ax.legend()
    fig.tight_layout(); fig.savefig(f"{OUT}/margin_dilution.png", dpi=130)
    print(f"\nWrote {OUT}/margin_dilution.png")


if __name__ == "__main__":
    main()
