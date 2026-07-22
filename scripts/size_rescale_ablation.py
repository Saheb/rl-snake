"""
Causal check (no retrain): is the food-margin dilution caused by avg-pool's / (H*W) normalization?

Identity: sum-pool = avg-pool * (H*W). At the 6x6 TRAINING size that factor (36) is a constant the
head absorbs, so an avg-pool net and a sum-pool net trained only on 6x6 are the SAME network. They
diverge only at TEST on other board sizes. So we can convert the frozen 6x6 avg-pool model into its
sum-pool-equivalent at inference by scaling the GAP branch by k = area / 36 = B^2 / 36 — a controlled
swap of ONLY the area-normalization, on frozen features.

If the avg-pool ÷area normalization is the culprit, k = B^2/36 should:
  - restore the food margin toward its 6x6 value (~0.96) and flatten it across board size,
  - lift toward-argmax / greedy score on big boards,
  - leave the danger margin (max-pool branch, untouched) roughly unchanged.

Caveat (why this is a check, not final proof): a UNIFORM factor over-corrects channels that don't
dilute (ones active on ~every cell), so expect partial, not perfect, recovery. Full isolation still
needs the retrain-with-area-invariant-pool ablation; this de-risks it and sharpens the prediction.
"""
import os, sys, random, numpy as np, torch
sys.path.insert(0, "/Users/saheb/home/rl-snake")
from scripts.size_transfer import SizeAgnosticQNet, get_conv_state, DEVICE
from envs.snake_game import SnakeGame

OUT = "/Users/saheb/home/rl-snake/size_transfer"
CKPT = f"{OUT}/sizeagnostic_6x6_best.pth"
SIZES = [6, 8, 10, 12, 16]
TRAIN_AREA = 36
DELTAS = [(-1, 0), (0, 1), (1, 0), (0, -1)]


def scaled_Q(model, x, k):
    """Q with the GAP (avg-pool) branch scaled by k; k=1 reproduces the original net."""
    h = model.conv(x)
    gap = h.mean(dim=(2, 3)) * k
    gmp = h.amax(dim=(2, 3))
    z = torch.cat([gap, gmp], dim=1)
    a = model.advantage_stream(z); v = model.value_stream(z)
    return v + (a - a.mean(dim=1, keepdim=True))


@torch.no_grad()
def margins(model, board, k, n_states=3000, eps=0.3, seed=0):
    random.seed(seed); np.random.seed(seed)
    food_m, danger_m, toward_arg = [], [], []
    collected = 0
    while collected < n_states:
        game = SnakeGame(board_size=board, reward_mode="dense"); game.reset()
        for _ in range(board * board * 4):
            head = game.snake_position[-1]; food = game.food_position
            if food is None:
                break
            d0 = abs(head[0] - food[0]) + abs(head[1] - food[1])
            q = scaled_Q(model, torch.tensor(get_conv_state(game)[None], device=DEVICE), k).cpu().numpy().ravel()
            toward = np.zeros(4, bool); fatal = np.zeros(4, bool)
            for a, (dx, dy) in enumerate(DELTAS):
                nh = (head[0] + dx, head[1] + dy)
                toward[a] = abs(nh[0] - food[0]) + abs(nh[1] - food[1]) < d0
                fatal[a] = game._check_collision(nh, nh == food)
            if toward.any() and (~toward).any():
                food_m.append(q[toward].max() - q[~toward].max())
                toward_arg.append(bool(toward[int(q.argmax())]))
            if fatal.any() and (~fatal).any():
                danger_m.append(q[~fatal].max() - q[fatal].max())
            collected += 1
            a = random.randint(0, 3) if random.random() < eps else int(q.argmax())
            _, _, done, _ = game.step(a)
            if done or collected >= n_states:
                break
    return float(np.mean(food_m)), float(np.mean(danger_m)), float(np.mean(toward_arg))


@torch.no_grad()
def greedy(model, board, k, n=30):
    tot = []
    for _ in range(n):
        game = SnakeGame(board_size=board, reward_mode="dense"); game.reset()
        info = {"score": 0}
        for _ in range(board * board * 4):
            q = scaled_Q(model, torch.tensor(get_conv_state(game)[None], device=DEVICE), k)
            _, _, done, info = game.step(int(torch.argmax(q).item()))
            if done:
                break
        tot.append(info["score"])
    return float(np.mean(tot))


def main():
    model = SizeAgnosticQNet().to(DEVICE)
    model.load_state_dict(torch.load(CKPT, map_location=DEVICE)); model.eval()
    print(f"{'board':>5} | {'k=area/36':>9} || {'food_m base':>11} {'food_m resc':>11} "
          f"| {'danger base':>11} {'danger resc':>11} | {'tw-arg base':>11} {'tw-arg resc':>11} "
          f"| {'greedy base':>11} {'greedy resc':>11}")
    for b in SIZES:
        k = b * b / TRAIN_AREA
        f0, d0, t0 = margins(model, b, 1.0)
        f1, d1, t1 = margins(model, b, k)
        g0 = greedy(model, b, 1.0); g1 = greedy(model, b, k)
        print(f"{b:5d} | {k:9.2f} || {f0:11.3f} {f1:11.3f} | {d0:11.3f} {d1:11.3f} "
              f"| {t0*100:10.1f}% {t1*100:10.1f}% | {g0:11.2f} {g1:11.2f}", flush=True)


if __name__ == "__main__":
    main()
