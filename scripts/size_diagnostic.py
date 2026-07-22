"""
Why does the size-agnostic agent cliff at ~12x12? Test the pooling-dilution hypothesis.

Prediction of the dissociation:
  - DANGER (local: collision at the head cell) survives global pooling -> decodable at all sizes.
  - FOOD-DIRECTION (global-relational: food vs head across the board) is diluted by pooling as
    the board grows -> decodability CLIFFS at the same board size as the score.

We probe the trained 6x6 model's pooled hidden for these features on each board size, using
states collected by the model's own (eps-mixed) play on that board. Reuses the probe rig.
"""
import os, sys, random, numpy as np, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, "/Users/saheb/home/rl-snake")
import scripts.probe_learned_features as P
from scripts.size_transfer import SizeAgnosticQNet, get_conv_state, DEVICE, greedy_score
from envs.snake_game import SnakeGame

OUT = "/Users/saheb/home/rl-snake/size_transfer"
CKPT = f"{OUT}/sizeagnostic_6x6_best.pth"
SIZES = [6, 8, 10, 12, 16]
FOOD = ["food_left", "food_right", "food_up", "food_down"]
DANGER = ["danger_straight", "danger_right", "danger_left"]


@torch.no_grad()
def collect(model, board, n=4000, eps=0.3):
    grids, Y = [], []
    while len(Y) < n:
        game = SnakeGame(board_size=board, reward_mode="dense"); game.reset()
        for _ in range(board * board * 4):
            s = get_conv_state(game)
            grids.append(s); Y.append(P.extract_features(game))
            if random.random() < eps:
                a = random.randint(0, 3)
            else:
                a = int(torch.argmax(model(torch.tensor(s[None], device=DEVICE))).item())
            _, _, done, _ = game.step(a)
            if done or len(Y) >= n:
                break
    return np.array(grids, np.float32), np.array(Y, np.float32)


@torch.no_grad()
def hidden(model, grids, batch=2048):
    out = []
    for i in range(0, len(grids), batch):
        out.append(model.hidden(torch.tensor(grids[i:i + batch], device=DEVICE)).cpu().numpy())
    return np.concatenate(out).astype(np.float32)


model = SizeAgnosticQNet().to(DEVICE)
model.load_state_dict(torch.load(CKPT, map_location=DEVICE)); model.eval()

food_dec, danger_dec, scores = [], [], []
for b in SIZES:
    grids, Y = collect(model, b, n=4000)
    H = hidden(model, grids)
    res = P.probe_all(H, Y)
    fd = np.mean([res[n]["score"] for n in FOOD])
    dg = np.mean([res[n]["score"] for n in DANGER])
    gs, _ = greedy_score(model, b, n=30)
    food_dec.append(fd); danger_dec.append(dg); scores.append(gs)
    print(f"board {b:2d}: food-dir decodability {fd:.2f} | danger {dg:.2f} | greedy score {gs:.1f}", flush=True)

# normalize score to [0,1] for overlay
sc = np.array(scores); sc_n = sc / sc.max()
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(SIZES, food_dec, "o-", color="#dc2626", label="food-direction decodability (global)")
ax.plot(SIZES, danger_dec, "s-", color="#2563eb", label="danger decodability (local)")
ax.plot(SIZES, sc_n, "^--", color="#6b7280", label="greedy score (normalized)")
ax.axvline(6, ls=":", color="gray", alpha=0.6)
ax.set_xlabel("board size (zero-shot)"); ax.set_ylabel("decodability / normalized score")
ax.set_title("Pooling-dilution test: does food-direction cliff while danger survives?")
ax.legend(); ax.set_ylim(0, 1.05)
fig.tight_layout(); fig.savefig(f"{OUT}/pooling_dilution.png", dpi=130)
print(f"\nWrote {OUT}/pooling_dilution.png")
