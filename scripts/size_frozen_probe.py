"""
Experiment A — the reconciling experiment: FROZEN vs REFIT linear probe across board sizes.

The open thread: on 16x16 the agent random-walks, yet a linear probe recovers food-direction
from its pooled hidden at ~0.90. But that probe was REFIT on each board's own states. The
policy head is the SAME kind of linear reader (advantage_stream's final Linear reads the 128-d
hidden) — but FROZEN at 6x6 statistics.

So we run one reader in both roles, for food-direction:
  - FROZEN probe: fit the logistic on 6x6 hidden only, standardize with 6x6 mean/std, apply
    unchanged to every board. This is the analog of the frozen policy head.
  - REFIT probe: fresh logistic fit + standardization per board (reproduces the ~0.90).

Prediction if the mechanism is pooled-representation distribution shift:
  frozen AUC decays toward chance (0.5) and TRACKS the policy's toward-food%, while refit AUC
  stays high. => the information is present (refit works) but a FIXED linear reader off its
  training distribution cannot use it (frozen fails, exactly like the policy). decodable != used.
"""
import os, sys, random, numpy as np, torch, torch.nn as nn
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, "/Users/saheb/home/rl-snake")
import scripts.probe_learned_features as P
from scripts.size_transfer import SizeAgnosticQNet, get_conv_state, DEVICE, greedy_score
from envs.snake_game import SnakeGame

OUT = "/Users/saheb/home/rl-snake/size_transfer"
CKPT = f"{OUT}/sizeagnostic_6x6_best.pth"
SIZES = [6, 8, 10, 12, 16]
FOOD = ["food_left", "food_right", "food_up", "food_down"]
FOOD_IDX = [P.FEATURE_NAMES.index(f) for f in FOOD]
NSTATES = 4000


# ---- state collection + hidden (pooled advantage hidden the policy also reads) ----
@torch.no_grad()
def collect(model, board, n=NSTATES, eps=0.3, seed=0):
    random.seed(seed); np.random.seed(seed)
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


# ---- standardization: FIT once (returns params), APPLY anywhere ----
def std_fit(X):
    mu = X.mean(0); sd = X.std(0); live = sd > 1e-2   # ignore ~constant units (no 1/tiny blowup)
    return mu, sd, live


def std_apply(X, mu, sd, live):
    Xc = X - mu
    Xc[:, live] /= sd[live]; Xc[:, ~live] = 0.0
    return np.clip(Xc, -8.0, 8.0)          # bound cross-board influence, mirror a sane linear reader


# ---- logistic probe as raw weights so it can be frozen and re-applied ----
def logistic_fit(X, y, steps=400, lr=0.05, lam=1e-3):
    Xt = torch.tensor(X, dtype=torch.float32); yt = torch.tensor(y, dtype=torch.float32)
    lin = nn.Linear(X.shape[1], 1)
    opt = torch.optim.Adam(lin.parameters(), lr=lr, weight_decay=lam)
    for _ in range(steps):
        opt.zero_grad()
        loss = nn.functional.binary_cross_entropy_with_logits(lin(Xt).squeeze(1), yt)
        loss.backward(); opt.step()
    return lin.weight.detach().numpy().ravel(), float(lin.bias.detach())


def logistic_score(X, w, b):
    s = X.astype(np.float64) @ w.astype(np.float64) + b
    return np.nan_to_num(s, nan=0.0, posinf=1e6, neginf=-1e6)


def auc(y, score):
    return P._auc(y.astype(int), score)


# ---- policy behaviour: fraction of greedy moves that reduce Manhattan dist to food ----
@torch.no_grad()
def toward_food(model, board, n=40):
    hits = tot = 0
    for _ in range(n):
        game = SnakeGame(board_size=board, reward_mode="dense"); game.reset()
        prev = 0
        for _ in range(board * board * 4):
            head = game.snake_position[-1]; food = game.food_position
            if food is None:
                break
            d0 = abs(head[0] - food[0]) + abs(head[1] - food[1])
            s = get_conv_state(game)
            a = int(torch.argmax(model(torch.tensor(s[None], device=DEVICE))).item())
            _, _, done, info = game.step(a)
            sc = info.get("score", prev)
            if sc > prev:                       # ate food this step -> moved onto it
                hits += 1; tot += 1; prev = sc
            elif not done:
                nh = game.snake_position[-1]
                d1 = abs(nh[0] - food[0]) + abs(nh[1] - food[1])
                tot += 1
                if d1 < d0:
                    hits += 1
            else:
                tot += 1                        # died: count as not-toward
            if done:
                break
    return hits / max(tot, 1)


def main():
    model = SizeAgnosticQNet().to(DEVICE)
    model.load_state_dict(torch.load(CKPT, map_location=DEVICE)); model.eval()

    # --- fit the FROZEN reference probe on 6x6 only ---
    g6, Y6 = collect(model, 6)
    H6 = hidden(model, g6)
    mu, sd, live = std_fit(H6)
    H6s = std_apply(H6, mu, sd, live)
    frozen_probes = {j: logistic_fit(H6s, Y6[:, j]) for j in FOOD_IDX}   # per food feature

    print(f"{'board':>5} | {'frozen-AUC':>10} | {'refit-AUC':>9} | {'toward-food%':>12} | {'greedy':>6}")
    frozen_a, refit_a, tf_a, gs_a = [], [], [], []
    for b in SIZES:
        grids, Y = collect(model, b)
        H = hidden(model, grids)

        # FROZEN: 6x6 standardization + 6x6 weights, applied unchanged
        Hf = std_apply(H, mu, sd, live)
        fr = np.mean([auc(Y[:, j], logistic_score(Hf, *frozen_probes[j])) for j in FOOD_IDX])

        # REFIT: fresh standardization + fresh fit on this board (train/test split)
        n = len(H); perm = np.random.default_rng(1).permutation(n); cut = int(0.7 * n)
        tr, te = perm[:cut], perm[cut:]
        mub, sdb, liveb = std_fit(H[tr])
        Htr = std_apply(H[tr].copy(), mub, sdb, liveb); Hte = std_apply(H[te].copy(), mub, sdb, liveb)
        re = np.mean([auc(Y[te, j], logistic_score(Hte, *logistic_fit(Htr, Y[tr, j]))) for j in FOOD_IDX])

        tf = toward_food(model, b)
        gs, _ = greedy_score(model, b, n=30)
        frozen_a.append(fr); refit_a.append(re); tf_a.append(tf); gs_a.append(gs)
        print(f"{b:5d} | {fr:10.3f} | {re:9.3f} | {tf*100:11.1f}% | {gs:6.2f}", flush=True)

    # --- plot: frozen vs refit AUC, overlaid with policy toward-food% ---
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(SIZES, refit_a, "o-", color="#16a34a", label="REFIT probe AUC (info present)")
    ax.plot(SIZES, frozen_a, "s-", color="#dc2626", label="FROZEN probe AUC (6x6-fit reader)")
    ax.plot(SIZES, tf_a, "^--", color="#6b7280", label="policy toward-food (fraction)")
    ax.axhline(0.5, ls=":", color="#9ca3af", lw=1)
    ax.axvline(6, ls=":", color="gray", alpha=0.5)
    ax.set_xlabel("board size (zero-shot)"); ax.set_ylabel("AUC / toward-food fraction")
    ax.set_title("Frozen vs refit reader: does a fixed linear reader fail like the policy?")
    ax.legend(); ax.set_ylim(0.4, 1.02)
    fig.tight_layout(); fig.savefig(f"{OUT}/frozen_probe.png", dpi=130)
    print(f"\nWrote {OUT}/frozen_probe.png")


if __name__ == "__main__":
    main()
