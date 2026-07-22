"""
Probe: did a from-pixels ConvNet *learn* the hand-crafted Snake features?

Structure-vs-scale, measured. The Feature-DQN is handed 24 engineered features
(danger flags, flood-fill open space, distances). The ConvNet sees only raw
pixels and still wins. Question: did it re-learn that structure internally?

Method (linear probing):
  - Train a pure-pixel ConvDuelingQNet on 6x6, checkpointing through training.
  - For each checkpoint, roll out states and record, per state:
      (a) the net's 128-d action-selection hidden,
      (b) the 24 hand-crafted features (reused verbatim from the Feature-DQN),
      (c) the raw 3-channel pixels.
  - Fit a LINEAR probe from each representation to each feature. If a linear map
    recovers the feature, the structure is linearly present in that representation.

Controls:
  - random-init net  -> is the structure LEARNED, or already there from architecture?
  - raw pixels       -> is the structure TRIVIALLY in the input?

No external deps beyond the repo's own (numpy/torch); probes are hand-rolled
(closed-form ridge + small logistic fit) so nothing in the venv is mutated.
"""
import os, sys, json, math, random, collections, argparse
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from envs.snake_game import SnakeGame
from utils.per import PrioritizedReplayBuffer

DEVICE = torch.device("cuda" if torch.cuda.is_available()
                      else "mps" if torch.backends.mps.is_available() else "cpu")
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "probe_results")
CKPT_DIR = os.path.join(OUT, "ckpts")

# ---------------------------------------------------------------------------
# Pixel state + model (replicated verbatim from scripts/train_dqn_conv.py so the
# probe is self-contained and free of wandb/argparse import side-effects)
# ---------------------------------------------------------------------------
def get_conv_state(game):
    H = W = game.board_size
    state = np.zeros((3, H, W), dtype=np.float32)
    snake = list(game.snake_position)
    for pos in snake[:-1]:
        state[0, pos[0], pos[1]] = 1.0
    head = snake[-1]
    state[1, head[0], head[1]] = 1.0
    if game.food_position:
        state[2, game.food_position[0], game.food_position[1]] = 1.0
    return state


class ConvDuelingQNet(nn.Module):
    def __init__(self, board_size, n_actions=4):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(),
        )
        flat = 128 * board_size * board_size
        self.value_stream = nn.Sequential(nn.Linear(flat, 128), nn.ReLU(), nn.Linear(128, 1))
        self.advantage_stream = nn.Sequential(nn.Linear(flat, 128), nn.ReLU(), nn.Linear(128, n_actions))

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        val = self.value_stream(x)
        adv = self.advantage_stream(x)
        return val + (adv - adv.mean(dim=1, keepdim=True))

    def hidden(self, x):
        """The 128-d action-selection representation (advantage stream, post-ReLU)."""
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.advantage_stream[1](self.advantage_stream[0](x))  # Linear -> ReLU


# ---------------------------------------------------------------------------
# The 24 hand-crafted features (reused verbatim from scripts/train_dqn.py)
# ---------------------------------------------------------------------------
def _reachable_count(game, start_pos):
    obstacles = set(game.snake_position)
    if start_pos in obstacles or game._check_collision(start_pos, False):
        return 0
    visited = {start_pos}
    queue = collections.deque([start_pos])
    count = 0
    max_search = max(len(game.snake_position) * 3, 20)
    while queue and count < max_search:
        cx, cy = queue.popleft()
        count += 1
        for nx, ny in [(cx-1, cy), (cx+1, cy), (cx, cy-1), (cx, cy+1)]:
            if (0 <= nx < game.board_size and 0 <= ny < game.board_size
                    and (nx, ny) not in obstacles and (nx, ny) not in visited):
                visited.add((nx, ny))
                queue.append((nx, ny))
    return count


FEATURE_NAMES = [
    "danger_straight", "danger_right", "danger_left",           # 0-2
    "dir_l", "dir_r", "dir_u", "dir_d",                          # 3-6
    "food_left", "food_right", "food_up", "food_down",          # 7-10
    "flood_straight", "flood_right", "flood_left",              # 11-13
    "tail_up", "tail_down", "tail_left", "tail_right",          # 14-17
    "snake_length", "food_dist",                                # 18-19
    "wall_up", "wall_down", "wall_left", "wall_right",          # 20-23
]
CONTINUOUS_IDX = {11, 12, 13, 18, 19, 20, 21, 22, 23}
TIER = {  # structural depth, for the narrative
    "trivial/positional": [3, 4, 5, 6, 7, 8, 9, 10, 20, 21, 22, 23],
    "local lookahead":    [0, 1, 2],
    "semi-global":        [14, 15, 16, 17, 18, 19],
    "global structural":  [11, 12, 13],
}


def extract_features(game):
    head = game.snake_position[-1]
    point_l = (head[0], head[1] - 1); point_r = (head[0], head[1] + 1)
    point_u = (head[0] - 1, head[1]); point_d = (head[0] + 1, head[1])

    if len(game.snake_position) > 1:
        neck = game.snake_position[-2]
        if head[0] < neck[0]:   dir_u, dir_r, dir_d, dir_l = 1, 0, 0, 0
        elif head[0] > neck[0]: dir_u, dir_r, dir_d, dir_l = 0, 0, 1, 0
        elif head[1] < neck[1]: dir_u, dir_r, dir_d, dir_l = 0, 0, 0, 1
        else:                   dir_u, dir_r, dir_d, dir_l = 0, 1, 0, 0
    else:
        dir_u, dir_r, dir_d, dir_l = 0, 1, 0, 0

    cw = [point_u, point_r, point_d, point_l]
    idx = 0 if dir_u else 1 if dir_r else 2 if dir_d else 3
    pos_straight, pos_right, pos_left = cw[idx], cw[(idx + 1) % 4], cw[(idx - 1) % 4]

    total = game.board_size * game.board_size
    b = game.board_size
    flood_s = _reachable_count(game, pos_straight) / total
    flood_r = _reachable_count(game, pos_right) / total
    flood_l = _reachable_count(game, pos_left) / total

    tail = game.snake_position[0]
    tail_up, tail_down = int(tail[0] < head[0]), int(tail[0] > head[0])
    tail_left, tail_right = int(tail[1] < head[1]), int(tail[1] > head[1])
    snake_len = len(game.snake_position) / total
    if game.food_position:
        food_dist = (abs(head[0]-game.food_position[0]) + abs(head[1]-game.food_position[1])) / (2*(b-1))
    else:
        food_dist = 0.0
    wall_up, wall_down = head[0]/(b-1), (b-1-head[0])/(b-1)
    wall_left, wall_right = head[1]/(b-1), (b-1-head[1])/(b-1)

    danger_s = int((dir_r and game._check_collision(point_r)) or (dir_l and game._check_collision(point_l))
                   or (dir_u and game._check_collision(point_u)) or (dir_d and game._check_collision(point_d)))
    danger_r = int((dir_u and game._check_collision(point_r)) or (dir_d and game._check_collision(point_l))
                   or (dir_l and game._check_collision(point_u)) or (dir_r and game._check_collision(point_d)))
    danger_l = int((dir_d and game._check_collision(point_r)) or (dir_u and game._check_collision(point_l))
                   or (dir_r and game._check_collision(point_u)) or (dir_l and game._check_collision(point_d)))
    food_left  = int(game.food_position[1] < head[1]) if game.food_position else 0
    food_right = int(game.food_position[1] > head[1]) if game.food_position else 0
    food_up    = int(game.food_position[0] < head[0]) if game.food_position else 0
    food_down  = int(game.food_position[0] > head[0]) if game.food_position else 0

    return np.array([
        danger_s, danger_r, danger_l, dir_l, dir_r, dir_u, dir_d,
        food_left, food_right, food_up, food_down, flood_s, flood_r, flood_l,
        tail_up, tail_down, tail_left, tail_right, snake_len, food_dist,
        wall_up, wall_down, wall_left, wall_right,
    ], dtype=np.float32)


# ---------------------------------------------------------------------------
# Minimal DQN training (uniform replay, 1-step TD) — just enough to learn 6x6.
# Hyperparameters follow the repo's sweep-tuned ConvDQN settings.
# ---------------------------------------------------------------------------
@torch.no_grad()
def greedy_score(model, board, n=20):
    """Mean score under greedy (eps=0) play — the honest measure of policy quality."""
    model.eval(); tot = []
    for _ in range(n):
        game = SnakeGame(board_size=board, reward_mode="dense"); game.reset()
        info = {"score": 0}
        for _ in range(board * board * 4):
            s = get_conv_state(game)
            a = int(torch.argmax(model(torch.tensor(s[None], device=DEVICE))).item())
            _, _, done, info = game.step(a)
            if done:
                break
        tot.append(info["score"])
    return float(np.mean(tot))


def train_convnet(board=6, n_games=4000, ckpt_games=None, seed=0, reward_mode="sparse"):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if ckpt_games is None:  # log-spaced snapshots + random-init (0) for the emergence curve
        pts = sorted({0} | {int(round(n_games * f)) for f in (0.05, 0.12, 0.25, 0.5, 1.0)})
        ckpt_games = tuple(pts)
    os.makedirs(CKPT_DIR, exist_ok=True)
    GAMMA, LR, BATCH, CAP, LEARN_START, TRAIN_CALLS = 0.995, 1.26e-4, 512, 100_000, 5000, 10
    EPS_END = 0.01
    EPS_DECAY_GAMES = 0.7 * n_games   # decay over 70% of training (matches repo's tuned schedule)
    max_steps = board * board * 4

    model = ConvDuelingQNet(board).to(DEVICE)
    target = ConvDuelingQNet(board).to(DEVICE); target.load_state_dict(model.state_dict()); target.eval()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    mem = PrioritizedReplayBuffer(capacity=CAP, priority_cap=1.0)  # cap prevents death-state Q-collapse
    total_steps = 0; best_greedy = -1.0
    ckpt_games = set(ckpt_games)

    def save(g):
        path = os.path.join(CKPT_DIR, f"conv_{board}x{board}_g{g:05d}.pth")
        torch.save(model.state_dict(), path)
        return path

    if 0 in ckpt_games:
        save(0)  # random-init snapshot (== the control, but also start of emergence curve)

    scores = collections.deque(maxlen=100)
    for g in range(1, n_games + 1):
        eps = max(EPS_END, 1.0 - g / EPS_DECAY_GAMES)  # per-game decay
        game = SnakeGame(board_size=board, reward_mode=reward_mode)
        game.reset()
        s = get_conv_state(game)
        for _ in range(max_steps):
            if random.random() < eps:
                a = random.randint(0, 3)
            else:
                with torch.no_grad():
                    a = int(torch.argmax(model(torch.tensor(s[None], device=DEVICE))).item())
            _, r, done, info = game.step(a)
            ns = get_conv_state(game)
            mem.add((s, a, r, ns, done))
            s = ns
            total_steps += 1
            if done:
                break
        scores.append(info.get("score", 0))

        if len(mem) >= LEARN_START:
            for _ in range(TRAIN_CALLS):
                batch, indices, is_w = mem.sample(BATCH)
                if len(batch) < 2:
                    continue
                is_w = np.asarray(is_w)[:len(batch)]  # sample() may skip empty leaves
                bs, ba, br, bns, bd = zip(*batch)
                bs = torch.tensor(np.array(bs), device=DEVICE)
                bns = torch.tensor(np.array(bns), device=DEVICE)
                ba = torch.tensor(ba, device=DEVICE, dtype=torch.long)
                br = torch.tensor(br, device=DEVICE, dtype=torch.float)
                bd = torch.tensor(bd, device=DEVICE, dtype=torch.bool)
                is_w = torch.tensor(is_w, device=DEVICE)
                q = model(bs).gather(1, ba[:, None]).squeeze(1)
                with torch.no_grad():
                    na = torch.argmax(model(bns), dim=1)
                    nq = target(bns).gather(1, na[:, None]).squeeze(1)
                    tgt = br + GAMMA * nq * (~bd)
                td = q - tgt
                loss = (is_w * td.pow(2)).mean()
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 10.0); opt.step()
                mem.update_priorities(indices, td.abs().detach().cpu().numpy())
                for tp, lp in zip(target.parameters(), model.parameters()):
                    tp.data.copy_(0.005 * lp.data + 0.995 * tp.data)

        if g % 500 == 0:
            gs = greedy_score(model, board, n=20)
            print(f"  game {g:5d} | eps {eps:.2f} | train_score(100) {np.mean(scores):.2f} "
                  f"| GREEDY {gs:.2f} | steps {total_steps}")
            if gs > best_greedy:
                best_greedy = gs
                torch.save(model.state_dict(), os.path.join(CKPT_DIR, f"conv_{board}x{board}_best.pth"))
        if g in ckpt_games:
            print(f"  [checkpoint] game {g} -> {save(g)}")
    print(f"  best greedy score: {best_greedy:.2f}")
    return board


# ---------------------------------------------------------------------------
# Dataset collection + linear probes
# ---------------------------------------------------------------------------
@torch.no_grad()
def collect_states(gen_model, board, n_states=12000, epsilon=0.2, seed=1):
    """Roll out ONE policy to build a FIXED evaluation set of states + features.
    Every checkpoint is later probed on these same states, so the emergence curve
    reflects representation change, not state-distribution shift."""
    random.seed(seed); np.random.seed(seed)
    grids = []; Y = []
    max_steps = board * board * 4
    while len(Y) < n_states:
        game = SnakeGame(board_size=board, reward_mode="dense"); game.reset()
        for _ in range(max_steps):
            s = get_conv_state(game)
            grids.append(s); Y.append(extract_features(game))
            if random.random() < epsilon:
                a = random.randint(0, 3)
            else:
                a = int(torch.argmax(gen_model(torch.tensor(s[None], device=DEVICE))).item())
            _, _, done, _ = game.step(a)
            if done or len(Y) >= n_states:
                break
    return np.array(grids, np.float32), np.array(Y, np.float32)


@torch.no_grad()
def hidden_reps(model, grids, batch=2048):
    out = []
    for i in range(0, len(grids), batch):
        t = torch.tensor(grids[i:i + batch], device=DEVICE)
        out.append(model.hidden(t).cpu().numpy())
    return np.concatenate(out).astype(np.float32)


def _standardize(Xtr, Xte):
    mu = Xtr.mean(0); sd = Xtr.std(0)
    live = sd > 1e-4                      # zero out (near-)dead units instead of /~0 (overflow guard)
    Xtr = Xtr - mu; Xte = Xte - mu
    Xtr[:, live] /= sd[live]; Xte[:, live] /= sd[live]
    Xtr[:, ~live] = 0.0; Xte[:, ~live] = 0.0
    return Xtr, Xte


def _ridge_r2(Xtr, ytr, Xte, yte, lam=10.0):
    if ytr.std() < 1e-6 or yte.std() < 1e-6:   # (near-)constant target: nothing to explain
        return 0.0
    Xtr = np.c_[Xtr, np.ones(len(Xtr))].astype(np.float64)
    Xte = np.c_[Xte, np.ones(len(Xte))].astype(np.float64)
    ytr = ytr.astype(np.float64); yte = yte.astype(np.float64)
    d = Xtr.shape[1]
    reg = lam * np.eye(d); reg[-1, -1] = 0.0  # regularize weights, not the bias term
    with np.errstate(all="ignore"):
        w = np.linalg.solve(Xtr.T @ Xtr + reg, Xtr.T @ ytr)
        pred = Xte @ w
    ss_res = ((yte - pred) ** 2).sum(); ss_tot = ((yte - yte.mean()) ** 2).sum() + 1e-9
    r2 = 1.0 - ss_res / ss_tot
    return float(max(-1.0, min(1.0, r2))) if np.isfinite(r2) else 0.0


def _auc(y, score):
    pos = score[y == 1]; neg = score[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(np.concatenate([neg, pos]))
    ranks = np.empty_like(order, dtype=float); ranks[order] = np.arange(1, len(order) + 1)
    r_pos = ranks[len(neg):].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def _logistic_auc(Xtr, ytr, Xte, yte, steps=300, lr=0.05, lam=1e-3):
    Xtr_t = torch.tensor(Xtr, dtype=torch.float32); Xte_t = torch.tensor(Xte, dtype=torch.float32)
    ytr_t = torch.tensor(ytr, dtype=torch.float32)
    lin = nn.Linear(Xtr.shape[1], 1)
    opt = torch.optim.Adam(lin.parameters(), lr=lr, weight_decay=lam)
    for _ in range(steps):
        opt.zero_grad()
        loss = nn.functional.binary_cross_entropy_with_logits(lin(Xtr_t).squeeze(1), ytr_t)
        loss.backward(); opt.step()
    with torch.no_grad():
        sc = lin(Xte_t).squeeze(1).numpy()
    return _auc(yte, sc)


def probe_all(X, Y, split=0.7, seed=2):
    """Return {feature: {'metric','score'}} where score in [0,1] (0=no info)."""
    rng = np.random.default_rng(seed)
    n = len(X); perm = rng.permutation(n); cut = int(split * n)
    tr, te = perm[:cut], perm[cut:]
    Xtr, Xte = _standardize(X[tr], X[te])
    out = {}
    for j, name in enumerate(FEATURE_NAMES):
        ytr, yte = Y[tr, j], Y[te, j]
        if j in CONTINUOUS_IDX:
            r2 = _ridge_r2(Xtr, ytr, Xte, yte)
            out[name] = {"metric": "R2", "raw": r2, "score": max(0.0, r2)}
        else:
            if len(np.unique(ytr)) < 2 or len(np.unique(yte)) < 2:
                out[name] = {"metric": "AUC", "raw": float("nan"), "score": 0.0}
            else:
                auc = _logistic_auc(Xtr, ytr, Xte, yte)
                out[name] = {"metric": "AUC", "raw": auc, "score": max(0.0, (auc - 0.5) * 2)}
    return out


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def sorted_ckpts(board):
    fs = [f for f in os.listdir(CKPT_DIR) if f.startswith(f"conv_{board}x{board}_g") and f.endswith(".pth")]
    return sorted(fs, key=lambda f: int(f.split("_g")[1].split(".")[0]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--board", type=int, default=6)
    ap.add_argument("--games", type=int, default=4000)
    ap.add_argument("--states", type=int, default=12000)
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--reward", default="sparse", choices=["dense", "sparse", "pure_sparse"])
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    if not args.skip_train:
        print(f"== Training pure-pixel ConvDuelingQNet on {args.board}x{args.board} "
              f"(reward={args.reward}) ==")
        train_convnet(board=args.board, n_games=args.games, reward_mode=args.reward)

    board = args.board
    ckpts = sorted_ckpts(board)
    assert ckpts, "no checkpoints found; run without --skip-train first"
    print(f"== Probing {len(ckpts)} checkpoints on a FIXED evaluation set ==")

    def load(f):
        m = ConvDuelingQNet(board).to(DEVICE)
        m.load_state_dict(torch.load(os.path.join(CKPT_DIR, f), map_location=DEVICE))
        return m.eval()

    # Fixed evaluation states: generated once by the STRONGEST policy (best-by-greedy)
    # so the set contains the long-snake states where flood-fill actually varies.
    best_path = os.path.join(CKPT_DIR, f"conv_{board}x{board}_best.pth")
    gen_file = f"conv_{board}x{board}_best.pth" if os.path.exists(best_path) else ckpts[-1]
    print(f"   eval-state generator: {gen_file}")
    gen = load(gen_file)
    grids, Y = collect_states(gen, board, n_states=args.states, epsilon=0.1)
    lens = Y[:, 18] * board * board
    print(f"   fixed eval set: {len(Y)} states | snake_len mean {lens.mean():.1f} max {lens.max():.0f} "
          f"| flood_straight std {Y[:,11].std():.3f}")

    results = {"board": board, "feature_tiers": TIER, "checkpoints": {}}
    # Raw-pixel baseline (control): probe the input pixels of the SAME states.
    results["pixels_baseline"] = probe_all(grids.reshape(len(grids), -1), Y)

    for f in ckpts:
        g = int(f.split("_g")[1].split(".")[0])
        H = hidden_reps(load(f), grids)
        res = probe_all(H, Y)
        results["checkpoints"][g] = res
        flood = np.mean([res[n]["score"] for n in ("flood_straight", "flood_right", "flood_left")])
        triv = np.mean([res[FEATURE_NAMES[i]]["score"] for i in TIER["trivial/positional"]])
        tag = "random-init" if g == 0 else f"game {g}"
        print(f"  {tag:>12}: flood-fill {flood:.2f} | trivial/positional {triv:.2f}")

    with open(os.path.join(OUT, "probe_results.json"), "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\nWrote {os.path.join(OUT, 'probe_results.json')}")
    try:
        make_figures(results)
    except Exception as e:
        print(f"(figure step skipped: {e})")


def make_figures(results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(OUT, exist_ok=True)

    ckpts = sorted(results["checkpoints"].keys(), key=int)
    final = str(ckpts[-1]) if False else ckpts[-1]
    trained = results["checkpoints"][final]
    random_init = results["checkpoints"][ckpts[0]]
    pixels = results["pixels_baseline"]

    # Fig 1: per-feature decodability, trained vs random-init vs pixels, grouped by tier
    order = [i for tier in TIER.values() for i in tier]
    names = [FEATURE_NAMES[i] for i in order]
    tr = [trained[n]["score"] for n in names]
    ri = [random_init[n]["score"] for n in names]
    px = [pixels[n]["score"] for n in names]
    x = np.arange(len(names)); w = 0.27
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(x - w, tr, w, label="trained net", color="#2563eb")
    ax.bar(x, ri, w, label="random-init net", color="#9ca3af")
    ax.bar(x + w, px, w, label="raw pixels", color="#f59e0b")
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=60, ha="right", fontsize=8)
    ax.set_ylabel("linear decodability (0-1)")
    ax.set_title(f"Did the from-pixels ConvNet learn the hand-crafted features?  (6x6, {final} games)")
    ax.legend()
    # tier separators
    b = 0
    for tier, idxs in TIER.items():
        ax.axvline(b - 0.5, color="k", lw=0.4, alpha=0.3)
        ax.text(b + len(idxs)/2 - 0.5, 1.02, tier, ha="center", fontsize=8, color="#374151")
        b += len(idxs)
    ax.set_ylim(0, 1.1)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "decodability_by_feature.png"), dpi=130)

    # Fig 2: emergence curve — flood-fill vs a trivial feature, over training
    flood = [np.mean([results["checkpoints"][g][n]["score"]
                      for n in ("flood_straight", "flood_right", "flood_left")]) for g in ckpts]
    triv = [np.mean([results["checkpoints"][g][FEATURE_NAMES[i]]["score"]
                     for i in TIER["trivial/positional"]]) for g in ckpts]
    danger = [np.mean([results["checkpoints"][g][n]["score"]
                       for n in ("danger_straight", "danger_right", "danger_left")]) for g in ckpts]
    fig, ax = plt.subplots(figsize=(8, 5))
    xs = [max(g, 1) for g in ckpts]
    ax.plot(xs, flood, "o-", label="flood-fill (global structural)", color="#dc2626")
    ax.plot(xs, danger, "s-", label="danger (local lookahead)", color="#7c3aed")
    ax.plot(xs, triv, "^-", label="trivial/positional", color="#9ca3af")
    ax.set_xscale("symlog"); ax.set_xlabel("training games"); ax.set_ylabel("linear decodability (0-1)")
    ax.set_title("When does each structure become linearly decodable?")
    ax.legend(); ax.set_ylim(0, 1.05)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "emergence_curve.png"), dpi=130)
    print(f"Wrote figures to {OUT}/")


if __name__ == "__main__":
    main()
