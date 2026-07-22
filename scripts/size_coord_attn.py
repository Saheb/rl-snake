"""
Iteration: add normalized coordinate channels to the attention-pool agent. ONE variable — no
multi-size curriculum (train on 6x6 only), same everything else as AttnPoolQNet.

Why: attention pooling fixed barrier #1 (background flooding) but not barrier #2 (long-range
direction) — it pools translation-invariant features that carry no absolute position, so it can't
steer to distant food. Coordinate channels [row/(H-1), col/(W-1)] give every cell its normalized
position (values in [0,1] at ANY board size — in-distribution at 100x100). Attention selecting the
food cell then recovers WHERE food is; sign(food-head) is scale-invariant, which is what argmax needs.

Test: is coord ALONE (trained on 6x6, no curriculum) enough? Compare zero-shot vs the attention-pool
model on transfer score, toward-food%, and a FAR-food canonical margin (food distance grows with board).
"""
import os, sys, random, collections, numpy as np, torch, torch.nn as nn
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, "/Users/saheb/home/rl-snake")
from scripts.size_transfer import get_conv_state, DEVICE
from scripts.size_attn_pool import AttnPoolQNet
from envs.snake_game import SnakeGame
from utils.per import PrioritizedReplayBuffer

OUT = "/Users/saheb/home/rl-snake/size_transfer"
CKPT = f"{OUT}/coordattn_6x6_best.pth"
ATTN_CKPT = f"{OUT}/attnpool_6x6_best.pth"
DELTAS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
TRANSFER = [6, 8, 10, 12, 16, 24, 32]
BIG = [48, 64, 100]


def add_coords(s3, B):
    r = np.broadcast_to((np.arange(B) / max(B - 1, 1))[:, None], (B, B)).astype(np.float32)
    c = np.broadcast_to((np.arange(B) / max(B - 1, 1))[None, :], (B, B)).astype(np.float32)
    return np.concatenate([s3, r[None], c[None]], axis=0)


def state5(game):
    return add_coords(get_conv_state(game), game.board_size)


class CoordAttnPoolQNet(AttnPoolQNet):
    """AttnPoolQNet but the conv trunk takes 5 channels (body, head, food, row_norm, col_norm)."""
    def __init__(self, n_actions=4, n_heads=4):
        super().__init__(n_actions=n_actions, n_heads=n_heads)
        self.conv = nn.Sequential(
            nn.Conv2d(5, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
        )


@torch.no_grad()
def greedy_score(model, board, state_fn, n, cap_mult=4):
    model.eval(); tot = []
    for _ in range(n):
        game = SnakeGame(board_size=board, reward_mode="dense"); game.reset()
        info = {"score": 0}
        for _ in range(board * board * cap_mult):
            a = int(torch.argmax(model(torch.tensor(state_fn(game)[None], device=DEVICE))).item())
            _, _, done, info = game.step(a)
            if done:
                break
        tot.append(info["score"])
    return float(np.mean(tot))


@torch.no_grad()
def toward_food(model, board, state_fn, max_steps=6000):
    hits = tot = 0
    while tot < max_steps:
        game = SnakeGame(board_size=board, reward_mode="dense"); game.reset()
        prev = 0
        for _ in range(board * board * 4):
            head = game.snake_position[-1]; food = game.food_position
            if food is None:
                break
            d0 = abs(head[0] - food[0]) + abs(head[1] - food[1])
            a = int(torch.argmax(model(torch.tensor(state_fn(game)[None], device=DEVICE))).item())
            _, _, done, info = game.step(a); sc = info.get("score", prev)
            if sc > prev:
                hits += 1; tot += 1; prev = sc
            elif not done:
                nh = game.snake_position[-1]
                tot += 1
                if abs(nh[0] - food[0]) + abs(nh[1] - food[1]) < d0:
                    hits += 1
            else:
                tot += 1
            if done or tot >= max_steps:
                break
    return hits / max(tot, 1)


def scene(B, d, anchor="corner"):
    """head + short snake, food d cells to the right (same row)."""
    s = np.zeros((3, B, B), np.float32)
    hr, hc = (2, 2) if anchor == "corner" else (B // 2, 2)
    s[0, hr, hc - 2] = 1.0; s[0, hr, hc - 1] = 1.0
    s[1, hr, hc] = 1.0; s[2, hr, hc + d] = 1.0
    return s, (hr, hc), (hr, hc + d)


@torch.no_grad()
def canonical(model, B, d, state_fn):
    s3, head, food = scene(B, d)
    s = add_coords(s3, B) if state_fn is state5 else s3
    q = model(torch.tensor(s[None], device=DEVICE)).cpu().numpy().ravel()
    d0 = abs(head[0]-food[0]) + abs(head[1]-food[1])
    toward = np.array([abs(head[0]+dx-food[0]) + abs(head[1]+dy-food[1]) < d0 for dx, dy in DELTAS])
    return float(q[toward].max() - q[~toward].max()), bool(toward[int(q.argmax())])


def train(board=6, n_games=4000, reward="dense", seed=0):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    GAMMA, LR, BATCH, LEARN_START, TRAIN_CALLS, TARGET_UPDATE = 0.99, 1e-4, 256, 5000, 8, 1000
    EPS_END, EPS_DECAY_GAMES = 0.02, 0.7 * n_games
    max_steps = board * board * 4
    model = CoordAttnPoolQNet().to(DEVICE)
    target = CoordAttnPoolQNet().to(DEVICE); target.load_state_dict(model.state_dict()); target.eval()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    mem = PrioritizedReplayBuffer(capacity=100_000, priority_cap=1.0)
    best = -1.0; grad_steps = 0; scores = collections.deque(maxlen=100)
    for g in range(1, n_games + 1):
        eps = max(EPS_END, 1.0 - g / EPS_DECAY_GAMES)
        game = SnakeGame(board_size=board, reward_mode=reward); game.reset()
        s = state5(game); info = {"score": 0}
        for _ in range(max_steps):
            if random.random() < eps:
                a = random.randint(0, 3)
            else:
                with torch.no_grad():
                    a = int(torch.argmax(model(torch.tensor(s[None], device=DEVICE))).item())
            _, r, done, info = game.step(a)
            ns = state5(game); mem.add((s, a, r, ns, done)); s = ns
            if done:
                break
        scores.append(info["score"])
        if len(mem) >= LEARN_START:
            for _ in range(TRAIN_CALLS):
                batch, idx, isw = mem.sample(BATCH)
                if len(batch) < 2:
                    continue
                isw = torch.tensor(np.asarray(isw)[:len(batch)], device=DEVICE)
                bs, ba, br, bns, bd = zip(*batch)
                bs = torch.tensor(np.array(bs), device=DEVICE); bns = torch.tensor(np.array(bns), device=DEVICE)
                ba = torch.tensor(ba, device=DEVICE, dtype=torch.long)
                br = torch.tensor(br, device=DEVICE, dtype=torch.float)
                bd = torch.tensor(bd, device=DEVICE, dtype=torch.bool)
                q = model(bs).gather(1, ba[:, None]).squeeze(1)
                with torch.no_grad():
                    na = torch.argmax(model(bns), dim=1)
                    nq = target(bns).gather(1, na[:, None]).squeeze(1)
                    tgt = br + GAMMA * nq * (~bd)
                td = q - tgt
                loss = (isw * td.pow(2)).mean()
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 10.0); opt.step()
                mem.update_priorities(idx, td.abs().detach().cpu().numpy())
                grad_steps += 1
                if grad_steps % TARGET_UPDATE == 0:
                    target.load_state_dict(model.state_dict())
        if g % 500 == 0:
            gs = greedy_score(model, board, state5, n=30)
            print(f"  game {g:5d} | eps {eps:.2f} | train {np.mean(scores):.2f} | GREEDY {gs:.2f}", flush=True)
            if gs > best:
                best = gs; torch.save(model.state_dict(), CKPT)
    print(f"  best greedy {best:.2f}", flush=True)


def evaluate():
    coord = CoordAttnPoolQNet().to(DEVICE); coord.load_state_dict(torch.load(CKPT, map_location=DEVICE)); coord.eval()
    attn = AttnPoolQNet().to(DEVICE); attn.load_state_dict(torch.load(ATTN_CKPT, map_location=DEVICE)); attn.eval()

    print("\n== zero-shot transfer: score/cells | toward-food% (attn 3ch vs coord 5ch) ==")
    print(f"{'board':>5} || {'attn s/c':>9} {'attn tf%':>8} | {'coord s/c':>9} {'coord tf%':>9} "
          f"|| {'attn farcanon':>13} {'coord farcanon':>14}")
    ac, cc = [], []
    for b in TRANSFER:
        n, cm = (20, 4) if b <= 16 else (8, 2)
        a_sc = greedy_score(attn, b, get_conv_state, n=n, cap_mult=cm) / (b*b)
        c_sc = greedy_score(coord, b, state5, n=n, cap_mult=cm) / (b*b)
        a_tf = toward_food(attn, b, get_conv_state); c_tf = toward_food(coord, b, state5)
        dfar = max(1, b - 5)          # far food: distance grows with board
        a_fc, _ = canonical(attn, b, dfar, get_conv_state); c_fc, _ = canonical(coord, b, dfar, state5)
        ac.append(c_sc); cc.append(c_tf)
        print(f"{b:5d} || {a_sc:9.3f} {a_tf*100:7.1f}% | {c_sc:9.3f} {c_tf*100:8.1f}% "
              f"|| {a_fc:13.3f} {c_fc:14.3f}", flush=True)

    print("\n== big-board probe: toward-food% + far-food canonical margin (coord) + greedy ==")
    for b in BIG:
        c_tf = toward_food(coord, b, state5, max_steps=4000)
        dfar = b - 5
        c_fc, arg = canonical(coord, b, dfar, state5)
        c_sc = greedy_score(coord, b, state5, n=2, cap_mult=1)
        print(f"{b:5d} || coord toward-food {c_tf*100:5.1f}% | far-canon margin {c_fc:7.3f} "
              f"argmax_toward={arg} | greedy {c_sc:6.2f}", flush=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(TRANSFER, cc, "o-", color="#16a34a", label="coord+attn toward-food%")
    ax.axhline(0.5, ls=":", color="#9ca3af", lw=1, label="chance")
    ax.axvline(6, ls=":", color="gray", alpha=0.5)
    ax.set_xlabel("board size (zero-shot)"); ax.set_ylabel("toward-food fraction")
    ax.set_title("Coord+attention (trained 6x6, no curriculum): does direction hold at scale?")
    ax.legend(); ax.set_ylim(0.4, 1.02)
    fig.tight_layout(); fig.savefig(f"{OUT}/coord_attn.png", dpi=130)
    print(f"\nWrote {OUT}/coord_attn.png")


if __name__ == "__main__":
    if "--eval-only" not in sys.argv:
        print("== Training CoordAttnPoolQNet on 6x6 (no curriculum) ==", flush=True)
        train()
    evaluate()
