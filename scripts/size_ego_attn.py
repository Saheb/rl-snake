"""
Egocentric encoding: head-RELATIVE squashed-offset coord channels (not absolute, not board-normalized).

Root-cause fix for barrier #2 (long-range direction). Instead of absolute normalized coords
[row/(H-1), col/(W-1)], each cell carries its offset from the head, in RAW cells, squashed:
    rel_r[i,j] = tanh((i - head_row)/kappa),   rel_c[i,j] = tanh((j - head_col)/kappa)
So "food 3 cells to the right" is tanh(3/kappa) on 6x6 AND on 100x100 — identical, board size never
enters. The small-offset regime that big boards live in is already sampled at 6x6 training; large
offsets saturate to +/-1 (same "far in that direction"). Direction/sign is IN the representation, not
something the net must reconstruct from two absolute readings. Attention pool is kept (fixes barrier
#1, background flooding). Architecture is identical to CoordAttnPoolQNet — ONLY the coord channels'
content changes.

Test: train on 6x6 ONLY (no curriculum). Does it hold direction at 32/64/100? Compare against the
absolute-coord model on an UNCONFOUNDED far-food probe (food mid-board, away from any edge).
"""
import os, sys, random, collections, numpy as np, torch, torch.nn as nn
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, "/Users/saheb/home/rl-snake")
from scripts.size_transfer import get_conv_state, DEVICE
from scripts.size_coord_attn import CoordAttnPoolQNet, greedy_score, toward_food, add_coords, DELTAS
from envs.snake_game import SnakeGame
from utils.per import PrioritizedReplayBuffer

OUT = "/Users/saheb/home/rl-snake/size_transfer"
CKPT = f"{OUT}/egoattn_6x6_best.pth"
ABS_CKPT = f"{OUT}/coordattn_6x6_best.pth"
KAPPA = 3.0
TRANSFER = [6, 8, 10, 12, 16, 24, 32]
BIG = [48, 64, 100]


def add_ego_coords(s3, head, B, kappa=KAPPA):
    rel_r = np.tanh((np.arange(B) - head[0]) / kappa)[:, None] * np.ones((1, B), np.float32)
    rel_c = np.tanh((np.arange(B) - head[1]) / kappa)[None, :] * np.ones((B, 1), np.float32)
    return np.concatenate([s3, rel_r[None].astype(np.float32), rel_c[None].astype(np.float32)], axis=0)


def state_ego(game):
    return add_ego_coords(get_conv_state(game), game.snake_position[-1], game.board_size)


def scene_centered(B, d):
    """Head + short snake near the CENTER, food d cells to the right — food stays interior (no edge cue)."""
    hr = B // 2; hc = B // 2 - d // 2; fc = hc + d
    s = np.zeros((3, B, B), np.float32)
    s[0, hr, hc - 2] = 1.0; s[0, hr, hc - 1] = 1.0
    s[1, hr, hc] = 1.0; s[2, hr, fc] = 1.0
    return s, (hr, hc), (hr, fc)


def margin_from_q(q, head, food):
    d0 = abs(head[0]-food[0]) + abs(head[1]-food[1])
    toward = np.array([abs(head[0]+dx-food[0]) + abs(head[1]+dy-food[1]) < d0 for dx, dy in DELTAS])
    return float(q[toward].max() - q[~toward].max()), bool(toward[int(q.argmax())])


@torch.no_grad()
def far_probe(model, B, d, kind):
    s3, head, food = scene_centered(B, d)
    s = add_ego_coords(s3, head, B) if kind == "ego" else add_coords(s3, B)
    q = model(torch.tensor(s[None], device=DEVICE)).cpu().numpy().ravel()
    return margin_from_q(q, head, food)


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
        s = state_ego(game); info = {"score": 0}
        for _ in range(max_steps):
            if random.random() < eps:
                a = random.randint(0, 3)
            else:
                with torch.no_grad():
                    a = int(torch.argmax(model(torch.tensor(s[None], device=DEVICE))).item())
            _, r, done, info = game.step(a)
            ns = state_ego(game); mem.add((s, a, r, ns, done)); s = ns
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
            gs = greedy_score(model, board, state_ego, n=30)
            print(f"  game {g:5d} | eps {eps:.2f} | train {np.mean(scores):.2f} | GREEDY {gs:.2f}", flush=True)
            if gs > best:
                best = gs; torch.save(model.state_dict(), CKPT)
    print(f"  best greedy {best:.2f}", flush=True)


def evaluate():
    ego = CoordAttnPoolQNet().to(DEVICE); ego.load_state_dict(torch.load(CKPT, map_location=DEVICE)); ego.eval()
    have_abs = os.path.exists(ABS_CKPT)
    if have_abs:
        absm = CoordAttnPoolQNet().to(DEVICE); absm.load_state_dict(torch.load(ABS_CKPT, map_location=DEVICE)); absm.eval()

    print("\n== ego (relative coords) zero-shot transfer + far-food direction (food mid-board) ==")
    print(f"{'board':>5} || {'ego s/c':>8} {'ego tf%':>8} | {'ego farM':>9} {'ego arg':>7} "
          f"|| {'abs farM':>9} {'abs arg':>7}")
    tf_curve, far_curve = [], []
    for b in TRANSFER:
        n, cm = (20, 4) if b <= 16 else (8, 2)
        sc = greedy_score(ego, b, state_ego, n=n, cap_mult=cm) / (b * b)
        tf = toward_food(ego, b, state_ego)
        d = max(2, b // 2)
        e_fm, e_arg = far_probe(ego, b, d, "ego")
        a_fm, a_arg = far_probe(absm, b, d, "abs") if have_abs else (float("nan"), None)
        tf_curve.append(tf); far_curve.append(e_fm)
        print(f"{b:5d} || {sc:8.3f} {tf*100:7.1f}% | {e_fm:9.3f} {str(e_arg):>7} "
              f"|| {a_fm:9.3f} {str(a_arg):>7}", flush=True)

    print("\n== ego big-board probe: toward-food% | far-food margin | greedy ==")
    for b in BIG:
        tf = toward_food(ego, b, state_ego, max_steps=4000)
        e_fm, e_arg = far_probe(ego, b, b // 2, "ego")
        sc = greedy_score(ego, b, state_ego, n=2, cap_mult=1)
        print(f"{b:5d} || toward-food {tf*100:5.1f}% | far-margin {e_fm:7.3f} argmax_toward={e_arg} "
              f"| greedy {sc:6.2f}", flush=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    ax1.plot(TRANSFER, tf_curve, "o-", color="#16a34a", label="ego (relative coords)")
    ax1.axhline(0.5, ls=":", color="#9ca3af", lw=1, label="chance")
    ax1.axvline(6, ls=":", color="gray", alpha=0.5)
    ax1.set_xlabel("board size (zero-shot)"); ax1.set_ylabel("toward-food fraction")
    ax1.set_title("Egocentric encoding (6x6, no curriculum): toward-food% vs scale"); ax1.legend(); ax1.set_ylim(0.4, 1.02)
    ax2.plot(TRANSFER, far_curve, "o-", color="#16a34a", label="ego far-food margin")
    ax2.axhline(0, ls=":", color="#9ca3af", lw=1)
    ax2.axvline(6, ls=":", color="gray", alpha=0.5)
    ax2.set_xlabel("board size"); ax2.set_ylabel("far-food margin (Q units)")
    ax2.set_title("Direction to distant food (unconfounded, mid-board)"); ax2.legend()
    fig.tight_layout(); fig.savefig(f"{OUT}/ego_attn.png", dpi=130)
    print(f"\nWrote {OUT}/ego_attn.png")


if __name__ == "__main__":
    if "--eval-only" not in sys.argv:
        print("== Training egocentric attention agent on 6x6 (no curriculum) ==", flush=True)
        train()
    evaluate()
