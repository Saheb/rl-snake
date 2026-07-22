"""
Interpolation test: does the attention/egocentric machinery hurt performance WITHIN the trained
regime, or only extrapolation?

The extrapolation table (all trained on 6x6) showed ego worse on bigger boards. But that conflates
"worse architecture" with "worse extrapolator". Interpolation isolates it: train on a LARGER board
(10x10) and test at and BELOW it. Baseline avg-pool trained on 10x10 already exists on disk
(sizeagnostic_10x10_best.pth) and interpolates flawlessly below 10 (per FINDINGS). So we only train
EGO on 10x10 and compare the fine grid.

  boards <=10 : interpolation — does ego hold like baseline, or degrade within the trained regime?
  boards >10  : extrapolation — reference.
"""
import os, sys, random, collections, numpy as np, torch, torch.nn as nn
sys.path.insert(0, "/Users/saheb/home/rl-snake")
from scripts.size_transfer import SizeAgnosticQNet, get_conv_state, DEVICE
from scripts.size_ego_attn import CoordAttnPoolQNet, state_ego
from envs.snake_game import SnakeGame
from utils.per import PrioritizedReplayBuffer

OUT = "/Users/saheb/home/rl-snake/size_transfer"
EGO_CKPT = f"{OUT}/egoattn_10x10_best.pth"
BASE_CKPT = f"{OUT}/sizeagnostic_10x10_best.pth"
GRID = [6, 8, 10, 11, 12, 14, 16]


def train_ego(board=10, n_games=4000, reward="dense", seed=0):
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
            gs = _score(model, state_ego, board, n=30)[1]
            print(f"  game {g:5d} | eps {eps:.2f} | train {np.mean(scores):.2f} | GREEDY {gs:.2f}", flush=True)
            if gs > best:
                best = gs; torch.save(model.state_dict(), EGO_CKPT)
    print(f"  best greedy {best:.2f}", flush=True)


@torch.no_grad()
def _score(model, sf, B, n=10, cap_mult=3):
    sc = []; hits = tot = 0
    for e in range(n):
        random.seed(1000 + e); np.random.seed(1000 + e)
        game = SnakeGame(board_size=B, reward_mode="dense"); game.reset(); info = {"score": 0}; prev = 0
        for _ in range(B * B * cap_mult):
            head = game.snake_position[-1]; food = game.food_position
            if food is None:
                break
            d0 = abs(head[0]-food[0]) + abs(head[1]-food[1])
            a = int(torch.argmax(model(torch.tensor(sf(game)[None], device=DEVICE))).item())
            _, _, done, info = game.step(a); s = info.get("score", prev)
            if s > prev:
                hits += 1; tot += 1; prev = s
            elif not done:
                nh = game.snake_position[-1]; tot += 1
                if abs(nh[0]-food[0]) + abs(nh[1]-food[1]) < d0:
                    hits += 1
            else:
                tot += 1
            if done:
                break
        sc.append(info["score"])
    return hits / max(tot, 1) * 100, float(np.mean(sc))


def evaluate():
    ego = CoordAttnPoolQNet().to(DEVICE); ego.load_state_dict(torch.load(EGO_CKPT, map_location=DEVICE)); ego.eval()
    base = SizeAgnosticQNet().to(DEVICE); base.load_state_dict(torch.load(BASE_CKPT, map_location=DEVICE)); base.eval()
    print("\n== both trained on 10x10 | interpolation (<=10) vs extrapolation (>10) ==")
    print(f"{'board':>5} | {'baseline tf% (score)':>22} | {'ego tf% (score)':>18} | region")
    for B in GRID:
        bt, bs = _score(base, get_conv_state, B)
        et, es = _score(ego, state_ego, B)
        region = "INTERP" if B <= 10 else "extrap"
        print(f"{B:5d} | {bt:6.0f}% ({bs:6.1f})        | {et:6.0f}% ({es:6.1f})   | {region}", flush=True)


if __name__ == "__main__":
    if "--eval-only" not in sys.argv:
        print("== Training EGO on 10x10 for interpolation test ==", flush=True)
        train_ego()
    evaluate()
