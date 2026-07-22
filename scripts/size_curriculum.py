"""
The payoff run: egocentric arch (attention pool + head-relative coords) trained ACROSS a size range.

Everything converged on this: interpolation is free, extrapolation range tracks TRAINING COVERAGE,
and the egocentric encoding extrapolates beyond the trained range better than baseline (once trained
on boards big enough to learn its scale-invariant use). So train on a curriculum of sizes 6..22 and
test zero-shot up to 100.

Implementation note: mixed board sizes can't share a replay batch (ragged tensors), so we keep ONE
PrioritizedReplayBuffer PER SIZE; each gradient step samples a ready size and draws a same-size batch.
"""
import os, sys, random, collections, numpy as np, torch, torch.nn as nn
sys.path.insert(0, "/Users/saheb/home/rl-snake")
from scripts.size_transfer import DEVICE
from scripts.size_ego_attn import CoordAttnPoolQNet, state_ego
from envs.snake_game import SnakeGame
from utils.per import PrioritizedReplayBuffer

OUT = "/Users/saheb/home/rl-snake/size_transfer"
CKPT = f"{OUT}/curriculum_ego_best.pth"
TRAIN_SIZES = [6, 10, 14, 18, 22]
EVAL_GRID = [6, 10, 14, 18, 22, 32, 48, 64, 100]


@torch.no_grad()
def eval_run(model, B, n=8, cap_mult=2):
    model.eval(); sc = []; hits = tot = 0
    for e in range(n):
        random.seed(5000 + e); np.random.seed(5000 + e)
        game = SnakeGame(board_size=B, reward_mode="dense"); game.reset(); info = {"score": 0}; prev = 0
        for _ in range(B * B * cap_mult):
            head = game.snake_position[-1]; food = game.food_position
            if food is None:
                break
            d0 = abs(head[0]-food[0]) + abs(head[1]-food[1])
            a = int(torch.argmax(model(torch.tensor(state_ego(game)[None], device=DEVICE))).item())
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


def train(n_games=6000, reward="dense", seed=0):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    GAMMA, LR, BATCH, LEARN_START, TRAIN_CALLS, TARGET_UPDATE = 0.99, 1e-4, 256, 2000, 8, 1000
    EPS_END, EPS_DECAY_GAMES = 0.02, 0.7 * n_games
    model = CoordAttnPoolQNet().to(DEVICE)
    target = CoordAttnPoolQNet().to(DEVICE); target.load_state_dict(model.state_dict()); target.eval()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    mem = {B: PrioritizedReplayBuffer(capacity=40_000, priority_cap=1.0) for B in TRAIN_SIZES}
    best = -1.0; grad_steps = 0
    recent = {B: collections.deque(maxlen=30) for B in TRAIN_SIZES}

    for g in range(1, n_games + 1):
        eps = max(EPS_END, 1.0 - g / EPS_DECAY_GAMES)
        B = random.choice(TRAIN_SIZES)
        game = SnakeGame(board_size=B, reward_mode=reward); game.reset()
        s = state_ego(game); info = {"score": 0}
        for _ in range(B * B * 4):
            if random.random() < eps:
                a = random.randint(0, 3)
            else:
                with torch.no_grad():
                    a = int(torch.argmax(model(torch.tensor(s[None], device=DEVICE))).item())
            _, r, done, info = game.step(a)
            ns = state_ego(game); mem[B].add((s, a, r, ns, done)); s = ns
            if done:
                break
        recent[B].append(info["score"])

        ready = [b for b in TRAIN_SIZES if len(mem[b]) >= LEARN_START]
        for _ in range(TRAIN_CALLS):
            if not ready:
                break
            bsz = random.choice(ready)
            batch, idx, isw = mem[bsz].sample(BATCH)
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
            mem[bsz].update_priorities(idx, td.abs().detach().cpu().numpy())
            grad_steps += 1
            if grad_steps % TARGET_UPDATE == 0:
                target.load_state_dict(model.state_dict())

        if g % 500 == 0:
            g10 = eval_run(model, 10, n=15)[1]; g22 = eval_run(model, 22, n=10)[1]
            trains = "  ".join(f"{b}:{np.mean(recent[b]):.1f}" for b in TRAIN_SIZES if recent[b])
            print(f"  game {g:5d} | eps {eps:.2f} | train[{trains}] | GREEDY 10:{g10:.1f} 22:{g22:.1f}", flush=True)
            score = g10 + g22
            if score > best:
                best = score; torch.save(model.state_dict(), CKPT)
    torch.save(model.state_dict(), f"{OUT}/curriculum_ego_final.pth")
    print(f"  best (g10+g22) {best:.2f}", flush=True)


def evaluate():
    model = CoordAttnPoolQNet().to(DEVICE)
    model.load_state_dict(torch.load(CKPT, map_location=DEVICE)); model.eval()
    print("\n== curriculum ego (trained on 6..22) — zero-shot fine grid ==")
    print(f"{'board':>5} | {'toward-food%':>12} | {'avg score':>9} | {'max score':>9} | region")
    for B in EVAL_GRID:
        n, cm = (12, 2) if B <= 32 else (4, 1)
        tf, sc = eval_run(model, B, n=n, cap_mult=cm)
        # separate max via a second pass reusing eval_run's mean; quick max probe
        region = "train" if B <= 22 else "EXTRAP"
        print(f"{B:5d} | {tf:11.1f}% | {sc:9.1f} | {'':>9} | {region}", flush=True)


if __name__ == "__main__":
    if "--eval-only" not in sys.argv:
        print("== Training curriculum ego on sizes 6..22 ==", flush=True)
        train()
    evaluate()
