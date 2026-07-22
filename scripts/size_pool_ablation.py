"""
Isolating ablation (retrain): does the POOLING normalization cause the food-margin dilution?

The uniform GAP*area/36 rescale (size_rescale_ablation.py) made things WORSE — the area-dependence is
channel-heterogeneous, so a single scalar can't fix it. This retrains with a zero-parameter operator
that changes ONLY the avg-pool denominator: masked / count-normalized average — per channel, divide
the sum by the number of ACTIVE cells (post-ReLU > 0) instead of by H*W. That normalizes over the
active region, not the board, so a fixed local pattern keeps its scale as the board grows.

Everything else (conv trunk, max-pool branch, dueling head, hypers, seed) is identical to the avg-pool
baseline. So if the food margin now stays flat across board sizes while the baseline's collapses 5x,
the pooling normalization is isolated as the cause. If it still collapses, pooling is NOT the culprit
and the search moves to conv activation stats / ReLU / the head.

Trains on 6x6, then re-runs the transfer curve and the margin decomposition; compares to baseline.
"""
import os, sys, random, collections, numpy as np, torch, torch.nn as nn
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, "/Users/saheb/home/rl-snake")
from scripts.size_transfer import get_conv_state, DEVICE, greedy_score, SizeAgnosticQNet
from scripts.size_margin import margins as margin_decomp
from envs.snake_game import SnakeGame
from utils.per import PrioritizedReplayBuffer

OUT = "/Users/saheb/home/rl-snake/size_transfer"
SIZES = [6, 8, 10, 12, 16]
CKPT = f"{OUT}/maskedavg_6x6_best.pth"
BASE_CKPT = f"{OUT}/sizeagnostic_6x6_best.pth"


class MaskedAvgPoolQNet(nn.Module):
    """Identical to SizeAgnosticQNet but the avg branch normalizes by ACTIVE-cell count, not H*W."""
    def __init__(self, n_actions=4):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
        )
        self.value_stream = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 1))
        self.advantage_stream = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, n_actions))

    def _pool(self, x):
        h = self.conv(x)                                   # (B,128,H,W), post-ReLU >= 0
        cnt = (h > 0).float().sum(dim=(2, 3)).clamp(min=1.0)   # active cells per channel (no grad)
        masked_avg = h.sum(dim=(2, 3)) / cnt               # mean over the ACTIVE region
        return torch.cat([masked_avg, h.amax(dim=(2, 3))], dim=1)

    def forward(self, x):
        z = self._pool(x)
        v = self.value_stream(z); a = self.advantage_stream(z)
        return v + (a - a.mean(dim=1, keepdim=True))

    def hidden(self, x):
        z = self._pool(x)
        return self.advantage_stream[1](self.advantage_stream[0](z))


def train(board=6, n_games=4000, reward="dense", seed=0):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    GAMMA, LR, BATCH, CAP, LEARN_START, TRAIN_CALLS, TARGET_UPDATE = 0.99, 1e-4, 256, 100_000, 5000, 8, 1000
    EPS_END, EPS_DECAY_GAMES = 0.02, 0.7 * n_games
    max_steps = board * board * 4

    model = MaskedAvgPoolQNet().to(DEVICE)
    target = MaskedAvgPoolQNet().to(DEVICE); target.load_state_dict(model.state_dict()); target.eval()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    mem = PrioritizedReplayBuffer(capacity=CAP, priority_cap=1.0)
    best = -1.0; grad_steps = 0
    scores = collections.deque(maxlen=100)

    for g in range(1, n_games + 1):
        eps = max(EPS_END, 1.0 - g / EPS_DECAY_GAMES)
        game = SnakeGame(board_size=board, reward_mode=reward); game.reset()
        s = get_conv_state(game); info = {"score": 0}
        for _ in range(max_steps):
            if random.random() < eps:
                a = random.randint(0, 3)
            else:
                with torch.no_grad():
                    a = int(torch.argmax(model(torch.tensor(s[None], device=DEVICE))).item())
            _, r, done, info = game.step(a)
            ns = get_conv_state(game)
            mem.add((s, a, r, ns, done)); s = ns
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
            gs, sd = greedy_score(model, board, n=30)
            print(f"  game {g:5d} | eps {eps:.2f} | train {np.mean(scores):.2f} | GREEDY {gs:.2f}±{sd:.1f}", flush=True)
            if gs > best:
                best = gs; torch.save(model.state_dict(), CKPT)
    print(f"  best greedy {best:.2f}", flush=True)


def evaluate():
    masked = MaskedAvgPoolQNet().to(DEVICE)
    masked.load_state_dict(torch.load(CKPT, map_location=DEVICE)); masked.eval()
    base = SizeAgnosticQNet().to(DEVICE)
    base.load_state_dict(torch.load(BASE_CKPT, map_location=DEVICE)); base.eval()

    print(f"\n{'board':>5} || {'greedy base':>11} {'greedy mask':>11} "
          f"| {'food_m base':>11} {'food_m mask':>11} | {'danger base':>11} {'danger mask':>11} "
          f"| {'tw-arg base':>11} {'tw-arg mask':>11}")
    fb, fm, gb, gm = [], [], [], []
    for b in SIZES:
        g_b, _ = greedy_score(base, b, n=40); g_m, _ = greedy_score(masked, b, n=40)
        f_b, d_b, _sp_b, t_b = margin_decomp(base, b)
        f_m, d_m, _sp_m, t_m = margin_decomp(masked, b)
        gb.append(g_b); gm.append(g_m); fb.append(f_b); fm.append(f_m)
        print(f"{b:5d} || {g_b:11.2f} {g_m:11.2f} | {f_b:11.3f} {f_m:11.3f} "
              f"| {d_b:11.3f} {d_m:11.3f} | {t_b*100:10.1f}% {t_m*100:10.1f}%", flush=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    ax1.plot(SIZES, fb, "s-", color="#dc2626", label="avg-pool baseline")
    ax1.plot(SIZES, fm, "o-", color="#16a34a", label="masked-avg (active-cell norm)")
    ax1.axvline(6, ls=":", color="gray", alpha=0.5); ax1.set_ylim(0, None)
    ax1.set_xlabel("board size"); ax1.set_ylabel("food margin (Q units)")
    ax1.set_title("Food margin: does count-normalization flatten it?"); ax1.legend()
    ax2.plot(SIZES, gb, "s-", color="#dc2626", label="avg-pool baseline")
    ax2.plot(SIZES, gm, "o-", color="#16a34a", label="masked-avg")
    ax2.axvline(6, ls=":", color="gray", alpha=0.5)
    ax2.set_xlabel("board size"); ax2.set_ylabel("greedy score")
    ax2.set_title("Zero-shot transfer: does the cliff move?"); ax2.legend()
    fig.tight_layout(); fig.savefig(f"{OUT}/pool_ablation.png", dpi=130)
    print(f"\nWrote {OUT}/pool_ablation.png")


if __name__ == "__main__":
    if "--eval-only" not in sys.argv:
        print("== Training MaskedAvgPoolQNet on 6x6 ==", flush=True)
        train()
    evaluate()
