"""
Does a *fully-convolutional* Snake agent zero-shot transfer to bigger boards?

The repo's ConvDuelingQNet flattens conv features into a board-size-locked FC head, so it
literally cannot run on a different board. Here we make the head size-agnostic (global
avg-pool ‖ max-pool → FC), train on 6x6, and measure zero-shot greedy score on 6/8/10/12/16.
Convolution is translation-invariant, so the *intuition* says it should transfer. Does it?

Stability: hard periodic target updates (not soft-every-step) to avoid the Q-drift collapse
seen earlier. Reward configurable; best checkpoint kept by greedy score.
"""
import os, sys, random, collections, argparse, numpy as np, torch, torch.nn as nn
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from envs.snake_game import SnakeGame
from utils.per import PrioritizedReplayBuffer

DEVICE = torch.device("cuda" if torch.cuda.is_available()
                      else "mps" if torch.backends.mps.is_available() else "cpu")
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "size_transfer")
os.makedirs(OUT, exist_ok=True)


def get_conv_state(game):
    H = W = game.board_size
    s = np.zeros((3, H, W), dtype=np.float32)
    snake = list(game.snake_position)
    for p in snake[:-1]:
        s[0, p[0], p[1]] = 1.0
    s[1, snake[-1][0], snake[-1][1]] = 1.0
    if game.food_position:
        s[2, game.food_position[0], game.food_position[1]] = 1.0
    return s


class SizeAgnosticQNet(nn.Module):
    """Conv trunk -> global avg+max pool -> dueling head. Runs on ANY board size."""
    def __init__(self, n_actions=4):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
        )
        self.value_stream = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 1))
        self.advantage_stream = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, n_actions))

    def _pool(self, x):
        h = self.conv(x)
        return torch.cat([h.mean(dim=(2, 3)), h.amax(dim=(2, 3))], dim=1)  # (B, 256)

    def forward(self, x):
        z = self._pool(x)
        v = self.value_stream(z); a = self.advantage_stream(z)
        return v + (a - a.mean(dim=1, keepdim=True))

    def hidden(self, x):
        z = self._pool(x)
        return self.advantage_stream[1](self.advantage_stream[0](z))


@torch.no_grad()
def greedy_score(model, board, n=30, cap_mult=4):
    model.eval(); tot = []
    for _ in range(n):
        game = SnakeGame(board_size=board, reward_mode="dense"); game.reset()
        info = {"score": 0}
        for _ in range(board * board * cap_mult):
            s = get_conv_state(game)
            a = int(torch.argmax(model(torch.tensor(s[None], device=DEVICE))).item())
            _, _, done, info = game.step(a)
            if done:
                break
        tot.append(info["score"])
    return float(np.mean(tot)), float(np.std(tot))


def train(board=6, n_games=4000, reward="dense", seed=0):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    GAMMA, LR, BATCH, CAP, LEARN_START, TRAIN_CALLS, TARGET_UPDATE = 0.99, 1e-4, 256, 100_000, 5000, 8, 1000
    EPS_END, EPS_DECAY_GAMES = 0.02, 0.7 * n_games
    max_steps = board * board * 4

    model = SizeAgnosticQNet().to(DEVICE)
    target = SizeAgnosticQNet().to(DEVICE); target.load_state_dict(model.state_dict()); target.eval()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    mem = PrioritizedReplayBuffer(capacity=CAP, priority_cap=1.0)
    total_steps = 0; best = -1.0; grad_steps = 0
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
            mem.add((s, a, r, ns, done)); s = ns; total_steps += 1
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
                if grad_steps % TARGET_UPDATE == 0:      # hard periodic target sync (stable)
                    target.load_state_dict(model.state_dict())

        if g % 500 == 0:
            gs, sd = greedy_score(model, board, n=30)
            print(f"  game {g:5d} | eps {eps:.2f} | train {np.mean(scores):.2f} | GREEDY {gs:.2f}±{sd:.1f}", flush=True)
            if gs > best:
                best = gs; torch.save(model.state_dict(), f"{OUT}/sizeagnostic_{board}x{board}_best.pth")
    torch.save(model.state_dict(), f"{OUT}/sizeagnostic_{board}x{board}_final.pth")
    print(f"  best greedy {best:.2f}", flush=True)
    return model


def transfer_curve(ckpt, sizes=(6, 8, 10, 12, 16), n=50):
    model = SizeAgnosticQNet().to(DEVICE)
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE)); model.eval()
    print(f"\n== Zero-shot transfer ({ckpt}) ==")
    means, stds = [], []
    for b in sizes:
        m, sd = greedy_score(model, b, n=n)
        means.append(m); stds.append(sd)
        print(f"  board {b:2d}x{b:<2d} | greedy {m:5.2f} ± {sd:.2f} | score/cells {m/(b*b):.3f}")
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar(sizes, means, yerr=stds, fmt="o-", capsize=4, color="#2563eb")
    ax.axvline(6, ls="--", color="gray", alpha=0.6, label="trained size (6x6)")
    ax.set_xlabel("board size (zero-shot)"); ax.set_ylabel("greedy score (food eaten)")
    ax.set_title("Does a size-agnostic Snake ConvNet transfer to bigger boards?")
    ax.legend(); fig.tight_layout(); fig.savefig(f"{OUT}/transfer_curve.png", dpi=130)
    print(f"Wrote {OUT}/transfer_curve.png")
    return means, stds


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--board", type=int, default=6)
    ap.add_argument("--games", type=int, default=4000)
    ap.add_argument("--reward", default="dense", choices=["dense", "sparse", "pure_sparse"])
    ap.add_argument("--skip-train", action="store_true")
    args = ap.parse_args()
    ckpt = f"{OUT}/sizeagnostic_{args.board}x{args.board}_best.pth"
    if not args.skip_train:
        print(f"== Training size-agnostic ConvNet on {args.board}x{args.board} (reward={args.reward}) ==", flush=True)
        train(board=args.board, n_games=args.games, reward=args.reward)
    transfer_curve(ckpt)
