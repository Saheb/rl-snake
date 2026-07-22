"""
Deploy the validated fix: attention pooling (background-suppressing) instead of global avg-pool.

Mechanism (isolated in size_canonical_scene.py): global avg-pool floods the pooled vector with the
empty-cell background constant as the board grows, diluting the food signal. Count-normalization
failed because empty cells carry positive post-ReLU bias (they "count as active"). Attention pooling
is the learned version of the local-box oracle: a softmax over cells can put ~0 weight on empty cells
regardless of their activation, and softmax is area-invariant (normalizes over H*W at any size).

Minimal change from SizeAgnosticQNet: the avg branch -> multi-head attention pool; max branch kept;
dueling head identical (256-d input). Train on 6x6 (same hypers as baseline), then compare zero-shot
transfer, food margin, and canonical-scene food margin against the avg-pool baseline. This tests
whether the fix moves the cliff — before adding coord channels / curriculum for the 100x100 agent.
"""
import os, sys, random, collections, numpy as np, torch, torch.nn as nn
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, "/Users/saheb/home/rl-snake")
from scripts.size_transfer import get_conv_state, DEVICE, SizeAgnosticQNet
from scripts.size_margin import margins as margin_decomp
from scripts.size_canonical_scene import make_scene, food_margin
from envs.snake_game import SnakeGame
from utils.per import PrioritizedReplayBuffer

OUT = "/Users/saheb/home/rl-snake/size_transfer"
CKPT = f"{OUT}/attnpool_6x6_best.pth"
BASE_CKPT = f"{OUT}/sizeagnostic_6x6_best.pth"
TRANSFER_SIZES = [6, 8, 10, 12, 16, 24, 32]
MARGIN_SIZES = [6, 8, 10, 12, 16]


class AttnPoolQNet(nn.Module):
    """avg branch -> multi-head attention pool (area-invariant, can zero out empty cells)."""
    def __init__(self, n_actions=4, n_heads=4):
        super().__init__()
        self.n_heads = n_heads
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
        )
        self.attn = nn.Conv2d(128, n_heads, 1)                       # per-head spatial logits
        self.proj = nn.Sequential(nn.Linear(128 * n_heads, 128), nn.ReLU())
        self.value_stream = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 1))
        self.advantage_stream = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, n_actions))

    def _pool(self, x):
        h = self.conv(x)
        B, C, H, W = h.shape
        w = torch.softmax(self.attn(h).view(B, self.n_heads, H * W), dim=2)   # attention over cells
        pooled = torch.einsum("bnk,bck->bnc", w, h.view(B, C, H * W))         # (B, heads, 128)
        attn_vec = self.proj(pooled.reshape(B, self.n_heads * C))            # (B, 128)
        return torch.cat([attn_vec, h.amax(dim=(2, 3))], dim=1)              # (B, 256)

    def forward(self, x):
        z = self._pool(x); v = self.value_stream(z); a = self.advantage_stream(z)
        return v + (a - a.mean(dim=1, keepdim=True))

    def hidden(self, x):
        z = self._pool(x)
        return self.advantage_stream[1](self.advantage_stream[0](z))


@torch.no_grad()
def greedy_score(model, board, n, cap_mult=4):
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
    model = AttnPoolQNet().to(DEVICE)
    target = AttnPoolQNet().to(DEVICE); target.load_state_dict(model.state_dict()); target.eval()
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
            ns = get_conv_state(game); mem.add((s, a, r, ns, done)); s = ns
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


@torch.no_grad()
def canonical_curve(model, sizes, anchor="corner"):
    out = []
    for B in sizes:
        hc = 2 if anchor == "corner" else B // 2
        if hc + 3 >= B or hc - 2 < 0:
            out.append(np.nan); continue
        s, head, food, _ = make_scene(B, anchor)
        q = model(torch.tensor(s[None], device=DEVICE)).cpu().numpy().ravel()
        m, _ = food_margin(q, head, food)
        out.append(m)
    return out


def evaluate():
    attn = AttnPoolQNet().to(DEVICE); attn.load_state_dict(torch.load(CKPT, map_location=DEVICE)); attn.eval()
    base = SizeAgnosticQNet().to(DEVICE); base.load_state_dict(torch.load(BASE_CKPT, map_location=DEVICE)); base.eval()

    print("\n== zero-shot transfer (greedy score / cells) ==")
    print(f"{'board':>5} || {'base score':>10} {'base /cells':>11} | {'attn score':>10} {'attn /cells':>11}")
    gb, ga = [], []
    for b in TRANSFER_SIZES:
        n, cm = (30, 4) if b <= 16 else (12, 2)
        g_b, _ = greedy_score(base, b, n=n, cap_mult=cm)
        g_a, _ = greedy_score(attn, b, n=n, cap_mult=cm)
        gb.append(g_b / (b * b)); ga.append(g_a / (b * b))
        print(f"{b:5d} || {g_b:10.2f} {g_b/(b*b):11.3f} | {g_a:10.2f} {g_a/(b*b):11.3f}", flush=True)
    # a couple of big-board probes
    print("\n== big-board probe (greedy score, few eps) ==")
    for b in [48, 64]:
        g_b, _ = greedy_score(base, b, n=4, cap_mult=2)
        g_a, _ = greedy_score(attn, b, n=4, cap_mult=2)
        print(f"{b:5d} || base {g_b:7.2f} | attn {g_a:7.2f}", flush=True)

    print("\n== food margin (real play) & canonical (fixed scene) ==")
    print(f"{'board':>5} || {'base fm':>8} {'attn fm':>8} | {'base canon':>10} {'attn canon':>10}")
    base_c = canonical_curve(base, MARGIN_SIZES); attn_c = canonical_curve(attn, MARGIN_SIZES)
    fmb, fma = [], []
    for i, b in enumerate(MARGIN_SIZES):
        f_b, _, _, _ = margin_decomp(base, b)
        f_a, _, _, _ = margin_decomp(attn, b)
        fmb.append(f_b); fma.append(f_a)
        print(f"{b:5d} || {f_b:8.3f} {f_a:8.3f} | {base_c[i]:10.3f} {attn_c[i]:10.3f}", flush=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    ax1.plot(TRANSFER_SIZES, gb, "s-", color="#dc2626", label="avg-pool baseline")
    ax1.plot(TRANSFER_SIZES, ga, "o-", color="#16a34a", label="attention pool (fix)")
    ax1.axvline(6, ls=":", color="gray", alpha=0.5)
    ax1.set_xlabel("board size (zero-shot)"); ax1.set_ylabel("score / cells")
    ax1.set_title("Zero-shot transfer: does the fix move the cliff?"); ax1.legend()
    ax2.plot(MARGIN_SIZES, fmb, "s-", color="#dc2626", label="avg-pool baseline")
    ax2.plot(MARGIN_SIZES, fma, "o-", color="#16a34a", label="attention pool (fix)")
    ax2.axvline(6, ls=":", color="gray", alpha=0.5)
    ax2.set_xlabel("board size"); ax2.set_ylabel("food margin (Q units)")
    ax2.set_title("Food margin: does the fix flatten the collapse?"); ax2.legend()
    fig.tight_layout(); fig.savefig(f"{OUT}/attn_pool.png", dpi=130)
    print(f"\nWrote {OUT}/attn_pool.png")


if __name__ == "__main__":
    if "--eval-only" not in sys.argv:
        print("== Training AttnPoolQNet on 6x6 ==", flush=True)
        train()
    evaluate()
