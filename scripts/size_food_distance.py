"""
#1 Definitive isolation: is the failure food-DISTANCE or board-SIZE?

They covary in the board-size sweep. Here we DECOUPLE them: fix the board at 10x10 for both
training and eval, and vary only the food-distance distribution.
  - near-only agent: food always placed <= DMAX_NEAR cells from the head (trained here).
  - full-range agent: normal random food = all distances (reuse sizeagnostic_10x10_best).
Then eval BOTH on 10x10 with food placed at controlled distances. If near-only fails on far
food on the SAME board the full-range agent handles, the cause is food-distance, not board-size.

Eval: capped length, fixed budget, greedy (no eval epsilon). Metrics: toward-food%, avg & max score.
"""
import os, sys, random, collections, argparse, numpy as np, torch, torch.nn as nn
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, "/Users/saheb/home/rl-snake")
from scripts.size_transfer import SizeAgnosticQNet, get_conv_state, DEVICE
from envs.snake_game import SnakeGame
from utils.per import PrioritizedReplayBuffer

OUT = "/Users/saheb/home/rl-snake/size_transfer"
BOARD, DMAX_NEAR = 10, 4
CAP, BUDGET, N = 5, 400, 40
DIST_BINS = [2, 4, 6, 8, 10, 12, 14]


def place_food(game, dmin=1, dmax=99):
    b = game.board_size; hx, hy = game.snake_position[-1]
    cands = [(x, y) for x in range(b) for y in range(b)
             if (x, y) not in game.snake_position and dmin <= abs(x - hx) + abs(y - hy) <= dmax]
    if cands:
        game.food_position = random.choice(cands)
        game._clear_board(); game._update_board_snake(); game._update_board_food()


def mv(head, a):
    x, y = head
    return [(x - 1, y), (x, y + 1), (x + 1, y), (x, y - 1)][a]


@torch.no_grad()
def greedy_score_near(model, board, dmax, n=30):
    model.eval(); tot = []
    for _ in range(n):
        g = SnakeGame(board_size=board, reward_mode="dense"); g.reset(); place_food(g, 1, dmax)
        info = {"score": 0}
        for _ in range(board * board * 4):
            s = get_conv_state(g)
            a = int(torch.argmax(model(torch.tensor(s[None], device=DEVICE))).item())
            prev = g.food_position
            _, _, done, info = g.step(a)
            if g.food_position != prev and not done:
                place_food(g, 1, dmax)
            if done:
                break
        tot.append(info["score"])
    return float(np.mean(tot))


def train_near(board=BOARD, n_games=4000, dmax=DMAX_NEAR, seed=0):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    GAMMA, LR, BATCH, CAP_, LEARN_START, TRAIN_CALLS, TARGET_UPDATE = 0.99, 1e-4, 256, 100_000, 5000, 8, 1000
    EPS_END, EPS_DECAY_GAMES = 0.02, 0.7 * n_games
    max_steps = board * board * 4
    model = SizeAgnosticQNet().to(DEVICE)
    target = SizeAgnosticQNet().to(DEVICE); target.load_state_dict(model.state_dict()); target.eval()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    mem = PrioritizedReplayBuffer(capacity=CAP_, priority_cap=1.0)
    best = -1.0; grad = 0; scores = collections.deque(maxlen=100)
    for g in range(1, n_games + 1):
        eps = max(EPS_END, 1.0 - g / EPS_DECAY_GAMES)
        game = SnakeGame(board_size=board, reward_mode="dense"); game.reset(); place_food(game, 1, dmax)
        s = get_conv_state(game); info = {"score": 0}
        for _ in range(max_steps):
            a = random.randint(0, 3) if random.random() < eps else \
                int(torch.argmax(model(torch.tensor(s[None], device=DEVICE))).item())
            prev = game.food_position
            _, r, done, info = game.step(a)
            if game.food_position != prev and not done:
                place_food(game, 1, dmax)
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
                br = torch.tensor(br, device=DEVICE, dtype=torch.float); bd = torch.tensor(bd, device=DEVICE, dtype=torch.bool)
                q = model(bs).gather(1, ba[:, None]).squeeze(1)
                with torch.no_grad():
                    na = torch.argmax(model(bns), dim=1)
                    tgt = br + GAMMA * target(bns).gather(1, na[:, None]).squeeze(1) * (~bd)
                td = q - tgt; loss = (isw * td.pow(2)).mean()
                opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 10.0); opt.step()
                mem.update_priorities(idx, td.abs().detach().cpu().numpy()); grad += 1
                if grad % TARGET_UPDATE == 0:
                    target.load_state_dict(model.state_dict())
        if g % 500 == 0:
            gs = greedy_score_near(model, board, dmax, n=30)
            print(f"  game {g:5d} | eps {eps:.2f} | GREEDY(near) {gs:.2f}", flush=True)
            if gs > best:
                best = gs; torch.save(model.state_dict(), f"{OUT}/sizeagnostic_10x10_near_best.pth")
    print(f"  best {best:.2f}", flush=True)


@torch.no_grad()
def eval_by_distance(model, d):
    """Food fixed at ~distance d on a 10x10 board. Returns toward-food%, avg score, max score."""
    tf = []; scs = []
    for _ in range(N):
        g = SnakeGame(board_size=BOARD, reward_mode="dense"); g.reset(); place_food(g, max(1, d - 1), d + 1)
        tw = tot = 0; info = {"score": 0}
        for _ in range(BUDGET):
            head = g.snake_position[-1]; fx = g.food_position
            d0 = abs(head[0] - fx[0]) + abs(head[1] - fx[1]) if fx else 0
            a = int(torch.argmax(model(torch.tensor(get_conv_state(g)[None], device=DEVICE))).item())
            nh = mv(head, a); prev = g.food_position
            _, _, done, info = g.step(a)
            if fx:
                tot += 1; tw += (abs(nh[0] - fx[0]) + abs(nh[1] - fx[1]) < d0)
            if g.food_position != prev and not done:
                place_food(g, max(1, d - 1), d + 1)
            while len(g.snake_position) > CAP:
                g.snake_position.popleft()
            g._clear_board(); g._update_board_snake(); g._update_board_food()
            if done:
                break
        tf.append(tw / max(tot, 1)); scs.append(info["score"])
    return 100 * np.mean(tf), np.mean(scs), np.max(scs)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--skip-train", action="store_true"); ap.add_argument("--games", type=int, default=4000)
    args = ap.parse_args()
    if not args.skip_train:
        print(f"== Training NEAR-ONLY agent on {BOARD}x{BOARD} (food <= {DMAX_NEAR}) ==", flush=True)
        train_near(n_games=args.games)
    agents = {"near-only (d<=4)": f"{OUT}/sizeagnostic_10x10_near_best.pth",
              "full-range": f"{OUT}/sizeagnostic_10x10_best.pth"}
    results = {}
    for name, ckpt in agents.items():
        m = SizeAgnosticQNet().to(DEVICE); m.load_state_dict(torch.load(ckpt, map_location=DEVICE)); m.eval()
        print(f"\n=== {name} — same 10x10 board, food at controlled distance ===")
        print(f'{"food_dist":>10} {"toward-food%":>12} {"avg_score":>10} {"max_score":>10}')
        rows = {"tf": [], "avg": [], "max": []}
        for d in DIST_BINS:
            tf, avg, mx = eval_by_distance(m, d)
            rows["tf"].append(tf); rows["avg"].append(avg); rows["max"].append(mx)
            print(f'{d:10d} {tf:12.0f} {avg:10.1f} {mx:10.0f}', flush=True)
        results[name] = rows
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))
    for name, c in [("near-only (d<=4)", "#dc2626"), ("full-range", "#2563eb")]:
        a1.plot(DIST_BINS, results[name]["tf"], "o-", color=c, label=name)
        a2.plot(DIST_BINS, results[name]["avg"], "o-", color=c, label=name + " avg")
        a2.plot(DIST_BINS, results[name]["max"], "o--", color=c, alpha=0.5, label=name + " max")
    a1.axvline(DMAX_NEAR, ls="--", color="#dc2626", alpha=0.5, label="near-training max")
    a1.axhline(50, ls=":", color="gray", alpha=0.6, label="chance")
    a1.set_xlabel("food distance (same 10x10 board)"); a1.set_ylabel("moves toward food (%)")
    a1.set_title("Isolated: food-distance, board fixed"); a1.legend(fontsize=8)
    a2.set_xlabel("food distance"); a2.set_ylabel("score (food/budget)"); a2.set_title("avg & max score"); a2.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(f"{OUT}/food_distance.png", dpi=130)
    print(f"\nWrote {OUT}/food_distance.png")
