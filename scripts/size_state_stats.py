"""
Stop tweaking architecture. Characterize how the STATE DISTRIBUTION changes with board size.

We roll out a FIXED scripted competent policy (greedy toward food + collision avoidance) — identical
behavior at every board size — so any change in the state statistics is due to board size alone, not
policy. We measure the quantities that could make a 6x6-trained policy OOD on bigger boards, and in
particular the FRACTION OF BIG-BOARD PLAY THAT FALLS OUTSIDE THE 6x6 TRAINING SUPPORT.

6x6 support bounds (hard limits of the training board):
  - food distance (Manhattan) <= 2*(6-1) = 10
  - nearest-wall distance     <= 2         (6x6 has no cell farther than 2 from every wall)
  - food within the 5x5 receptive field (Chebyshev <= 2) is common; mid-board-far-food is not.
If a statistic on 10x10/16x16/25x25 lives largely OUTSIDE these bounds, the policy simply never trained
on that regime — architecture was never the bottleneck.
"""
import sys, random, numpy as np
sys.path.insert(0, "/Users/saheb/home/rl-snake")
from envs.snake_game import SnakeGame

DELTAS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
BOARDS = [6, 10, 16, 25]
STEPS = 20000


def scripted_action(game):
    """Greedy toward food among non-fatal moves; tie-break random. Competent at any board size."""
    head = game.snake_position[-1]; food = game.food_position
    d0 = abs(head[0]-food[0]) + abs(head[1]-food[1])
    safe_toward, safe_any = [], []
    for a, (dx, dy) in enumerate(DELTAS):
        nh = (head[0]+dx, head[1]+dy)
        if game._check_collision(nh, nh == food):
            continue
        safe_any.append(a)
        if abs(nh[0]-food[0]) + abs(nh[1]-food[1]) < d0:
            safe_toward.append(a)
    if safe_toward:
        return random.choice(safe_toward)
    if safe_any:
        return random.choice(safe_any)
    return random.randint(0, 3)


def rollout_stats(board, steps=STEPS, seed=0):
    random.seed(seed); np.random.seed(seed)
    fd, cheby, wd, slen = [], [], [], []
    turns = 0; n = 0; prev_a = None
    ep_steps = []; ep_food = []; cur_steps = 0; cur_food = 0
    game = SnakeGame(board_size=board, reward_mode="dense"); game.reset()
    while n < steps:
        head = game.snake_position[-1]; food = game.food_position
        if food is None:
            game.reset(); prev_a = None; continue
        fd.append(abs(head[0]-food[0]) + abs(head[1]-food[1]))
        cheby.append(max(abs(head[0]-food[0]), abs(head[1]-food[1])))
        wd.append(min(head[0], board-1-head[0], head[1], board-1-head[1]))
        slen.append(len(game.snake_position))
        a = scripted_action(game)
        if prev_a is not None and a != prev_a:
            turns += 1
        prev_a = a; n += 1; cur_steps += 1
        pscore = game.score
        _, _, done, info = game.step(a)
        if game.score > pscore:
            cur_food += 1
        if done:
            ep_steps.append(cur_steps); ep_food.append(cur_food)
            cur_steps = 0; cur_food = 0; prev_a = None
            game = SnakeGame(board_size=board, reward_mode="dense"); game.reset()
    fd = np.array(fd); cheby = np.array(cheby); wd = np.array(wd); slen = np.array(slen)
    steps_per_food = (sum(ep_steps) / max(sum(ep_food), 1)) if ep_food else float("nan")
    return {
        "food_dist_mean": fd.mean(), "food_dist_p90": np.percentile(fd, 90),
        "pct_food_beyond_RF": 100 * (cheby > 2).mean(),
        "wall_dist_mean": wd.mean(), "pct_far_from_walls": 100 * (wd > 2).mean(),
        "snake_len_mean": slen.mean(), "snake_len_max": slen.max(),
        "turn_freq": 100 * turns / n, "steps_per_food": steps_per_food,
        "pct_outside_6x6_support": 100 * ((fd > 10) | (wd > 2)).mean(),
    }


def main():
    rows = {b: rollout_stats(b) for b in BOARDS}
    keys = [
        ("food_dist_mean", "food dist (mean)", "{:.1f}"),
        ("food_dist_p90", "food dist (p90)", "{:.1f}"),
        ("pct_food_beyond_RF", "% food beyond 5x5 RF", "{:.0f}%"),
        ("wall_dist_mean", "wall dist (mean)", "{:.1f}"),
        ("pct_far_from_walls", "% far from ALL walls (>2)", "{:.0f}%"),
        ("snake_len_mean", "snake length (mean)", "{:.1f}"),
        ("turn_freq", "turn frequency", "{:.0f}%"),
        ("steps_per_food", "steps per food", "{:.1f}"),
        ("pct_outside_6x6_support", "% states OUTSIDE 6x6 support", "{:.0f}%"),
    ]
    hdr = "  ".join(f"{b:>8d}" for b in BOARDS)
    print(f"\n{'statistic':>30} | {hdr}   (scripted competent play)")
    print("-" * (33 + 10 * len(BOARDS)))
    for k, label, fmt in keys:
        vals = "  ".join(f"{fmt.format(rows[b][k]):>8}" for b in BOARDS)
        print(f"{label:>30} | {vals}")


if __name__ == "__main__":
    main()
