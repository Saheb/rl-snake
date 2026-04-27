"""
==============================================================================
PPO (Proximal Policy Optimization) for Snake — production setup
==============================================================================

This script is the on-policy counterpart to `train_dqn.py`. It mirrors that
file's CLI, state representation, environment, and logging conventions so
runs can be compared apples-to-apples (including under sparse / pure_sparse
reward regimes).

Differences from train_dqn.py:
  * On-policy: no replay buffer, no PER, no foundation memory.
  * Single ActorCritic network with a shared trunk + policy / value heads.
  * Clipped surrogate objective with GAE-λ advantage estimation.
  * ICM (when enabled) is trained on the on-policy rollout each update,
    so intrinsic rewards always reflect the *current* world model.

Usage examples (same flags as train_dqn.py where they overlap):
    python scripts/train_ppo.py --board_size 8 --num_games 5000 \
        --reward_mode sparse --icm_eta 0.1 --seed 1 --run_tag ppo_test
"""

import argparse
import collections
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import wandb
from torch.distributions import Categorical

from envs.snake_game import SnakeGame
from utils.icm import ICM


# =============================================================================
# State dimensionality must match train_dqn.py for fair comparison.
# =============================================================================
STATE_DIM = 24
ACTION_DIM = 3  # relative actions: straight, right, left


# =============================================================================
# Action helpers (mirrors train_dqn.py)
# =============================================================================
def get_absolute_action(relative_move, game):
    """Convert relative [straight, right, left] one-hot to absolute direction."""
    head = game.snake_position[-1]
    if len(game.snake_position) > 1:
        neck = game.snake_position[-2]
        if head[0] < neck[0]:
            direction = 0
        elif head[0] > neck[0]:
            direction = 1
        elif head[1] < neck[1]:
            direction = 2
        else:
            direction = 3
    else:
        direction = 3

    right_turn = {0: 3, 3: 1, 1: 2, 2: 0}
    left_turn = {0: 2, 2: 1, 1: 3, 3: 0}

    if relative_move == [1, 0, 0]:
        return direction
    elif relative_move == [0, 1, 0]:
        return right_turn[direction]
    elif relative_move == [0, 0, 1]:
        return left_turn[direction]
    return direction


def _get_reachable_count(game, start_pos):
    """BFS flood-fill over the empty cells reachable from start_pos."""
    obstacles = set(game.snake_position)
    if start_pos in obstacles or game._check_collision(start_pos, False):
        return 0

    visited = set([start_pos])
    queue = collections.deque([start_pos])
    count = 0
    max_search = max(len(game.snake_position) * 3, 20)
    while queue and count < max_search:
        cx, cy = queue.popleft()
        count += 1
        for nx, ny in [(cx - 1, cy), (cx + 1, cy), (cx, cy - 1), (cx, cy + 1)]:
            if (
                0 <= nx < game.board_size
                and 0 <= ny < game.board_size
                and (nx, ny) not in obstacles
                and (nx, ny) not in visited
            ):
                visited.add((nx, ny))
                queue.append((nx, ny))
    return count


def get_state_features(game):
    """24-dim state representation (identical to train_dqn.py's DQNAgent.get_state)."""
    head = game.snake_position[-1]

    point_l = (head[0], head[1] - 1)
    point_r = (head[0], head[1] + 1)
    point_u = (head[0] - 1, head[1])
    point_d = (head[0] + 1, head[1])

    if len(game.snake_position) > 1:
        neck = game.snake_position[-2]
        if head[0] < neck[0]:
            dir_u, dir_r, dir_d, dir_l = True, False, False, False
        elif head[0] > neck[0]:
            dir_u, dir_r, dir_d, dir_l = False, False, True, False
        elif head[1] < neck[1]:
            dir_u, dir_r, dir_d, dir_l = False, False, False, True
        else:
            dir_u, dir_r, dir_d, dir_l = False, True, False, False
    else:
        dir_u, dir_r, dir_d, dir_l = False, True, False, False

    clock_wise_points = [point_u, point_r, point_d, point_l]
    if dir_u:
        idx = 0
    elif dir_r:
        idx = 1
    elif dir_d:
        idx = 2
    else:
        idx = 3

    pos_straight = clock_wise_points[idx]
    pos_right = clock_wise_points[(idx + 1) % 4]
    pos_left = clock_wise_points[(idx - 1) % 4]

    total_cells = game.board_size * game.board_size
    b = game.board_size

    flood_straight = _get_reachable_count(game, pos_straight) / total_cells
    flood_right = _get_reachable_count(game, pos_right) / total_cells
    flood_left = _get_reachable_count(game, pos_left) / total_cells

    tail = game.snake_position[0]
    tail_up = int(tail[0] < head[0])
    tail_down = int(tail[0] > head[0])
    tail_left = int(tail[1] < head[1])
    tail_right = int(tail[1] > head[1])

    snake_length_norm = len(game.snake_position) / total_cells

    if game.food_position:
        food_dist_norm = (
            abs(head[0] - game.food_position[0]) + abs(head[1] - game.food_position[1])
        ) / (2 * (b - 1))
    else:
        food_dist_norm = 0.0

    wall_up = head[0] / (b - 1)
    wall_down = (b - 1 - head[0]) / (b - 1)
    wall_left = head[1] / (b - 1)
    wall_right = (b - 1 - head[1]) / (b - 1)

    state = [
        # Danger straight / right / left (3)
        (dir_r and game._check_collision(point_r))
        or (dir_l and game._check_collision(point_l))
        or (dir_u and game._check_collision(point_u))
        or (dir_d and game._check_collision(point_d)),
        (dir_u and game._check_collision(point_r))
        or (dir_d and game._check_collision(point_l))
        or (dir_l and game._check_collision(point_u))
        or (dir_r and game._check_collision(point_d)),
        (dir_d and game._check_collision(point_r))
        or (dir_u and game._check_collision(point_l))
        or (dir_r and game._check_collision(point_u))
        or (dir_l and game._check_collision(point_d)),
        # Direction (4)
        dir_l, dir_r, dir_u, dir_d,
        # Food direction (4)
        (game.food_position[1] < head[1]) if game.food_position else 0,
        (game.food_position[1] > head[1]) if game.food_position else 0,
        (game.food_position[0] < head[0]) if game.food_position else 0,
        (game.food_position[0] > head[0]) if game.food_position else 0,
        # Flood-fill open space (3)
        flood_straight, flood_right, flood_left,
        # Tail direction (4)
        tail_up, tail_down, tail_left, tail_right,
        # Snake length normalized (1)
        snake_length_norm,
        # Food distance normalized (1)
        food_dist_norm,
        # Wall distances normalized (4)
        wall_up, wall_down, wall_left, wall_right,
    ]
    return np.asarray(state, dtype=np.float32)


# =============================================================================
# ActorCritic network
# =============================================================================
class ActorCritic(nn.Module):
    """Shared trunk with separate policy logits + scalar value heads."""

    def __init__(self, input_size=STATE_DIM, hidden_size=256, output_size=ACTION_DIM):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
        )
        # Policy logits (we apply softmax via Categorical for numerical stability).
        self.actor_head = nn.Linear(hidden_size // 2, output_size)
        # Scalar state value
        self.critic_head = nn.Linear(hidden_size // 2, 1)

    def forward(self, x):
        feat = self.shared(x)
        logits = self.actor_head(feat)
        value = self.critic_head(feat).squeeze(-1)
        return logits, value


# =============================================================================
# PPO Agent
# =============================================================================
class PPOAgent:
    """On-policy PPO agent with optional ICM."""

    def __init__(
        self,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        n_epochs=4,
        n_minibatches=4,
        entropy_coef=0.01,
        value_coef=0.5,
        max_grad_norm=0.5,
        use_icm=False,
        icm_eta=0.01,
        icm_lr=1e-3,
    ):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.n_epochs = n_epochs
        self.n_minibatches = n_minibatches
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm

        self.network = ActorCritic()
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)

        self.use_icm = use_icm
        self.icm = (
            ICM(state_dim=STATE_DIM, action_dim=ACTION_DIM, lr=icm_lr, eta=icm_eta)
            if use_icm
            else None
        )

        # Rollout storage
        self._reset_rollout()

        self.n_games = 0
        self.total_steps = 0

    def _reset_rollout(self):
        self.buf_states = []
        self.buf_actions = []
        self.buf_log_probs = []
        self.buf_rewards = []
        self.buf_values = []
        self.buf_dones = []
        self.buf_next_states = []

    def select_action(self, state):
        """Sample action from current policy. Returns (relative_one_hot, action_idx, log_prob, value)."""
        with torch.no_grad():
            t = torch.from_numpy(state).unsqueeze(0)
            logits, value = self.network(t)
            dist = Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        idx = int(action.item())
        relative = [0, 0, 0]
        relative[idx] = 1
        return relative, idx, float(log_prob.item()), float(value.item())

    def compute_intrinsic(self, state, action_idx, next_state):
        if not self.use_icm:
            return 0.0
        action_one_hot = np.zeros(ACTION_DIM, dtype=np.float32)
        action_one_hot[action_idx] = 1.0
        return self.icm.compute_intrinsic_reward(state, action_one_hot, next_state)

    def store(self, state, action_idx, log_prob, reward, value, done, next_state):
        self.buf_states.append(state)
        self.buf_actions.append(action_idx)
        self.buf_log_probs.append(log_prob)
        self.buf_rewards.append(reward)
        self.buf_values.append(value)
        self.buf_dones.append(done)
        self.buf_next_states.append(next_state)

    def _compute_gae(self, last_value):
        """GAE-λ advantage estimation. Returns advantages, returns (= adv + values)."""
        T = len(self.buf_rewards)
        advantages = np.zeros(T, dtype=np.float32)
        gae = 0.0
        for t in reversed(range(T)):
            if self.buf_dones[t]:
                next_value = 0.0
                gae_carry = 0.0  # reset GAE at episode boundaries
            else:
                next_value = self.buf_values[t + 1] if t + 1 < T else last_value
                gae_carry = gae
            delta = self.buf_rewards[t] + self.gamma * next_value - self.buf_values[t]
            gae = delta + self.gamma * self.gae_lambda * gae_carry
            advantages[t] = gae
        returns = advantages + np.asarray(self.buf_values, dtype=np.float32)
        return advantages, returns

    def update(self, last_state, last_done):
        """Run K PPO epochs over the current rollout, then clear."""
        # Bootstrap value if rollout ended mid-episode.
        if last_done:
            last_value = 0.0
        else:
            with torch.no_grad():
                t = torch.from_numpy(last_state).unsqueeze(0)
                _, v = self.network(t)
                last_value = float(v.item())

        advantages, returns = self._compute_gae(last_value)

        states = torch.from_numpy(np.asarray(self.buf_states, dtype=np.float32))
        next_states = torch.from_numpy(np.asarray(self.buf_next_states, dtype=np.float32))
        actions = torch.tensor(self.buf_actions, dtype=torch.long)
        old_log_probs = torch.tensor(self.buf_log_probs, dtype=torch.float32)
        adv_t = torch.from_numpy(advantages)
        ret_t = torch.from_numpy(returns)

        # Per-rollout advantage normalization.
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        N = states.shape[0]
        minibatch_size = max(1, N // self.n_minibatches)
        indices = np.arange(N)

        last_loss = {"actor": 0.0, "critic": 0.0, "entropy": 0.0, "approx_kl": 0.0, "clipfrac": 0.0}

        for _ in range(self.n_epochs):
            np.random.shuffle(indices)
            for start in range(0, N, minibatch_size):
                mb_idx = indices[start : start + minibatch_size]
                if len(mb_idx) == 0:
                    continue

                mb_states = states[mb_idx]
                mb_actions = actions[mb_idx]
                mb_old_log_probs = old_log_probs[mb_idx]
                mb_adv = adv_t[mb_idx]
                mb_ret = ret_t[mb_idx]

                logits, values = self.network(mb_states)
                dist = Categorical(logits=logits)
                new_log_probs = dist.log_prob(mb_actions)
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_log_probs - mb_old_log_probs)
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * mb_adv
                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = F.mse_loss(values, mb_ret)

                loss = actor_loss + self.value_coef * critic_loss - self.entropy_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()

                # Diagnostics (overwritten each minibatch; final values reflect the last mb).
                with torch.no_grad():
                    approx_kl = (mb_old_log_probs - new_log_probs).mean().item()
                    clipfrac = ((ratio - 1.0).abs() > self.clip_epsilon).float().mean().item()
                last_loss["actor"] = float(actor_loss.item())
                last_loss["critic"] = float(critic_loss.item())
                last_loss["entropy"] = float(entropy.item())
                last_loss["approx_kl"] = approx_kl
                last_loss["clipfrac"] = clipfrac

        # Train ICM on the rollout (one pass on the full batch).
        if self.use_icm and self.icm is not None:
            actions_one_hot = F.one_hot(actions, num_classes=ACTION_DIM).float().to(self.icm.device)
            states_dev = states.to(self.icm.device)
            next_states_dev = next_states.to(self.icm.device)
            self.icm.train_batch(states_dev, actions_one_hot, next_states_dev)

        self._reset_rollout()
        return last_loss


# =============================================================================
# Training loop
# =============================================================================
def train(
    use_icm=True,
    icm_eta=0.01,
    icm_lr=1e-3,
    board_size=8,
    num_games=5000,
    reward_mode="dense",
    rollout_length=2048,
    seed=None,
    run_tag=None,
):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    if run_tag is None:
        run_tag = f"PPO_{'ICM' if use_icm else 'Baseline'}_{board_size}x{board_size}_{reward_mode}"

    wandb.init(
        project="rl-snake",
        name=run_tag,
        config={
            "algorithm": "PPO",
            "board_size": board_size,
            "use_icm": use_icm,
            "reward_mode": reward_mode,
            "icm_eta": icm_eta if use_icm else 0,
            "icm_lr": icm_lr if use_icm else 0,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_epsilon": 0.2,
            "n_epochs": 4,
            "n_minibatches": 4,
            "rollout_length": rollout_length,
            "learning_rate": 3e-4,
            "num_games": num_games,
            "seed": seed,
        },
    )
    wandb.define_metric("Game")
    wandb.define_metric("*", step_metric="Game")

    agent = PPOAgent(
        use_icm=use_icm,
        icm_eta=icm_eta,
        icm_lr=icm_lr,
    )
    game = SnakeGame(board_size=board_size, reward_mode=reward_mode)

    MAX_STEPS = 500 if board_size <= 5 else (3000 if board_size <= 8 else 5000)
    print(
        f"Starting {'ICM' if use_icm else 'Baseline'} PPO Training on "
        f"{board_size}x{board_size}, reward_mode={reward_mode}..."
    )

    record = 0
    total_score = 0
    mean_score = 0.0
    steps_in_game = 0
    game_intrinsic_reward = 0.0

    visited_cumulative = set()
    visited_window = set()

    last_state = get_state_features(game)
    last_done = False

    while agent.n_games < num_games:
        state = get_state_features(game)
        relative, action_idx, log_prob, value = agent.select_action(state)
        absolute = get_absolute_action(relative, game)
        _, reward, done, info = game.step(absolute)
        score = info["score"]

        steps_in_game += 1
        agent.total_steps += 1

        if steps_in_game > MAX_STEPS:
            done = True
            reward = -1  # match envs/snake_game.py death reward

        next_state = get_state_features(game)

        # Coverage tracking (post-step head/food pair).
        if game.snake_position and game.food_position:
            head = game.snake_position[-1]
            key = (head[0], head[1], game.food_position[0], game.food_position[1])
            visited_cumulative.add(key)
            visited_window.add(key)

        intrinsic_reward = 0.0
        if use_icm:
            intrinsic_reward = agent.compute_intrinsic(state, action_idx, next_state)
            # Mask intrinsic on terminal: keeps parity with train_dqn.py mask_terminal_intrinsic=True.
            intrinsic_reward = intrinsic_reward * (1.0 - float(done))
            reward += intrinsic_reward
            game_intrinsic_reward += intrinsic_reward

        agent.store(state, action_idx, log_prob, reward, value, done, next_state)

        last_state = next_state
        last_done = done

        # PPO update on rollout boundary.
        if len(agent.buf_states) >= rollout_length:
            agent.update(last_state, last_done)

        if done:
            game.reset()
            agent.n_games += 1
            steps_in_game = 0

            if score > record:
                record = score

            total_score += score
            mean_score = total_score / agent.n_games

            if agent.n_games % 100 == 0:
                icm_eta_str = f" | Eta: {agent.icm.eta:.5f}" if (use_icm and agent.icm is not None) else ""
                cov_cum = len(visited_cumulative)
                cov_win = len(visited_window)
                cov_max = board_size ** 4
                print(
                    f"Game {agent.n_games} | Score: {score} | Record: {record} | "
                    f"Mean Score: {mean_score:.2f} | Intrinsic: {game_intrinsic_reward:.3f}"
                    f"{icm_eta_str}"
                )
                print(
                    f"[coverage] Game {agent.n_games} | unique_states_cumulative: "
                    f"{cov_cum}/{cov_max} ({100 * cov_cum / cov_max:.1f}%) | "
                    f"last100_unique: {cov_win}"
                )

                log_metrics = {
                    "Game": agent.n_games,
                    "Score": score,
                    "Record": record,
                    "Mean_Score": mean_score,
                    "Coverage_Cumulative": cov_cum,
                    "Coverage_Window100": cov_win,
                    "Coverage_Cumulative_Frac": cov_cum / cov_max,
                }
                if use_icm:
                    log_metrics["Intrinsic_Reward"] = game_intrinsic_reward
                    log_metrics["Eta"] = agent.icm.eta
                wandb.log(log_metrics)
                visited_window = set()

            game_intrinsic_reward = 0.0

    # Flush any remaining rollout before exit.
    if len(agent.buf_states) > 0:
        agent.update(last_state, last_done)

    print("\nTraining Complete!")
    print(f"Best Score: {record}")
    print(f"Final Mean Score: {mean_score:.2f}")
    wandb.finish()


# =============================================================================
# CLI
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--board_size", type=int, default=8)
    parser.add_argument("--num_games", type=int, default=5000)
    parser.add_argument("--disable_icm", action="store_true")
    parser.add_argument("--icm_eta", type=float, default=0.01)
    parser.add_argument(
        "--reward_mode",
        type=str,
        default="dense",
        choices=["dense", "sparse", "pure_sparse"],
    )
    parser.add_argument("--rollout_length", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--run_tag", type=str, default=None)
    args = parser.parse_args()

    train(
        use_icm=not args.disable_icm,
        icm_eta=args.icm_eta,
        icm_lr=1e-3,
        board_size=args.board_size,
        num_games=args.num_games,
        reward_mode=args.reward_mode,
        rollout_length=args.rollout_length,
        seed=args.seed,
        run_tag=args.run_tag,
    )
