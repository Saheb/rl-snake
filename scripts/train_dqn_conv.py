import sys
sys.stdout.reconfigure(line_buffering=True)

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import argparse
import wandb

from envs.snake_game import SnakeGame
from utils.per import PrioritizedReplayBuffer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

# ============================================================================
# HYPERPARAMETERS (tuned via 30-trial W&B random sweep, sweep ID: igzvteil)
# Winner: trial 7 — mean 9.38 at 2000 games on 10×10 dense
# Key findings from sweep:
#   - N_STEPS=1: multi-step returns hurt when combined with PER (all top configs)
#   - ch1=64, ch2=128: larger second conv layer consistently better
#   - epsilon_end=0.01: exploit harder than 0.05
#   - epsilon_decay_fraction=0.7: slow decay gives more exploration budget
#   - gamma=0.995: very patient, values distant food
#   - train_calls=10: more gradient updates per episode = better
GAMMA = 0.995
BATCH_SIZE = 512
REPLAY_CAPACITY = 100_000
LEARNING_STARTS = 5_000   # random steps before any training
TRAIN_CALLS = 10          # gradient updates per episode end
LR = 0.000126
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY_FRACTION = 0.7  # fraction of est. total steps over which to decay
N_STEPS = 1


# ============================================================================
# STATE REPRESENTATION
# ============================================================================
def get_conv_state(game):
    """
    Return a (3, H, W) float32 array.
      Ch 0 — snake body (all segments except head)
      Ch 1 — snake head
      Ch 2 — food
    Direction is implicit in the body shape; 4 absolute actions, no feature engineering.
    """
    H = W = game.board_size
    state = np.zeros((3, H, W), dtype=np.float32)
    snake = list(game.snake_position)
    for pos in snake[:-1]:
        state[0, pos[0], pos[1]] = 1.0
    head = snake[-1]
    state[1, head[0], head[1]] = 1.0
    if game.food_position:
        state[2, game.food_position[0], game.food_position[1]] = 1.0
    return state


# ============================================================================
# NEURAL NETWORK
# ============================================================================
class ConvDuelingQNet(nn.Module):
    """
    Two-layer CNN + Dueling head.
    Two conv layers give a 5×5 receptive field — sufficient for 10×10 and 16×16.
    The global FC layer aggregates spatial information across the full board.
    """
    def __init__(self, board_size, n_actions=4):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(),
        )
        flat = 128 * board_size * board_size
        self.value_stream = nn.Sequential(nn.Linear(flat, 128), nn.ReLU(), nn.Linear(128, 1))
        self.advantage_stream = nn.Sequential(nn.Linear(flat, 128), nn.ReLU(), nn.Linear(128, n_actions))

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        val = self.value_stream(x)
        adv = self.advantage_stream(x)
        return val + (adv - adv.mean(dim=1, keepdim=True))

    def save(self, path):
        torch.save(self.state_dict(), path)


# ============================================================================
# Q-TRAINER
# ============================================================================
class ConvQTrainer:
    def __init__(self, model, target_model):
        self.model = model
        self.target_model = target_model
        self.optimizer = optim.Adam(model.parameters(), lr=LR)
        self.criterion = nn.MSELoss(reduction="none")

    def train_step(self, state, action, reward, next_state, done, is_weights=None):
        state = torch.tensor(np.array(state), dtype=torch.float).to(DEVICE)
        next_state = torch.tensor(np.array(next_state), dtype=torch.float).to(DEVICE)
        action = torch.tensor(np.array(action), dtype=torch.long).to(DEVICE)
        reward = torch.tensor(np.array(reward), dtype=torch.float).to(DEVICE)

        if state.dim() == 3:   # single transition: add batch dim
            state = state.unsqueeze(0); next_state = next_state.unsqueeze(0)
            action = action.unsqueeze(0); reward = reward.unsqueeze(0); done = (done,)

        pred = self.model(state)

        with torch.no_grad():
            next_best = torch.argmax(self.model(next_state), dim=1)
            next_q = self.target_model(next_state)

        target = pred.clone()
        gamma_n = GAMMA ** N_STEPS
        for i in range(len(done)):
            q_new = reward[i] if done[i] else reward[i] + gamma_n * next_q[i][next_best[i]]
            target[i][action[i].item()] = q_new

        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        td_errors = torch.abs(target - pred).sum(dim=1).detach().cpu().numpy()

        if is_weights is not None:
            w = torch.tensor(is_weights, dtype=torch.float).to(DEVICE)
            loss = (loss.mean(dim=1) * w).mean()
        else:
            loss = loss.mean()

        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
        self.optimizer.step()
        return td_errors


# ============================================================================
# AGENT
# ============================================================================
class ConvDQNAgent:
    def __init__(self, board_size, epsilon_decay_per_step):
        self.n_games = 0
        self.total_steps = 0
        self.epsilon = EPSILON_START
        self.epsilon_decay_per_step = epsilon_decay_per_step
        self.board_size = board_size

        self.n_step_buffer = deque(maxlen=N_STEPS)
        self.memory = PrioritizedReplayBuffer(capacity=REPLAY_CAPACITY, priority_cap=1.0)

        self.model = ConvDuelingQNet(board_size, n_actions=4).to(DEVICE)
        self.target_model = ConvDuelingQNet(board_size, n_actions=4).to(DEVICE)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()
        self.trainer = ConvQTrainer(self.model, self.target_model)

    def update_target_network(self, tau=0.005):
        for tp, lp in zip(self.target_model.parameters(), self.model.parameters()):
            tp.data.copy_(tau * lp.data + (1.0 - tau) * tp.data)

    def get_action(self, state):
        # Epsilon decays per environment step, not per episode.
        # This ensures equal exploration budget regardless of board size / game length.
        self.epsilon = max(EPSILON_END, EPSILON_START - self.total_steps * self.epsilon_decay_per_step)
        if random.random() < self.epsilon:
            return random.randint(0, 3)
        s = torch.tensor(state, dtype=torch.float).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            return torch.argmax(self.model(s)).item()

    def remember(self, state, action, reward, next_state, done):
        self.n_step_buffer.append((state, action, reward, next_state, done))
        if len(self.n_step_buffer) < N_STEPS:
            return
        R, next_s, d = self._n_step_info()
        s0, a0 = self.n_step_buffer[0][:2]
        self.memory.add((s0, a0, R, next_s, d))

    def _n_step_info(self):
        R = 0
        for i, t in enumerate(self.n_step_buffer):
            R += t[2] * (GAMMA ** i)
            if t[4]:
                return R, t[3], True
        return R, self.n_step_buffer[-1][3], False

    def train(self):
        if len(self.memory) < BATCH_SIZE or self.total_steps < LEARNING_STARTS:
            return
        batch, indices, weights = self.memory.sample(BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*batch)
        td_errors = self.trainer.train_step(
            states, actions, rewards, next_states, dones, is_weights=weights
        )
        self.memory.update_priorities(indices, td_errors)


# ============================================================================
# TRAINING LOOP
# ============================================================================
def train(board_size=10, num_games=5000, reward_mode="dense", seed=None, run_tag=None):
    if seed is not None:
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

    if run_tag is None:
        run_tag = (
            f"ConvDQN_v3_{board_size}x{board_size}_{reward_mode}_g{num_games}"
            + (f"_seed{seed}" if seed is not None else "")
        )

    wandb.init(
        project="rl-snake", name=run_tag,
        config={
            "algorithm": "ConvDQN_v3",
            "board_size": board_size,
            "reward_mode": reward_mode,
            "gamma": GAMMA,
            "batch_size": BATCH_SIZE,
            "train_calls_per_episode": TRAIN_CALLS,
            "learning_starts": LEARNING_STARTS,
            "lr": LR,
            "n_steps": N_STEPS,
            "epsilon_start": EPSILON_START,
            "epsilon_end": EPSILON_END,
            "epsilon_decay_fraction": EPSILON_DECAY_FRACTION,
            "num_games": num_games,
            "seed": seed,
            "sweep_id": "igzvteil",
        },
    )
    wandb.define_metric("Game")
    wandb.define_metric("*", step_metric="Game")

    # Epsilon decays from 1.0 → EPSILON_END over first EPSILON_DECAY_FRACTION of est. total steps.
    # Estimate: num_games × board_size² × 0.15 steps/cell (conservative avg game length)
    est_total_steps = num_games * board_size * board_size * 0.15
    epsilon_decay_per_step = (EPSILON_START - EPSILON_END) / (est_total_steps * EPSILON_DECAY_FRACTION)

    agent = ConvDQNAgent(board_size=board_size, epsilon_decay_per_step=epsilon_decay_per_step)
    game = SnakeGame(board_size=board_size, reward_mode=reward_mode)

    MAX_STEPS = board_size * board_size * 30  # generous cap

    total_score = 0
    record = 0
    steps_in_game = 0

    print(
        f"ConvDQN v3 | {board_size}×{board_size} {reward_mode} | "
        f"γ={GAMMA} batch={BATCH_SIZE} calls/ep={TRAIN_CALLS} "
        f"lr={LR} n_steps={N_STEPS} ε_end={EPSILON_END} ε_decay/step={epsilon_decay_per_step:.2e}"
    )

    while agent.n_games < num_games:
        state_old = get_conv_state(game)
        action = agent.get_action(state_old)
        _, reward, done, info = game.step(action)
        score = info["score"]
        steps_in_game += 1
        agent.total_steps += 1

        if steps_in_game > MAX_STEPS:
            done = True; reward = -1

        state_new = get_conv_state(game)
        agent.remember(state_old, action, reward, state_new, done)

        if done:
            # Flush n-step buffer
            while agent.n_step_buffer:
                R, next_s, d = agent._n_step_info()
                s0, a0 = agent.n_step_buffer.popleft()[:2]
                agent.memory.add((s0, a0, R, next_s, d))

            game.reset()
            agent.n_games += 1
            steps_in_game = 0

            # Batch training — TRAIN_CALLS gradient updates per episode
            for _ in range(TRAIN_CALLS):
                agent.train()

            # Soft Polyak target update
            agent.update_target_network(tau=0.005)

            if score > record:
                record = score
                agent.model.save(f"model_conv_v2_{board_size}x{board_size}.pth")

            total_score += score
            mean_score = total_score / agent.n_games

            # LR cooldown at 87.5% of training
            if agent.n_games == int(num_games * 0.875):
                for pg in agent.trainer.optimizer.param_groups:
                    pg["lr"] = 1e-5
                print(f"[Game {agent.n_games}] LR cooldown → 1e-5")

            if agent.n_games % 100 == 0:
                print(
                    f"Game {agent.n_games} | Score: {score} | Record: {record}"
                    f" | Mean: {mean_score:.2f} | ε: {agent.epsilon:.3f}"
                    f" | Steps: {agent.total_steps:,}"
                )
                wandb.log({
                    "Game": agent.n_games,
                    "Score": score,
                    "Record": record,
                    "Mean_Score": mean_score,
                    "Epsilon": agent.epsilon,
                    "Total_Steps": agent.total_steps,
                })

    print(f"\nDone. Best: {record}, Final Mean: {mean_score:.2f}")
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--board_size", type=int, default=10)
    parser.add_argument("--num_games", type=int, default=5000)
    parser.add_argument("--reward_mode", type=str, default="dense",
                        choices=["dense", "sparse", "pure_sparse"])
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--run_tag", type=str, default=None)
    args = parser.parse_args()
    train(board_size=args.board_size, num_games=args.num_games,
          reward_mode=args.reward_mode, seed=args.seed, run_tag=args.run_tag)
