import sys
sys.stdout.reconfigure(line_buffering=True)

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import wandb

from envs.snake_game import SnakeGame
from utils.per import PrioritizedReplayBuffer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


# ============================================================================
# STATE REPRESENTATION
# ============================================================================
def get_conv_state(game):
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
    def __init__(self, board_size, n_actions=4, ch1=32, ch2=64, hidden=256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, ch1, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(ch1, ch2, kernel_size=3, padding=1), nn.ReLU(),
        )
        flat = ch2 * board_size * board_size
        self.value_stream = nn.Sequential(nn.Linear(flat, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        self.advantage_stream = nn.Sequential(nn.Linear(flat, hidden), nn.ReLU(), nn.Linear(hidden, n_actions))

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        val = self.value_stream(x)
        adv = self.advantage_stream(x)
        return val + (adv - adv.mean(dim=1, keepdim=True))


# ============================================================================
# TRAINER
# ============================================================================
class ConvQTrainer:
    def __init__(self, model, target_model, lr, n_steps, gamma):
        self.model = model
        self.target_model = target_model
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.criterion = nn.MSELoss(reduction="none")
        self.n_steps = n_steps
        self.gamma_n = gamma ** n_steps

    def train_step(self, state, action, reward, next_state, done, is_weights=None):
        state = torch.tensor(np.array(state), dtype=torch.float).to(DEVICE)
        next_state = torch.tensor(np.array(next_state), dtype=torch.float).to(DEVICE)
        action = torch.tensor(np.array(action), dtype=torch.long).to(DEVICE)
        reward = torch.tensor(np.array(reward), dtype=torch.float).to(DEVICE)

        if state.dim() == 3:
            state = state.unsqueeze(0); next_state = next_state.unsqueeze(0)
            action = action.unsqueeze(0); reward = reward.unsqueeze(0); done = (done,)

        pred = self.model(state)
        with torch.no_grad():
            next_best = torch.argmax(self.model(next_state), dim=1)
            next_q = self.target_model(next_state)

        target = pred.clone()
        for i in range(len(done)):
            q_new = reward[i] if done[i] else reward[i] + self.gamma_n * next_q[i][next_best[i]]
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
    def __init__(self, cfg, board_size):
        self.n_games = 0
        self.total_steps = 0
        self.epsilon = 1.0
        self.cfg = cfg
        self.board_size = board_size

        self.n_step_buffer = deque(maxlen=cfg.n_steps)
        self.memory = PrioritizedReplayBuffer(capacity=100_000, priority_cap=1.0)

        self.model = ConvDuelingQNet(board_size, n_actions=4,
                                     ch1=cfg.ch1, ch2=cfg.ch2, hidden=cfg.hidden_dim).to(DEVICE)
        self.target_model = ConvDuelingQNet(board_size, n_actions=4,
                                            ch1=cfg.ch1, ch2=cfg.ch2, hidden=cfg.hidden_dim).to(DEVICE)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()
        self.trainer = ConvQTrainer(self.model, self.target_model,
                                    lr=cfg.lr, n_steps=cfg.n_steps, gamma=cfg.gamma)
        self.epsilon_decay_per_step = None  # set after agent init

    def update_target_network(self, tau=0.005):
        for tp, lp in zip(self.target_model.parameters(), self.model.parameters()):
            tp.data.copy_(tau * lp.data + (1.0 - tau) * tp.data)

    def get_action(self, state):
        self.epsilon = max(self.cfg.epsilon_end,
                           1.0 - self.total_steps * self.epsilon_decay_per_step)
        if random.random() < self.epsilon:
            return random.randint(0, 3)
        s = torch.tensor(state, dtype=torch.float).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            return torch.argmax(self.model(s)).item()

    def remember(self, state, action, reward, next_state, done):
        self.n_step_buffer.append((state, action, reward, next_state, done))
        if len(self.n_step_buffer) < self.cfg.n_steps:
            return
        R, next_s, d = self._n_step_info()
        s0, a0 = self.n_step_buffer[0][:2]
        self.memory.add((s0, a0, R, next_s, d))

    def _n_step_info(self):
        R = 0
        for i, t in enumerate(self.n_step_buffer):
            R += t[2] * (self.cfg.gamma ** i)
            if t[4]:
                return R, t[3], True
        return R, self.n_step_buffer[-1][3], False

    def train(self, batch_size):
        learning_starts = 5_000
        if len(self.memory) < batch_size or self.total_steps < learning_starts:
            return
        batch, indices, weights = self.memory.sample(batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        td_errors = self.trainer.train_step(
            states, actions, rewards, next_states, dones, is_weights=weights
        )
        self.memory.update_priorities(indices, td_errors)


# ============================================================================
# SWEEP TRIAL
# ============================================================================
DEFAULTS = dict(
    lr=1e-4, batch_size=256, train_calls=6, gamma=0.99, n_steps=4,
    epsilon_end=0.05, epsilon_decay_fraction=0.50,
    ch1=32, ch2=64, hidden_dim=256, seed=42,
)

def run_trial():
    wandb.init(project="rl-snake", config=DEFAULTS)
    cfg = wandb.config
    board_size = 10
    num_games = 2000

    seed = getattr(cfg, "seed", 0)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

    agent = ConvDQNAgent(cfg, board_size)
    game = SnakeGame(board_size=board_size, reward_mode="dense")

    # Epsilon decays from 1 → epsilon_end over first (epsilon_decay_fraction × est_total_steps)
    est_total_steps = num_games * board_size * board_size * 0.15
    agent.epsilon_decay_per_step = (1.0 - cfg.epsilon_end) / (
        est_total_steps * cfg.epsilon_decay_fraction
    )

    MAX_STEPS = board_size * board_size * 30
    total_score = 0
    record = 0
    steps_in_game = 0

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
            while agent.n_step_buffer:
                R, next_s, d = agent._n_step_info()
                s0, a0 = agent.n_step_buffer.popleft()[:2]
                agent.memory.add((s0, a0, R, next_s, d))

            game.reset()
            agent.n_games += 1
            steps_in_game = 0

            for _ in range(cfg.train_calls):
                agent.train(cfg.batch_size)

            agent.update_target_network(tau=0.005)

            if score > record:
                record = score

            total_score += score
            mean_score = total_score / agent.n_games

            if agent.n_games % 100 == 0:
                wandb.log({
                    "Game": agent.n_games,
                    "Mean_Score": mean_score,
                    "Record": record,
                    "Epsilon": agent.epsilon,
                })

    print(f"Done. Games={num_games} | Final Mean: {mean_score:.3f} | Best: {record}")
    wandb.finish()


if __name__ == "__main__":
    run_trial()
