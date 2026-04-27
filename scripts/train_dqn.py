import sys
sys.stdout.reconfigure(line_buffering=True)  # Ensure print() flushes per line when redirected to file

import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import collections
from collections import deque
import matplotlib.pyplot as plt
import argparse
import wandb
from envs.snake_game import SnakeGame  # Reusing the existing game logic
from utils.icm import ICM
from utils.per import PrioritizedReplayBuffer


# ============================================================================
# NEURAL NETWORK
# ============================================================================
class DuelingQNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()

        # Shared feature extraction (two hidden layers)
        self.shared = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )

        # Value stream: dedicated hidden layer + output
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1),
        )

        # Advantage stream: dedicated hidden layer + output
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, output_size),
        )

    def forward(self, x):
        x = self.shared(x)
        val = self.value_stream(x)
        adv = self.advantage_stream(x)

        # If input is 1D (unbatched), dimensions are [hidden], output is [out].
        # If input is 2D (batch), dimensions are [batch, hidden], output is [batch, out].
        if adv.dim() == 1:
            return val + (adv - adv.mean())
        else:
            return val + (adv - adv.mean(dim=1, keepdim=True))

    def save(self, file_name="model.pth"):
        torch.save(self.state_dict(), file_name)


# ============================================================================
# Q-TRAINER
# ============================================================================
class QTrainer:
    def __init__(self, model, target_model, lr, gamma, n_steps=1):
        self.lr = lr
        self.gamma = gamma
        self.n_steps = n_steps
        self.model = model
        self.target_model = target_model  # Target Network
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        # We use reduction='none' so we can apply importance sampling weights per-item
        self.criterion = nn.MSELoss(reduction='none')

    def train_step(self, state, action, reward, next_state, done, is_weights=None):
        state = torch.tensor(np.array(state), dtype=torch.float)
        next_state = torch.tensor(np.array(next_state), dtype=torch.float)
        action = torch.tensor(np.array(action), dtype=torch.long)
        reward = torch.tensor(np.array(reward), dtype=torch.float)
        # (n, x)

        if len(state.shape) == 1:
            # (1, x)
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done,)

        # 1: predicted Q values with current state
        pred = self.model(state)

        # 2. Double DQN Logic
        # Q_new = r + y^n * Q_target(next_state, argmax(Q_local(next_state)))

        # Get best actions from current (local) model
        with torch.no_grad():
            next_pred_local = self.model(next_state)
            next_best_actions = torch.argmax(next_pred_local, dim=1)

            # Get Q-values from target model
            next_pred_target = self.target_model(next_state)

        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                # Use value of best action (chosen by local model) from target model
                # Apply gamma^n_steps
                gamma_n = self.gamma**self.n_steps
                Q_new = (
                    reward[idx]
                    + gamma_n * next_pred_target[idx][next_best_actions[idx]]
                )

            target[idx][torch.argmax(action[idx]).item()] = Q_new

        self.optimizer.zero_grad()
        
        # Compute unreduced loss (batch_size, num_actions)
        loss = self.criterion(target, pred)
        
        # Calculate TD Errors for PER: Since target=pred except at chosen action, sum is exactly the abs error.
        td_errors = torch.abs(target - pred).sum(dim=1).detach().cpu().numpy()
        
        if is_weights is not None:
            is_weights_tensor = torch.tensor(is_weights, dtype=torch.float)
            # Apply IS weights and take the mean across the batch
            loss = (loss.mean(dim=1) * is_weights_tensor).mean()
        else:
            loss = loss.mean()
            
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
        self.optimizer.step()
        
        return td_errors


# ============================================================================
# DQN AGENT
# ============================================================================
class DQNAgent:
    def __init__(
        self,
        use_icm=False,
        use_per=True,
        icm_eta=0.01,
        icm_lr=0.001,
        epsilon_decay_per_game=0.0001,
        epsilon_end=0.02,
        use_priority_cap=True,
        use_foundation_memory=True,
    ):
        self.n_games = 0
        self.epsilon = 1.0
        self.gamma = 0.95
        self.use_per = use_per
        self.total_steps = 0
        self.epsilon_start = 1.0
        self.epsilon_end = epsilon_end
        self.epsilon_decay_per_game = epsilon_decay_per_game
        self.use_foundation_memory = use_foundation_memory
        if use_per:
            # priority_cap=inf disables the hard ceiling that otherwise prevents
            # high-TD-error transitions (e.g. ICM-inflated terminal steps) from dominating sampling.
            cap = 1.0 if use_priority_cap else float("inf")
            self.memory = PrioritizedReplayBuffer(capacity=100_000, priority_cap=cap)
        else:
            self.memory = deque(maxlen=100_000)
        self.foundation_memory = deque(maxlen=20_000) if use_foundation_memory else None

        # N-Step Learning
        self.n_steps = 4
        self.n_step_buffer = deque(maxlen=self.n_steps)

        # Dueling Network (Input 24)
        self.model = DuelingQNet(24, 256, 3)
        self.target_model = DuelingQNet(24, 256, 3)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()

        self.trainer = QTrainer(
            self.model,
            self.target_model,
            lr=0.0003,
            gamma=self.gamma,
            n_steps=self.n_steps,
        )

        # Intrinsic Curiosity Module
        self.use_icm = use_icm
        # state_dim must match the dim returned by get_state() (24); the DuelingQNet above also takes 24.
        self.icm = ICM(state_dim=24, action_dim=3, lr=icm_lr, eta=icm_eta) if use_icm else None

    def update_target_network(self, tau=0.005):
        for target_param, local_param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def _get_reachable_count(self, game, start_pos):
        """BFS to count reachable free cells from start_pos (Borrowed from QLearningAgentV4)"""
        obstacles = set(game.snake_position)
        if start_pos in obstacles or game._check_collision(start_pos, False):
            return 0

        visited = set([start_pos])
        queue = collections.deque([start_pos])
        count = 0
        max_search = max(len(game.snake_position) * 3, 20)

        while queue and count < max_search:
            curr = queue.popleft()
            count += 1
            cx, cy = curr
            neighbors = [(cx - 1, cy), (cx + 1, cy), (cx, cy - 1), (cx, cy + 1)]
            for nx, ny in neighbors:
                if (
                    0 <= nx < game.board_size
                    and 0 <= ny < game.board_size
                    and (nx, ny) not in obstacles
                    and (nx, ny) not in visited
                ):
                    visited.add((nx, ny))
                    queue.append((nx, ny))
        return count

    def get_state(self, game):
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
            idx = 3  # dir_l

        pos_straight = clock_wise_points[idx]
        pos_right = clock_wise_points[(idx + 1) % 4]
        pos_left = clock_wise_points[(idx - 1) % 4]

        total_cells = game.board_size * game.board_size
        b = game.board_size

        flood_straight = self._get_reachable_count(game, pos_straight) / total_cells
        flood_right = self._get_reachable_count(game, pos_right) / total_cells
        flood_left = self._get_reachable_count(game, pos_left) / total_cells

        # Tail direction (where the tail tip is relative to head)
        tail = game.snake_position[0]
        tail_up   = int(tail[0] < head[0])
        tail_down  = int(tail[0] > head[0])
        tail_left  = int(tail[1] < head[1])
        tail_right = int(tail[1] > head[1])

        # Normalized snake length
        snake_length_norm = len(game.snake_position) / total_cells

        # Normalized Manhattan distance to food
        if game.food_position:
            food_dist_norm = (abs(head[0] - game.food_position[0]) + abs(head[1] - game.food_position[1])) / (2 * (b - 1))
        else:
            food_dist_norm = 0.0

        # Normalized distances to each wall
        wall_up    = head[0] / (b - 1)
        wall_down  = (b - 1 - head[0]) / (b - 1)
        wall_left  = head[1] / (b - 1)
        wall_right = (b - 1 - head[1]) / (b - 1)

        state = [
            # Danger (3)
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
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            # Food direction (4)
            (game.food_position[1] < head[1]) if game.food_position else 0,
            (game.food_position[1] > head[1]) if game.food_position else 0,
            (game.food_position[0] < head[0]) if game.food_position else 0,
            (game.food_position[0] > head[0]) if game.food_position else 0,
            # Flood-fill open space (3)
            flood_straight,
            flood_right,
            flood_left,
            # Tail direction (4)
            tail_up,
            tail_down,
            tail_left,
            tail_right,
            # Snake length normalized (1)
            snake_length_norm,
            # Food distance normalized (1)
            food_dist_norm,
            # Wall distances normalized (4)
            wall_up,
            wall_down,
            wall_left,
            wall_right,
        ]

        return np.array(state, dtype=float)

    def remember(self, state, action, reward, next_state, done, step_idx=0):
        # Apply N-Step Buffer logic
        self.n_step_buffer.append((state, action, reward, next_state, done))

        if len(self.n_step_buffer) < self.n_steps:
            return  # Buffer not full yet

        # Compute N-step Reward
        # R = r_0 + gamma*r_1 + gamma^2*r_2 ...
        R, next_s, d = self._get_n_step_info()
        state_0, action_0 = self.n_step_buffer[0][:2]

        valid_transition = (state_0, action_0, R, next_s, d)
        if self.use_per:
            self.memory.add(valid_transition)
        else:
            self.memory.append(valid_transition)
        if self.foundation_memory is not None and step_idx <= 50:
            self.foundation_memory.append(valid_transition)

        return valid_transition  # Return computed transition for online training

    def _get_n_step_info(self):
        R = 0
        for i, transition in enumerate(self.n_step_buffer):
            r = transition[2]
            R += r * (self.gamma**i)
            if transition[4]:  # done is True
                return (
                    R,
                    transition[3],
                    True,
                )  # next_state is terminal state of this step

        # If no done, next_state is the next_state of the LAST item in buffer
        return R, self.n_step_buffer[-1][3], False

    def train_long_memory(self):
        if not self.use_per:
            # Simple uniform replay — no IS weights, no priority updates
            batch_size = 1000
            if len(self.memory) < batch_size:
                return
            mini_batch = random.sample(self.memory, batch_size)
            states, actions, rewards, next_states, dones = zip(*mini_batch)
            self.trainer.train_step(states, actions, rewards, next_states, dones, is_weights=None)
            return

        # Sample standard buffer via PER. When foundation memory is enabled, it takes 25% of the
        # batch; when disabled, PER takes the entire 1000-sample batch (= textbook PER).
        per_target = 750 if self.use_foundation_memory else 1000
        if len(self.memory) > per_target:
            per_batch, per_indices, per_weights = self.memory.sample(per_target)
        elif len(self.memory) > 0:
            per_batch, per_indices, per_weights = self.memory.sample(len(self.memory))
        else:
            per_batch, per_indices, per_weights = [], [], np.array([])

        # Sample foundation buffer (25%) via uniform random — disabled cleanly when use_foundation_memory=False
        foundation_sample_size = 250
        foundation_batch = []
        foundation_weights = []
        if self.foundation_memory is not None and len(self.foundation_memory) > 0:
            if len(self.foundation_memory) > foundation_sample_size:
                foundation_batch = random.sample(self.foundation_memory, foundation_sample_size)
            else:
                foundation_batch = list(self.foundation_memory)
            # Foundation weights are 1.0 since it's uniform
            foundation_weights = np.ones(len(foundation_batch), dtype=np.float32)

        mini_batch = per_batch + foundation_batch
        
        if not mini_batch:
            return
            
        is_weights = np.concatenate([per_weights, foundation_weights]) if len(per_batch) > 0 else np.array(foundation_weights)

        states, actions, rewards, next_states, dones = zip(*mini_batch)
        
        # Train and get TD errors
        td_errors = self.trainer.train_step(states, actions, rewards, next_states, dones, is_weights=is_weights)
        
        # We only update the PER tree for the indices that actually came from the PER buffer
        if len(per_indices) > 0:
            per_td_errors = td_errors[:len(per_indices)]
            self.memory.update_priorities(per_indices, per_td_errors)

    def train_short_memory(self, state, action, reward, next_state, done):
        # Online 1-step training for immediate feedback (latency reduction)
        self.trainer.train_step(
            [state], [action], [reward], [next_state], [done], is_weights=None
        )

    def get_action(self, state):
        # Linear game-based decay. Rate scales with num_games so short pilots floor early
        # enough to leave room for policy consolidation.
        self.epsilon = max(self.epsilon_end, 1.0 - (self.n_games * self.epsilon_decay_per_game))
        final_move = [0, 0, 0]

        if random.random() < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move

    def train_icm(self, state, action, next_state):
        """Train the ICM on the given experience"""
        if not self.use_icm:
            return 0.0

        # Convert action to one-hot for ICM
        action_one_hot = np.zeros(3)
        action_one_hot[np.argmax(action)] = 1

        # Train ICM and return intrinsic reward
        intrinsic_reward = self.icm.train_step(state, action_one_hot, next_state)
        return intrinsic_reward


def get_absolute_action(move, game):
    # move is [straight, right, left]
    # We need to convert this relative move to absolute action (0: up, 1: right, 2: down, 3: left)

    # Determine current direction
    head = game.snake_position[-1]
    if len(game.snake_position) > 1:
        neck = game.snake_position[-2]
        if head[0] < neck[0]:
            current_dir = 0  # UP
        elif head[0] > neck[0]:
            current_dir = 2  # DOWN
        elif head[1] < neck[1]:
            current_dir = 3  # LEFT
        else:
            current_dir = 1  # RIGHT
    else:
        # Default start direction (e.g., right) or infer from last action if passed
        current_dir = 1  # Assume growing right initially

    # Clockwise order: [UP, RIGHT, DOWN, LEFT] -> [0, 1, 2, 3]
    clock_wise = [0, 1, 2, 3]
    idx = clock_wise.index(current_dir)

    if np.array_equal(move, [1, 0, 0]):  # Straight
        new_dir = clock_wise[idx]
    elif np.array_equal(move, [0, 1, 0]):  # Right turn
        new_dir = clock_wise[(idx + 1) % 4]
    else:  # [0, 0, 1] Left turn
        new_dir = clock_wise[(idx - 1) % 4]

    return new_dir


def train(
    use_icm=True,
    use_per=True,
    icm_eta=0.01,
    icm_lr=0.001,
    board_size=5,
    num_games=16000,
    mask_terminal_intrinsic=True,
    use_priority_cap=True,
    use_foundation_memory=True,
    reward_mode="dense",
    seed=None,
    run_tag=None,
):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    if run_tag is None:
        run_tag = (
            f"DQN_{'ICM' if use_icm else 'Baseline'}"
            f"_{'PER' if use_per else 'noPER'}"
            f"_{'masked' if mask_terminal_intrinsic else 'unmasked'}"
            f"_{board_size}x{board_size}"
            f"_g{num_games}"
            + (f"_seed{seed}" if seed is not None else "")
        )

    wandb.init(
        project="rl-snake",
        name=run_tag,
        config={
            "algorithm": "DQN",
            "board_size": board_size,
            "use_icm": use_icm,
            "use_per": use_per,
            "mask_terminal_intrinsic": mask_terminal_intrinsic,
            "use_priority_cap": use_priority_cap,
            "use_foundation_memory": use_foundation_memory,
            "reward_mode": reward_mode,
            "icm_eta": icm_eta if use_icm else 0,
            "icm_lr": icm_lr if use_icm else 0,
            "gamma": 0.95,
            "n_steps": 4,
            "learning_rate": 0.001,
            "num_games": num_games,
            "seed": seed,
        }
    )
    
    # Set default x-axis in W&B to Game instead of Step
    wandb.define_metric("Game")
    wandb.define_metric("*", step_metric="Game")

    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    # Scale epsilon decay so ε reaches its floor at ~50% of training, leaving the rest for consolidation.
    # For num_games=16000 this evaluates to ~6e-5 (≈ original 0.0001 schedule, slightly slower);
    # for num_games=4000 it evaluates to ~2.45e-4 (≈ ε floors at game 4000, post-decay zone is small but nonzero).
    epsilon_decay_per_game = max((1.0 - 0.02) / max(num_games * 0.5, 1.0), 1e-6)
    agent = DQNAgent(
        use_icm=use_icm,
        use_per=use_per,
        icm_eta=icm_eta,
        icm_lr=icm_lr,
        epsilon_decay_per_game=epsilon_decay_per_game,
        use_priority_cap=use_priority_cap,
        use_foundation_memory=use_foundation_memory,
    )
    game = SnakeGame(board_size=board_size, reward_mode=reward_mode)
    
    MAX_STEPS = 500 if board_size <= 5 else (3000 if board_size <= 8 else 5000)
    print(
        f"Starting {'ICM' if use_icm else 'Baseline'} DQN Training (Dueling + 3-Step) on {board_size}x{board_size} Board..."
    )

    steps_in_game = 0
    game_intrinsic_reward = 0.0
    optimizer_decayed = False
    # Exploration coverage: unique (head_x, head_y, food_x, food_y) tuples visited.
    # Cumulative set saturates at board_size**4 (4096 for 8x8). Windowed set is reset every 100 games.
    visited_cumulative = set()
    visited_window = set()
    while agent.n_games < num_games:  # Train for num_games games
        # get old state
        state_old = agent.get_state(game)

        # get move
        final_move = agent.get_action(state_old)

        # perform move and get new state
        action = get_absolute_action(final_move, game)
        _, reward, done, info = game.step(action)
        score = info["score"]

        # Coverage: log the post-step (head, food) pair the agent transitioned into.
        if game.snake_position and game.food_position:
            head = game.snake_position[-1]
            state_key = (head[0], head[1], game.food_position[0], game.food_position[1])
            visited_cumulative.add(state_key)
            visited_window.add(state_key)

        steps_in_game += 1
        agent.total_steps += 1

        if steps_in_game > MAX_STEPS:
            done = True
            reward = -1  # Penalize timeout

        state_new = agent.get_state(game)

        # Add intrinsic reward if using ICM (train ICM on every experience)
        if use_icm:
            intrinsic_reward = agent.train_icm(state_old, final_move, state_new)
            # Fix 1: MASK TERMINAL INTRINSIC REWARD
            # The ICM generates massive "surprise" on death (game-over physics are unpredictable).
            # Multiplying by (1 - done) strictly zeroes out the intrinsic reward on terminal steps,
            # preventing the agent from learning to suicide for dopamine.
            # Disabling this mask reproduces the "poisoned" PER + ICM baseline.
            if mask_terminal_intrinsic:
                intrinsic_reward = intrinsic_reward * (1.0 - float(done))
            reward += intrinsic_reward
            game_intrinsic_reward += intrinsic_reward

        # remember returns None if buffer not full, or tuple if N-step ready
        n_step_transition = agent.remember(
            state_old, final_move, reward, state_new, done, steps_in_game
        )

        if n_step_transition:
            # Train short memory with the valid N-step transition
            s, a, r, ns, d = n_step_transition
            agent.train_short_memory(s, a, r, ns, d)

        if done:
            # FLUSH BUFFER
            # We must process remaining items in buffer with their truncated returns
            while len(agent.n_step_buffer) > 0:
                R, next_s, d = agent._get_n_step_info()
                state_0, action_0 = agent.n_step_buffer.popleft()[:2]
                if agent.use_per:
                    agent.memory.add((state_0, action_0, R, next_s, d))
                else:
                    agent.memory.append((state_0, action_0, R, next_s, d))
                agent.train_short_memory(state_0, action_0, R, next_s, d)

            # train long memory, plot result
            game.reset()
            agent.n_games += 1
            for _ in range(3):
                agent.train_long_memory()
            steps_in_game = 0  # Reset counter

            # Soft Polyak target update every game (τ=0.005)
            agent.update_target_network(tau=0.005)

            if score > record:
                record = score
                agent.model.save()
                if use_icm and agent.icm is not None:
                    agent.icm.save_models("model_icm")

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)

            if use_icm and agent.icm is not None:
                # Phase out curiosity when the agent is consistently performing well
                if mean_score > 15.0:
                    agent.icm.eta = agent.icm.eta * 0.99
                    
                    if not optimizer_decayed:
                        for param_group in agent.trainer.optimizer.param_groups:
                            param_group['lr'] = param_group['lr'] * 0.1
                        for param_group in agent.icm.forward_optimizer.param_groups:
                            param_group['lr'] = param_group['lr'] * 0.1
                        for param_group in agent.icm.inverse_optimizer.param_groups:
                            param_group['lr'] = param_group['lr'] * 0.1
                        optimizer_decayed = True
            
            # Fix 3: FINAL LR COOLDOWN
            # At 87.5% through training, force a hard 1e-5 floor on LR.
            # This prevents late-game gradient spikes from overwriting the converged policy
            # regardless of whether the mean_score threshold has been hit or not.
            lr_cooldown_game = int(num_games * 0.875)
            if agent.n_games == lr_cooldown_game:
                for param_group in agent.trainer.optimizer.param_groups:
                    param_group['lr'] = 1e-5
                if use_icm and agent.icm is not None:
                    for param_group in agent.icm.forward_optimizer.param_groups:
                        param_group['lr'] = 1e-5
                    for param_group in agent.icm.inverse_optimizer.param_groups:
                        param_group['lr'] = 1e-5
                print(f"[Game {agent.n_games}] Final LR cooldown applied → 1e-5")

            if agent.n_games % 100 == 0:
                icm_eta_str = f" | Eta: {agent.icm.eta:.5f}" if (use_icm and agent.icm is not None) else ""
                # Replay-buffer composition diagnostics (PER only).
                buf_term = sample_term = oversample = 0.0
                if use_per and hasattr(agent.memory, "buffer_terminal_frac"):
                    buf_term = agent.memory.buffer_terminal_frac()
                    sample_term = agent.memory.last_sample_terminal_frac()
                    oversample = (sample_term / buf_term) if buf_term > 0 else 0.0
                buf_str = f" | BufTerm: {buf_term:.3f} | SampTerm: {sample_term:.3f} | Oversample: {oversample:.2f}x" if use_per else ""
                print(
                    f"Game {agent.n_games} | Score: {score} | Record: {record} | Mean Score: {mean_score:.2f} | Epsilon: {agent.epsilon:.2f} | Intrinsic: {game_intrinsic_reward:.3f}{icm_eta_str}{buf_str}"
                )

                log_metrics = {
                    "Game": agent.n_games,
                    "Score": score,
                    "Record": record,
                    "Mean_Score": mean_score,
                    "Epsilon": agent.epsilon,
                }
                if use_icm:
                    log_metrics["Intrinsic_Reward"] = game_intrinsic_reward
                    log_metrics["Eta"] = agent.icm.eta
                if use_per:
                    log_metrics["Buf_Terminal_Frac"] = buf_term
                    log_metrics["Sample_Terminal_Frac"] = sample_term
                    log_metrics["Terminal_Oversample_Factor"] = oversample
                # Coverage diagnostics
                cov_cum = len(visited_cumulative)
                cov_win = len(visited_window)
                cov_max = board_size ** 4
                print(
                    f"[coverage] Game {agent.n_games} | unique_states_cumulative: {cov_cum}/{cov_max} ({100*cov_cum/cov_max:.1f}%) | last100_unique: {cov_win}"
                )
                log_metrics["Coverage_Cumulative"] = cov_cum
                log_metrics["Coverage_Window100"] = cov_win
                log_metrics["Coverage_Cumulative_Frac"] = cov_cum / cov_max
                visited_window = set()
                wandb.log(log_metrics)
            
            game_intrinsic_reward = 0.0

    print("\nTraining Complete!")
    print(f"Best Score: {record}")
    print(f"Final Mean Score: {mean_score:.2f}")
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--board_size', type=int, default=5, help='Size of the board (e.g., 5 for 5x5)')
    parser.add_argument('--disable_icm', action='store_true', help='Disable the Intrinsic Curiosity Module for pure baseline runs')
    parser.add_argument('--disable_per', action='store_true', help='Use simple uniform replay instead of PER')
    parser.add_argument('--disable_terminal_mask', action='store_true', help='Do NOT mask intrinsic reward at terminal steps (reproduces the poisoned PER+ICM baseline)')
    parser.add_argument('--disable_priority_cap', action='store_true', help='Disable PER hard ceiling on TD-error priorities (allows high-TD events to dominate sampling)')
    parser.add_argument('--disable_foundation_memory', action='store_true', help='Use textbook PER without the 75/25 foundation-memory blend')
    parser.add_argument('--num_games', type=int, default=16000, help='Number of games to train for')
    parser.add_argument('--icm_eta', type=float, default=0.01, help='ICM intrinsic reward scaling (default 0.01; raise for clearer ICM signal)')
    parser.add_argument('--reward_mode', type=str, default='dense', choices=['dense', 'sparse', 'pure_sparse'], help='dense = full shaping (default); sparse = food/death/step penalty only; pure_sparse = food/death only, zero step penalty')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility')
    parser.add_argument('--run_tag', type=str, default=None, help='Override wandb run name (useful for ablation logging)')
    args = parser.parse_args()

    use_icm = not args.disable_icm
    use_per = not args.disable_per
    mask_terminal = not args.disable_terminal_mask
    use_priority_cap = not args.disable_priority_cap
    use_foundation_memory = not args.disable_foundation_memory
    train(
        use_icm=use_icm,
        use_per=use_per,
        icm_eta=args.icm_eta,
        icm_lr=0.001,
        board_size=args.board_size,
        num_games=args.num_games,
        mask_terminal_intrinsic=mask_terminal,
        use_priority_cap=use_priority_cap,
        use_foundation_memory=use_foundation_memory,
        reward_mode=args.reward_mode,
        seed=args.seed,
        run_tag=args.run_tag,
    )
