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
        self.linear1 = nn.Linear(input_size, hidden_size)

        # Value stream (V(s))
        self.value_stream = nn.Linear(hidden_size, 1)

        # Advantage stream (A(s, a))
        self.advantage_stream = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))

        val = self.value_stream(x)
        adv = self.advantage_stream(x)

        # If input is 1D (unbatched), dimensions are [hidden], output is [out].
        # If input is 2D (batch), dimensions are [batch, hidden], output is [batch, out].

        if adv.dim() == 1:
            # Single item
            return val + (adv - adv.mean())
        else:
            # Batch
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
        self.optimizer.step()
        
        return td_errors


# ============================================================================
# DQN AGENT
# ============================================================================
class DQNAgent:
    def __init__(self, use_icm=False, icm_eta=0.01, icm_lr=0.001):
        self.n_games = 0
        self.epsilon = 0
        self.gamma = 0.9
        self.memory = PrioritizedReplayBuffer(capacity=100_000)
        self.foundation_memory = deque(maxlen=20_000)

        # N-Step Learning
        self.n_steps = 4
        self.n_step_buffer = deque(maxlen=self.n_steps)

        # Dueling Network (Input 14)
        self.model = DuelingQNet(14, 256, 3)
        self.target_model = DuelingQNet(14, 256, 3)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()

        self.trainer = QTrainer(
            self.model,
            self.target_model,
            lr=0.001,
            gamma=self.gamma,
            n_steps=self.n_steps,
        )

        # Intrinsic Curiosity Module
        self.use_icm = use_icm
        self.icm = ICM(state_dim=14, action_dim=3, lr=icm_lr, eta=icm_eta) if use_icm else None

    def update_target_network(self):
        self.target_model.load_state_dict(self.model.state_dict())

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

        flood_straight = self._get_reachable_count(game, pos_straight) / total_cells
        flood_right = self._get_reachable_count(game, pos_right) / total_cells
        flood_left = self._get_reachable_count(game, pos_left) / total_cells

        state = [
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
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            (game.food_position[1] < head[1]) if game.food_position else 0,
            (game.food_position[1] > head[1]) if game.food_position else 0,
            (game.food_position[0] < head[0]) if game.food_position else 0,
            (game.food_position[0] > head[0]) if game.food_position else 0,
            flood_straight,
            flood_right,
            flood_left,
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
        self.memory.add(valid_transition)
        if step_idx <= 50:
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
        # Sample standard buffer (75%) via PER
        standard_sample_size = 750
        if len(self.memory) > standard_sample_size:
            per_batch, per_indices, per_weights = self.memory.sample(standard_sample_size)
        elif len(self.memory) > 0:
            per_batch, per_indices, per_weights = self.memory.sample(len(self.memory))
        else:
            per_batch, per_indices, per_weights = [], [], np.array([])

        # Sample foundation buffer (25%) via uniform random
        foundation_sample_size = 250
        foundation_batch = []
        foundation_weights = []
        if len(self.foundation_memory) > 0:
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
        # Epsilon decay: 1.0 -> 0.05 floor (5% grease prevents death-loop local minima)
        self.epsilon = max(0.01, 1.0 - (self.n_games * 0.0005))
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


def train(use_icm=True, icm_eta=0.01, icm_lr=0.001, board_size=5):
    wandb.init(
        project="rl-snake",
        name=f"DQN_PER_v2_{'ICM' if use_icm else 'Baseline'}_{board_size}x{board_size}",
        config={
            "algorithm": "DQN",
            "board_size": board_size,
            "use_icm": use_icm,
            "icm_eta": icm_eta if use_icm else 0,
            "icm_lr": icm_lr if use_icm else 0,
            "gamma": 0.9,
            "n_steps": 4,
            "learning_rate": 0.001
        }
    )
    
    # Set default x-axis in W&B to Game instead of Step
    wandb.define_metric("Game")
    wandb.define_metric("*", step_metric="Game")

    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = DQNAgent(use_icm=use_icm, icm_eta=icm_eta, icm_lr=icm_lr)
    game = SnakeGame(board_size=board_size)
    
    MAX_STEPS = 500 if board_size <= 5 else (3000 if board_size <= 8 else 5000)
    print(
        f"Starting {'ICM' if use_icm else 'Baseline'} DQN Training (Dueling + 3-Step) on {board_size}x{board_size} Board..."
    )

    steps_in_game = 0
    game_intrinsic_reward = 0.0
    optimizer_decayed = False
    while agent.n_games < 16000:  # Train for 16000 games
        # get old state
        state_old = agent.get_state(game)

        # get move
        final_move = agent.get_action(state_old)

        # perform move and get new state
        action = get_absolute_action(final_move, game)
        _, reward, done, info = game.step(action)
        score = info["score"]

        steps_in_game += 1
        if steps_in_game > MAX_STEPS:
            done = True
            reward = -10  # Penalize timeout

        state_new = agent.get_state(game)

        # Add intrinsic reward if using ICM (train ICM on every experience)
        if use_icm:
            intrinsic_reward = agent.train_icm(state_old, final_move, state_new)
            # Fix 1: MASK TERMINAL INTRINSIC REWARD
            # The ICM generates massive "surprise" on death (game-over physics are unpredictable).
            # Multiplying by (1 - done) strictly zeroes out the intrinsic reward on terminal steps,
            # preventing the agent from learning to suicide for dopamine.
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
                agent.memory.add((state_0, action_0, R, next_s, d))
                agent.train_short_memory(state_0, action_0, R, next_s, d)

            # train long memory, plot result
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()
            steps_in_game = 0  # Reset counter

            # Update Target Network every 50 games
            if agent.n_games % 50 == 0:
                agent.update_target_network()

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
            # At 87.5% through training (game 14000), force a hard 1e-5 floor on LR.
            # This prevents late-game gradient spikes from overwriting the converged policy
            # regardless of whether the mean_score threshold has been hit or not.
            if agent.n_games == 14000:
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
                print(
                    f"Game {agent.n_games} | Score: {score} | Record: {record} | Mean Score: {mean_score:.2f} | Epsilon: {agent.epsilon:.2f} | Intrinsic: {game_intrinsic_reward:.3f}{icm_eta_str}"
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
    args = parser.parse_args()
    
    use_icm = not args.disable_icm
    train(use_icm=use_icm, icm_eta=0.01, icm_lr=0.001, board_size=args.board_size)
