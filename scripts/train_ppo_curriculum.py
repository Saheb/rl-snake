"""
================================================================================
CURRICULUM LEARNING: IMITATION LEARNING (5x5) → TRANSFER (8x8)
================================================================================

Our strategy:
1. Use Imitation Learning (REINFORCE -> PPO) to master 5x5 board.
2. Transfer the learned policy to 8x8 board.
3. Continue fine-uning on 8x8.

This combines the best of all worlds:
- Imitation Learning solves the "bootstrap trap"
- Curriculum Learning eases the difficulty curve
- PPO provides stable policy updates

================================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
import random
import collections
from collections import deque
from enum import Enum
import copy

# ============================================================================
# GAME ENVIRONMENT
# ============================================================================
class GameState(Enum):
    PLAYING = 1
    GAME_OVER = 2
    WIN = 3

class SnakeGame:
    def __init__(self, board_size=5):
        self.board_size = board_size
        self.reset()

    def reset(self):
        self.board = [['.' for _ in range(self.board_size)] for _ in range(self.board_size)]
        start_x = self.board_size // 2
        start_y = self.board_size // 2
        self.snake_position = deque([(start_x, start_y)])
        self.food_position = self._pick_food_position()
        self.game_state = GameState.PLAYING
        self.score = 0
        self.steps = 0
        self.head_history = deque(maxlen=8)
        self._update_board_snake()
        self._update_board_food()
        return self.get_state()

    def _pick_food_position(self):
        empty_cells = []
        for i in range(self.board_size):
            for j in range(self.board_size):
                if (i, j) not in self.snake_position:
                    empty_cells.append((i, j))
        return random.choice(empty_cells) if empty_cells else None

    def _clear_board(self):
        for i in range(self.board_size):
            for j in range(self.board_size):
                self.board[i][j] = '.'
    
    def _update_board_snake(self):
        for i, (x, y) in enumerate(self.snake_position):
            self.board[x][y] = 'H' if i == len(self.snake_position) - 1 else 'S'
    
    def _update_board_food(self):
        if self.food_position:
            self.board[self.food_position[0]][self.food_position[1]] = 'F'

    def get_state(self):
        return [row[:] for row in self.board]
    
    def _check_collision(self, position, will_eat_food=False):
        x, y = position
        if x < 0 or x >= self.board_size or y < 0 or y >= self.board_size:
            return True
        snake_body = list(self.snake_position)[:-1] if not will_eat_food else list(self.snake_position)
        if position in snake_body:
            return True
        return False

    def step(self, action):
        self.steps += 1
        head_x, head_y = self.snake_position[-1]
        old_dist = abs(head_x - self.food_position[0]) + abs(head_y - self.food_position[1])
        new_head = self._move_snake(action)
        will_eat_food = (new_head == self.food_position)

        is_biting_neck = new_head in list(self.snake_position)[-3:]
        is_waffling = new_head in self.head_history
        
        if self._check_collision(new_head, will_eat_food):
            self.game_state = GameState.GAME_OVER
            reward = -10
            return self.get_state(), reward, True, {"score": self.score}
        
        if will_eat_food:
            reward = 10
            self.score += 1
            self.snake_position.append(new_head)
            self.food_position = self._pick_food_position()
            if not self.food_position:
                self.game_state = GameState.WIN
                reward = 50
                done = True
            else:
                done = False
        else:
            self.snake_position.append(new_head)
            self.snake_position.popleft()
            reward = -0.01
            new_dist = abs(new_head[0] - self.food_position[0]) + abs(new_head[1] - self.food_position[1])
            if new_dist < old_dist:
                reward += 0.1 / max(1, len(self.snake_position))
            else:
                reward -= 0.15 / max(1, len(self.snake_position))
            done = False

            if is_biting_neck:
                reward -= 0.4
            if is_waffling:
                reward -= 0.5
            self.head_history.append(new_head)
        
        self._clear_board()
        self._update_board_snake()
        self._update_board_food()

        if self.score >= self.board_size * self.board_size - 1:
            self.game_state = GameState.WIN
            reward = 50
            done = True
        
        return self.get_state(), reward, done, {"score": self.score}
    
    def _move_snake(self, action):
        head_x, head_y = self.snake_position[-1]
        if action == 0:   return (head_x - 1, head_y)
        elif action == 1: return (head_x + 1, head_y)
        elif action == 2: return (head_x, head_y - 1)
        elif action == 3: return (head_x, head_y + 1)
        return (head_x, head_y)


# ============================================================================
# 🎓 EXPERT COMPONENT (REINFORCE)
# ============================================================================

class PolicyNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=-1)

class ExpertCollector:
    def __init__(self, board_size=5):
        self.board_size = board_size
        self.game = SnakeGame(board_size=board_size)
        
    def _get_reachable_count(self, game, start_pos):
        obstacles = set(game.snake_position)
        if start_pos in obstacles or game._check_collision(start_pos, False):
            return 0
        visited = set([start_pos])
        queue = collections.deque([start_pos])
        count = 0
        max_search = len(game.snake_position) * 3
        while queue and count < max_search:
            curr = queue.popleft()
            count += 1
            cx, cy = curr
            for nx, ny in [(cx-1, cy), (cx+1, cy), (cx, cy-1), (cx, cy+1)]:
                if (0 <= nx < game.board_size and 0 <= ny < game.board_size and
                    (nx, ny) not in obstacles and (nx, ny) not in visited):
                    visited.add((nx, ny))
                    queue.append((nx, ny))
        return count
        
    def get_state_features(self, game):
        head = game.snake_position[-1]
        point_l = (head[0], head[1] - 1)
        point_r = (head[0], head[1] + 1)
        point_u = (head[0] - 1, head[1])
        point_d = (head[0] + 1, head[1])
        
        if len(game.snake_position) > 1:
            neck = game.snake_position[-2]
            if head[0] < neck[0]:   dir_u, dir_r, dir_d, dir_l = True, False, False, False
            elif head[0] > neck[0]: dir_u, dir_r, dir_d, dir_l = False, False, True, False
            elif head[1] < neck[1]: dir_u, dir_r, dir_d, dir_l = False, False, False, True
            else:                   dir_u, dir_r, dir_d, dir_l = False, True, False, False
        else:
            dir_u, dir_r, dir_d, dir_l = False, True, False, False

        clock_wise_points = [point_u, point_r, point_d, point_l]
        if dir_u:   idx = 0
        elif dir_r: idx = 1
        elif dir_d: idx = 2
        else:       idx = 3
        
        pos_straight = clock_wise_points[idx]
        pos_right = clock_wise_points[(idx + 1) % 4]
        pos_left = clock_wise_points[(idx - 1) % 4]
        
        total_cells = game.board_size * game.board_size
        flood_straight = self._get_reachable_count(game, pos_straight) / total_cells
        flood_right = self._get_reachable_count(game, pos_right) / total_cells
        flood_left = self._get_reachable_count(game, pos_left) / total_cells

        state = [
            (dir_r and game._check_collision(point_r)) or 
            (dir_l and game._check_collision(point_l)) or 
            (dir_u and game._check_collision(point_u)) or 
            (dir_d and game._check_collision(point_d)),
            (dir_u and game._check_collision(point_r)) or 
            (dir_d and game._check_collision(point_l)) or 
            (dir_l and game._check_collision(point_u)) or 
            (dir_r and game._check_collision(point_d)),
            (dir_d and game._check_collision(point_r)) or 
            (dir_u and game._check_collision(point_l)) or 
            (dir_r and game._check_collision(point_u)) or 
            (dir_l and game._check_collision(point_d)),
            dir_l, dir_r, dir_u, dir_d,
            (game.food_position[1] < head[1]) if game.food_position else 0,
            (game.food_position[1] > head[1]) if game.food_position else 0,
            (game.food_position[0] < head[0]) if game.food_position else 0,
            (game.food_position[0] > head[0]) if game.food_position else 0,
            flood_straight, flood_right, flood_left
        ]
        return np.array(state, dtype=float)
    
    def _get_absolute_action(self, relative_move, game):
        """Convert relative action to absolute"""
        head = game.snake_position[-1]
        if len(game.snake_position) > 1:
            neck = game.snake_position[-2]
            if head[0] < neck[0]:   direction = 0
            elif head[0] > neck[0]: direction = 1
            elif head[1] < neck[1]: direction = 2
            else:                   direction = 3
        else:
            direction = 3
        right_turn = {0: 3, 3: 1, 1: 2, 2: 0}
        left_turn = {0: 2, 2: 1, 1: 3, 3: 0}
        if relative_move == [1, 0, 0]: return direction
        elif relative_move == [0, 1, 0]: return right_turn[direction]
        elif relative_move == [0, 0, 1]: return left_turn[direction]
        return direction

    def collect_with_trained_reinforce(self, num_episodes=50, min_score=3, gamma=0.99):
        print("\n" + "=" * 60)
        print(f"🎓 PHASE 1a: Training REINFORCE Expert on {self.board_size}x{self.board_size}")
        print("=" * 60)
        
        policy = PolicyNetwork(14, 256, 3)
        optimizer = optim.Adam(policy.parameters(), lr=0.001)
        expert_data = []
        
        # Train expert quickly
        print("\nStep 1: Training expert policy...")
        baseline = 0
        baseline_decay = 0.99
        
        for episode in range(5000):  # Increased to 5000 to ensure convergence
            trajectory = []
            log_probs = []
            rewards = []
            entropies = []
            
            self.game.reset()
            done = False
            steps = 0
            MAX_STEPS = 500
            
            while not done and steps < MAX_STEPS:
                state = self.get_state_features(self.game)
                state_tensor = torch.tensor(state, dtype=torch.float).unsqueeze(0)
                probs = policy(state_tensor)
                dist = Categorical(probs)
                action = dist.sample()
                log_prob = dist.log_prob(action)
                entropy = dist.entropy()
                log_probs.append(log_prob)
                entropies.append(entropy)
                
                action_onehot = [0, 0, 0]
                action_onehot[action.item()] = 1
                abs_action = self._get_absolute_action(action_onehot, self.game)
                
                trajectory.append((state.copy(), action.item()))
                _, reward, done, info = self.game.step(abs_action)
                rewards.append(reward)
                steps += 1
                if steps >= MAX_STEPS: done, reward = True, -10
            
            if len(rewards) > 0:
                returns = []
                G = 0
                for r in reversed(rewards):
                    G = r + gamma * G
                    returns.insert(0, G)
                returns = torch.tensor(returns, dtype=torch.float)
                
                episode_return = returns[0].item()
                baseline = baseline_decay * baseline + (1 - baseline_decay) * episode_return
                advantages = returns - baseline
                if len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                
                policy_loss = 0
                entropy_bonus = 0
                for lp, adv, ent in zip(log_probs, advantages, entropies):
                    policy_loss += -lp * adv
                    entropy_bonus += ent
                loss = policy_loss - 0.05 * entropy_bonus
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
                optimizer.step()
            
            if (episode + 1) % 500 == 0:
                print(f"  Expert training episode {episode + 1}/2000, last score: {info['score']}")

        # Collect demos
        print("\nStep 2: Collecting expert demos...")
        collected = 0
        attempts = 0
        while collected < num_episodes and attempts < num_episodes * 10:
            attempts += 1
            trajectory = []
            rewards = []
            self.game.reset()
            done, steps = False, 0
            while not done and steps < 500:
                state = self.get_state_features(self.game)
                state_tensor = torch.tensor(state, dtype=torch.float).unsqueeze(0)
                with torch.no_grad(): probs = policy(state_tensor)
                dist = Categorical(probs)
                action = dist.sample()
                trajectory.append((state.copy(), action.item()))
                action_onehot = [0, 0, 0]
                action_onehot[action.item()] = 1
                abs_action = self._get_absolute_action(action_onehot, self.game)
                _, reward, done, info = self.game.step(abs_action)
                rewards.append(reward)
                steps += 1
            
            if info['score'] >= min_score:
                collected += 1
                returns = []
                G = 0
                for r in reversed(rewards):
                    G = r + gamma * G
                    returns.insert(0, G)
                for (state, action), ret in zip(trajectory, returns):
                    expert_data.append({'state': state, 'action': action, 'return': ret})
        
        print(f"✅ Collected {len(expert_data)} expert pairs from {collected} episodes.")
        return expert_data, policy

# ============================================================================
# 🎓 PPO COMPONENTS
# ============================================================================

class ActorCritic(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU()
        )
        self.actor = nn.Linear(hidden_size // 2, output_size)
        self.critic = nn.Linear(hidden_size // 2, 1)
    
    def forward(self, x):
        shared = self.shared(x)
        return F.softmax(self.actor(shared), dim=-1), self.critic(shared)
    
    def get_action(self, state):
        probs, value = self.forward(state)
        dist = Categorical(probs)
        action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value

class PPOAgent:
    def __init__(self, network=None, lr=0.0003, gamma=0.99, n_epochs=4):
        self.gamma, self.n_epochs = gamma, n_epochs
        self.clip_epsilon, self.entropy_coef = 0.2, 0.01
        self.n_games = 0
        self.states, self.actions, self.log_probs = [], [], []
        self.rewards, self.values, self.dones = [], [], []
        
        if network:
            print("🎓 Using Pretrained Network")
            self.network = network
        else:
            self.network = ActorCritic(14, 256, 3)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)

    def _get_reachable_count(self, game, start_pos):
        # ... copy of previously defined method ...
        # (Assuming ExpertCollector has the logic, we need it here too)
        # For brevity, I'll copy the logic inside get_state
        pass

    def get_state(self, game):
        # Helper for BFS inside get_state
        def bfs_count(game, start_pos):
            obstacles = set(game.snake_position)
            if start_pos in obstacles or game._check_collision(start_pos, False): return 0
            visited, queue = set([start_pos]), deque([start_pos])
            count, max_search = 0, len(game.snake_position) * 3
            while queue and count < max_search:
                curr = queue.popleft()
                count += 1
                cx, cy = curr
                for nx, ny in [(cx-1, cy), (cx+1, cy), (cx, cy-1), (cx, cy+1)]:
                    if (0 <= nx < game.board_size and 0 <= ny < game.board_size and 
                        (nx, ny) not in obstacles and (nx, ny) not in visited):
                        visited.add((nx, ny))
                        queue.append((nx, ny))
            return count

        head = game.snake_position[-1]
        point_l = (head[0], head[1] - 1)
        point_r = (head[0], head[1] + 1)
        point_u = (head[0] - 1, head[1])
        point_d = (head[0] + 1, head[1])
        
        if len(game.snake_position) > 1:
            neck = game.snake_position[-2]
            if head[0] < neck[0]:   dir_u, dir_r, dir_d, dir_l = True, False, False, False
            elif head[0] > neck[0]: dir_u, dir_r, dir_d, dir_l = False, False, True, False
            elif head[1] < neck[1]: dir_u, dir_r, dir_d, dir_l = False, False, False, True
            else:                   dir_u, dir_r, dir_d, dir_l = False, True, False, False
        else:
            dir_u, dir_r, dir_d, dir_l = False, True, False, False

        clock_wise_points = [point_u, point_r, point_d, point_l]
        if dir_u:   idx = 0
        elif dir_r: idx = 1
        elif dir_d: idx = 2
        else:       idx = 3
        
        pos_straight = clock_wise_points[idx]
        pos_right = clock_wise_points[(idx + 1) % 4]
        pos_left = clock_wise_points[(idx - 1) % 4]
        
        total_cells = game.board_size * game.board_size
        flood_straight = bfs_count(game, pos_straight) / total_cells
        flood_right = bfs_count(game, pos_right) / total_cells
        flood_left = bfs_count(game, pos_left) / total_cells

        state = [
            (dir_r and game._check_collision(point_r)) or (dir_l and game._check_collision(point_l)) or (dir_u and game._check_collision(point_u)) or (dir_d and game._check_collision(point_d)),
            (dir_u and game._check_collision(point_r)) or (dir_d and game._check_collision(point_l)) or (dir_l and game._check_collision(point_u)) or (dir_r and game._check_collision(point_d)),
            (dir_d and game._check_collision(point_r)) or (dir_u and game._check_collision(point_l)) or (dir_r and game._check_collision(point_u)) or (dir_l and game._check_collision(point_d)),
            dir_l, dir_r, dir_u, dir_d,
            (game.food_position[1] < head[1]) if game.food_position else 0,
            (game.food_position[1] > head[1]) if game.food_position else 0,
            (game.food_position[0] < head[0]) if game.food_position else 0,
            (game.food_position[0] > head[0]) if game.food_position else 0,
            flood_straight, flood_right, flood_left
        ]
        return np.array(state, dtype=float)

    def select_action(self, state):
        state_tensor = torch.tensor(state, dtype=torch.float).unsqueeze(0)
        with torch.no_grad():
            action, log_prob, entropy, value = self.network.get_action(state_tensor)
        final_move, final_move[action.item()] = [0, 0, 0], 1
        return final_move, action.item(), log_prob.item(), value.item()
    
    def store(self, state, action, log_prob, reward, value, done):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)

    def compute_gae(self, next_value, gae_lambda=0.95):
        advantages, gae = [], 0
        values = self.values + [next_value]
        for t in reversed(range(len(self.rewards))):
            delta = self.rewards[t] - self.values[t] if self.dones[t] else self.rewards[t] + self.gamma * values[t + 1] - self.values[t]
            gae = delta if self.dones[t] else delta + self.gamma * gae_lambda * gae
            advantages.insert(0, gae)
        return advantages

    def update(self, next_state):
        next_state_tensor = torch.tensor(next_state, dtype=torch.float).unsqueeze(0)
        with torch.no_grad(): _, next_value = self.network(next_state_tensor)
        advantages = self.compute_gae(next_value.item())
        
        states = torch.tensor(np.array(self.states), dtype=torch.float)
        actions = torch.tensor(self.actions, dtype=torch.long)
        old_log_probs = torch.tensor(self.log_probs, dtype=torch.float)
        advantages = torch.tensor(advantages, dtype=torch.float)
        returns = advantages + torch.tensor(self.values, dtype=torch.float)
        
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        for _ in range(self.n_epochs):
            probs, values = self.network(states)
            dist = Categorical(probs)
            new_log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = F.mse_loss(values.squeeze(), returns)
            loss = actor_loss + 0.5 * critic_loss - self.entropy_coef * entropy
            
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
            self.optimizer.step()
        
        self.states, self.actions, self.log_probs = [], [], []
        self.rewards, self.values, self.dones = [], [], []

def get_absolute_action(relative_move, game):
    head = game.snake_position[-1]
    if len(game.snake_position) > 1:
        neck = game.snake_position[-2]
        if head[0] < neck[0]:   direction = 0
        elif head[0] > neck[0]: direction = 1
        elif head[1] < neck[1]: direction = 2
        else:                   direction = 3
    else:
        direction = 3
    right_turn = {0: 3, 3: 1, 1: 2, 2: 0}
    left_turn = {0: 2, 2: 1, 1: 3, 3: 0}
    if relative_move == [1, 0, 0]: return direction
    elif relative_move == [0, 1, 0]: return right_turn[direction]
    elif relative_move == [0, 0, 1]: return left_turn[direction]
    return direction

# ============================================================================
# MAIN CURRICULUM PIPELINE
# ============================================================================

def pretrain_critic(actor_critic, expert_data, epochs=50):
    print("\n" + "=" * 60)
    print("🎓 PHASE 1b: Pretraining PPO on 5x5 Expert Data")
    print("=" * 60)
    
    optimizer = optim.Adam(actor_critic.parameters(), lr=0.001)
    states = torch.tensor([d['state'] for d in expert_data], dtype=torch.float)
    returns = torch.tensor([d['return'] for d in expert_data], dtype=torch.float)
    
    for epoch in range(epochs):
        indices = torch.randperm(len(states))
        total_loss = 0
        for i in range(0, len(states), 64):
            batch_st = states[indices[i:i+64]]
            batch_ret = returns[indices[i:i+64]]
            _, values = actor_critic(batch_st)
            loss = F.mse_loss(values.squeeze(), batch_ret)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if (epoch+1) % 10 == 0:
            print(f"  Pretrain Epoch {epoch+1}/{epochs}, Loss: {total_loss/(len(states)/64):.4f}")

def pretrain_actor(actor_critic, expert_data, epochs=50):
    print("\n" + "=" * 60)
    print("🎓 PHASE 1c: Behavioral Cloning (Actor) on 5x5 Expert Data")
    print("=" * 60)
    
    optimizer = optim.Adam(actor_critic.parameters(), lr=0.001)
    states = torch.tensor([d['state'] for d in expert_data], dtype=torch.float)
    actions = torch.tensor([d['action'] for d in expert_data], dtype=torch.long)
    
    for epoch in range(epochs):
        indices = torch.randperm(len(states))
        correct = 0
        total_loss = 0
        batches = 0
        
        for i in range(0, len(states), 64):
            batch_st = states[indices[i:i+64]]
            batch_act = actions[indices[i:i+64]]
            
            probs, _ = actor_critic(batch_st)
            loss = F.cross_entropy(probs, batch_act)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pred = probs.argmax(dim=1)
            correct += (pred == batch_act).sum().item()
            batches += 1
            
        if (epoch+1) % 10 == 0:
            acc = correct / len(states) * 100
            print(f"  BC Epoch {epoch+1}/{epochs}, Loss: {total_loss/batches:.4f}, Acc: {acc:.1f}%")

def train_phase(agent, board_size, num_games, max_steps):
    print("\n" + "=" * 60)
    print(f"🎓 TRAINING PHASE: {board_size}x{board_size} Board | {num_games} Games")
    print("=" * 60)
    game = SnakeGame(board_size=board_size)
    total_score, record = 0, 0
    steps_in_game, total_steps = 0, 0
    agent.n_games = 0
    
    while agent.n_games < num_games:
        state = agent.get_state(game)
        action_relative, action_idx, log_prob, value = agent.select_action(state)
        action = get_absolute_action(action_relative, game)
        _, reward, done, info = game.step(action)
        steps_in_game += 1
        total_steps += 1
        
        if steps_in_game > max_steps: done, reward = True, -10
        agent.store(state, action_idx, log_prob, reward, value, done)
        
        if total_steps % 128 == 0:
            next_state = agent.get_state(game)
            agent.update(next_state)
        
        if done:
            game.reset()
            agent.n_games += 1
            if info['score'] > record: record = info['score']
            total_score += info['score']
            steps_in_game = 0
            if agent.n_games % 100 == 0:
                print(f"  Game {agent.n_games}, Score: {info['score']}, Record: {record}, Mean: {total_score/agent.n_games:.2f}")

def main():
    print("\n" + "=" * 70)
    print("🎓 PPO CURRICULUM LEARNING EXPERIMENT (EXTENDED)")
    print("   Stage 1: Imitation Learning on 5x5")
    print("   Stage 2: Transfer to 8x8")
    print("   Stage 3: Scale to 10x10")
    print("=" * 70)
    
    # 1. Collect Expert Data on 5x5
    collector = ExpertCollector(board_size=5)
    expert_data, _ = collector.collect_with_trained_reinforce(num_episodes=50, min_score=3)
    
    # 2. Pretrain PPO on 5x5
    network = ActorCritic(14, 256, 3)
    pretrain_critic(network, expert_data, epochs=50)
    pretrain_actor(network, expert_data, epochs=50)
    
    # 3. Train PPO on 5x5 (Warmup)
    agent = PPOAgent(network=network, lr=0.0003)
    train_phase(agent, board_size=5, num_games=3000, max_steps=500)
    
    # 4. Transfer to 8x8
    print("\n🚀 STAGE 2: TRANSFERRING TO 8x8 BOARD...")
    train_phase(agent, board_size=8, num_games=5000, max_steps=2000)
    
    # 5. Transfer to 10x10!
    print("\n🚀 STAGE 3: SCALING TO 10x10 BOARD...")
    train_phase(agent, board_size=10, num_games=10000, max_steps=5000)

if __name__ == '__main__':
    main()

