"""
==============================================================================
PPO (Proximal Policy Optimization) for Snake
==============================================================================

🎓 WHAT IS PPO?

PPO is the most popular policy gradient algorithm used today. It's used by:
- OpenAI for ChatGPT's RLHF training
- DeepMind for various game-playing agents
- Robotics research worldwide

The key idea: CLIP the policy update to prevent it from changing too much.

==============================================================================
WHY PPO IS BETTER THAN VANILLA A2C
==============================================================================

Problem with A2C/REINFORCE:
    - Large policy updates can be catastrophic
    - If you update too much, the policy can collapse
    - "Trust region" problem: how much can we safely update?

PPO's Solution: CLIPPING
    - Compute the "probability ratio": r(θ) = π_new(a|s) / π_old(a|s)
    - If the ratio gets too far from 1.0, CLIP it
    - This prevents the policy from changing too much

==============================================================================
THE PPO OBJECTIVE
==============================================================================

    L_CLIP = min( r(θ) * A,  clip(r(θ), 1-ε, 1+ε) * A )

Where:
    - r(θ) = π_new(a|s) / π_old(a|s)  (probability ratio)
    - A = advantage estimate
    - ε = clipping parameter (usually 0.2)

Intuition:
    - If advantage > 0: action was good, increase probability
      But CLIP at 1+ε to prevent increasing too much
    - If advantage < 0: action was bad, decrease probability
      But CLIP at 1-ε to prevent decreasing too much

==============================================================================
PPO ALGORITHM
==============================================================================

1. Collect N steps of experience using current policy π_old
2. Compute advantages for all steps
3. For K epochs:
   a. Compute probability ratio r(θ) = π_new / π_old
   b. Compute clipped objective
   c. Update policy (multiple times on same data!)
4. Repeat

The magic: PPO can reuse the same batch of data for multiple updates!
(Unlike vanilla policy gradient which must use fresh data each time)

==============================================================================
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
# 🎓 ACTOR-CRITIC NETWORK (Shared backbone for PPO)
# ============================================================================

class ActorCritic(nn.Module):
    """
    Combined Actor-Critic Network
    
    PPO typically uses a SHARED backbone with separate heads:
    - Actor head: outputs action probabilities
    - Critic head: outputs state value
    
    Sharing early layers helps both learn useful state representations.
    """
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        # Shared backbone
        self.shared = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU()
        )
        # Actor head (policy)
        self.actor = nn.Linear(hidden_size // 2, output_size)
        # Critic head (value)
        self.critic = nn.Linear(hidden_size // 2, 1)
    
    def forward(self, x):
        shared = self.shared(x)
        return F.softmax(self.actor(shared), dim=-1), self.critic(shared)
    
    def get_action(self, state):
        probs, value = self.forward(state)
        dist = Categorical(probs)
        action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value


# ============================================================================
# 🎓 PPO AGENT
# ============================================================================

class PPOAgent:
    """
    Proximal Policy Optimization Agent
    
    Key hyperparameters:
    - clip_epsilon: How much the policy can change (default 0.2)
    - n_epochs: How many times to reuse each batch of data
    - batch_size: Mini-batch size for updates
    """
    
    def __init__(self, lr=0.0003, gamma=0.99, clip_epsilon=0.2, 
                 n_epochs=4, entropy_coef=0.01, value_coef=0.5):
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.n_epochs = n_epochs
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.n_games = 0
        
        # Storage for rollout
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
        
        # Combined network
        self.network = ActorCritic(14, 256, 3)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
    
    def _get_reachable_count(self, game, start_pos):
        """BFS flood fill"""
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

    def get_state(self, game):
        """Extract state features"""
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

    def select_action(self, state):
        """Select action and store transition data"""
        state_tensor = torch.tensor(state, dtype=torch.float).unsqueeze(0)
        
        with torch.no_grad():
            action, log_prob, entropy, value = self.network.get_action(state_tensor)
        
        # Convert action to one-hot for game
        final_move = [0, 0, 0]
        final_move[action.item()] = 1
        
        return final_move, action.item(), log_prob.item(), value.item()
    
    def store(self, state, action, log_prob, reward, value, done):
        """Store transition"""
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)
    
    def compute_gae(self, next_value, gae_lambda=0.95):
        """
        🎓 GENERALIZED ADVANTAGE ESTIMATION (GAE)
        
        GAE is a way to compute advantages that balances bias vs variance:
        
        A_t = δ_t + (γλ)δ_{t+1} + (γλ)²δ_{t+2} + ...
        
        where δ_t = r_t + γV(s_{t+1}) - V(s_t)  (TD error)
        
        - λ = 0: Just use TD error (high bias, low variance)
        - λ = 1: Full Monte Carlo (no bias, high variance)
        - λ = 0.95: Good balance (what we use)
        """
        advantages = []
        gae = 0
        
        values = self.values + [next_value]
        
        for t in reversed(range(len(self.rewards))):
            if self.dones[t]:
                delta = self.rewards[t] - self.values[t]
                gae = delta
            else:
                delta = self.rewards[t] + self.gamma * values[t + 1] - self.values[t]
                gae = delta + self.gamma * gae_lambda * gae
            advantages.insert(0, gae)
        
        return advantages
    
    def update(self, next_state):
        """
        🎓 PPO UPDATE
        
        The key innovation: we can reuse the same batch for multiple epochs!
        This is safe because of the clipped objective.
        """
        # Get next value for GAE computation
        next_state_tensor = torch.tensor(next_state, dtype=torch.float).unsqueeze(0)
        with torch.no_grad():
            _, next_value = self.network(next_state_tensor)
            next_value = next_value.item()
        
        # Compute GAE advantages
        advantages = self.compute_gae(next_value)
        
        # Convert to tensors
        states = torch.tensor(np.array(self.states), dtype=torch.float)
        actions = torch.tensor(self.actions, dtype=torch.long)
        old_log_probs = torch.tensor(self.log_probs, dtype=torch.float)
        advantages = torch.tensor(advantages, dtype=torch.float)
        returns = advantages + torch.tensor(self.values, dtype=torch.float)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # 🎓 MULTIPLE EPOCHS on same data (PPO's key feature!)
        for _ in range(self.n_epochs):
            # Get current policy outputs
            probs, values = self.network(states)
            dist = Categorical(probs)
            new_log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()
            
            # 🎓 PROBABILITY RATIO
            # r(θ) = π_new(a|s) / π_old(a|s) = exp(log π_new - log π_old)
            ratio = torch.exp(new_log_probs - old_log_probs)
            
            # 🎓 CLIPPED SURROGATE OBJECTIVE
            # L = min(r * A, clip(r, 1-ε, 1+ε) * A)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            
            # Critic loss
            critic_loss = F.mse_loss(values.squeeze(), returns)
            
            # Total loss
            loss = actor_loss + self.value_coef * critic_loss - self.entropy_coef * entropy
            
            # Update
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
            self.optimizer.step()
        
        # Clear storage
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []


# ============================================================================
# TRAINING LOOP
# ============================================================================

def get_absolute_action(relative_move, game):
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

    if relative_move == [1, 0, 0]:
        return direction
    elif relative_move == [0, 1, 0]:
        return right_turn[direction]
    elif relative_move == [0, 0, 1]:
        return left_turn[direction]
    return direction

def train():
    total_score = 0
    record = 0
    
    agent = PPOAgent(lr=0.0003, gamma=0.99, clip_epsilon=0.2, n_epochs=4)
    game = SnakeGame(board_size=5)
    
    ROLLOUT_LENGTH = 128  # Collect this many steps before updating
    MAX_STEPS = 500
    
    print("Starting PPO Training on 5x5 Board...")
    print("=" * 60)
    
    steps_in_game = 0
    total_steps = 0
    
    while agent.n_games < 5000:
        state = agent.get_state(game)
        
        # Select action
        action_relative, action_idx, log_prob, value = agent.select_action(state)
        
        # Execute action
        action = get_absolute_action(action_relative, game)
        _, reward, done, info = game.step(action)
        score = info['score']
        
        steps_in_game += 1
        total_steps += 1
        
        if steps_in_game > MAX_STEPS:
            done = True
            reward = -10
        
        # Store transition
        agent.store(state, action_idx, log_prob, reward, value, done)
        
        # 🎓 UPDATE every ROLLOUT_LENGTH steps
        if total_steps % ROLLOUT_LENGTH == 0:
            next_state = agent.get_state(game)
            agent.update(next_state)
        
        if done:
            game.reset()
            agent.n_games += 1
            steps_in_game = 0
            
            if score > record:
                record = score
            
            total_score += score
            mean_score = total_score / agent.n_games
            
            if agent.n_games % 100 == 0:
                print(f"Game {agent.n_games} | Score: {score} | Record: {record} | Mean: {mean_score:.2f}")
    
    print("=" * 60)
    print(f"Training Complete!")
    print(f"Best Score: {record}")
    print(f"Final Mean Score: {mean_score:.2f}")

if __name__ == '__main__':
    train()
