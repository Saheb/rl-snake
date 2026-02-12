"""
==============================================================================
REINFORCE: A Policy Gradient Algorithm for Snake
==============================================================================

🎓 WHAT IS REINFORCE?

REINFORCE is the simplest "policy gradient" algorithm. Unlike DQN which learns
Q-values (how good is action A in state S?), REINFORCE directly learns the 
POLICY (what's the probability of taking action A in state S?).

The key insight is the "Policy Gradient Theorem":
    
    ∇J(θ) = E[ G_t · ∇log π(a|s; θ) ]
    
Translation:
    - J(θ) = Expected total reward (what we want to maximize)
    - G_t = Return (sum of rewards from time t onwards)
    - π(a|s; θ) = Policy network (outputs action probabilities)
    - ∇log π(a|s) = "How to change θ to increase probability of action a"

In plain English:
    "If an action led to high returns (G_t), increase its probability.
     If it led to low returns, decrease its probability."

==============================================================================
KEY DIFFERENCES FROM DQN
==============================================================================

| DQN                          | REINFORCE                      |
|------------------------------|--------------------------------|
| Learns Q(s,a) values         | Learns π(a|s) probabilities    |
| Updates every step (TD)      | Updates at END of episode (MC) |
| Off-policy (uses replay)     | On-policy (no replay buffer)   |
| Action = argmax Q            | Action = sample from π         |
| Output: 3 raw scores         | Output: 3 probabilities (sum=1)|

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
from matplotlib import pyplot as plt

# ============================================================================
# GAME ENVIRONMENT (Same as DQN)
# ============================================================================
class GameState(Enum):
    PLAYING = 1
    GAME_OVER = 2
    WIN = 3

class SnakeGame:
    def __init__(self, board_size=8):
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
            return self.get_state(), reward, True, {"score": self.score, "game_state": self.game_state.value}
        
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

        max_score = self.board_size * self.board_size - 1
        if self.score >= max_score:
            self.game_state = GameState.WIN
            reward = 50
            done = True
        
        return self.get_state(), reward, done, {"score": self.score, "game_state": self.game_state.value}
    
    def _move_snake(self, action):
        head_x, head_y = self.snake_position[-1]
        if action == 0:   new_head = (head_x - 1, head_y)  # UP
        elif action == 1: new_head = (head_x + 1, head_y)  # DOWN
        elif action == 2: new_head = (head_x, head_y - 1)  # LEFT
        elif action == 3: new_head = (head_x, head_y + 1)  # RIGHT
        else:             new_head = (head_x, head_y)
        return new_head

# ============================================================================
# 🎓 POLICY NETWORK
# ============================================================================
# Unlike DQN which outputs Q-values, the Policy Network outputs ACTION 
# PROBABILITIES. We use Softmax at the end to ensure outputs sum to 1.
#
# Example:
#   DQN Output:    [2.3, -1.2, 0.5]  <- Raw Q-values (can be any number)
#   Policy Output: [0.7, 0.1, 0.2]  <- Probabilities (sum to 1)
# ============================================================================

class PolicyNetwork(nn.Module):
    """
    The Policy Network π(a|s; θ)
    
    Input:  State features (14-dimensional vector)
    Output: Probability distribution over actions (3 actions: straight, right, left)
    
    Key difference from DQN:
        - DQN uses NO activation on final layer (raw Q-values)
        - Policy Net uses SOFTMAX to get probabilities
    """
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        # 🎓 SOFTMAX: Converts raw logits to probabilities (sum = 1)
        # This is THE key difference from DQN's output
        return F.softmax(x, dim=-1)

# ============================================================================
# 🎓 REINFORCE AGENT
# ============================================================================

class REINFORCEAgent:
    """
    REINFORCE Agent
    
    Key differences from DQN Agent:
    1. No replay buffer (on-policy learning)
    2. Stores (log_prob, reward) for the CURRENT episode only
    3. Updates ONLY at the end of each episode
    4. Uses the Policy Gradient loss
    """
    
    def __init__(self, lr=0.002, gamma=0.99, entropy_coef=0.01):
        self.gamma = gamma
        self.n_games = 0
        self.entropy_coef = entropy_coef  # 🎓 Entropy bonus coefficient
        
        # 🎓 BASELINE: A moving average of returns
        # Subtracting a baseline reduces variance without adding bias!
        # Instead of: "Action A got return +5, increase probability"
        # We say: "Action A got +5, but average is +3, so it's only +2 above average"
        self.baseline = 0.0
        self.baseline_decay = 0.99  # Exponential moving average
        
        # 🎓 EPISODE MEMORY
        # Unlike DQN's replay buffer, we only store the current episode.
        # REINFORCE is "on-policy" - we can only learn from data collected
        # by the current policy.
        self.log_probs = []   # log π(a|s) for each step
        self.rewards = []     # r_t for each step
        self.entropies = []   # entropy at each step (for bonus)
        
        # Policy Network (14 input features, 512 hidden, 3 outputs)
        self.policy = PolicyNetwork(14, 512, 3)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
    
    def _get_reachable_count(self, game, start_pos):
        """BFS flood fill (same as DQN agent)"""
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
        """Extract state features (same as DQN agent)"""
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

    def get_action(self, state):
        """
        🎓 ACTION SELECTION (The KEY difference from DQN!)
        
        DQN:       action = argmax Q(s, a)  (deterministic, greedy)
        REINFORCE: action = sample from π(a|s)  (stochastic, probabilistic)
        
        We also store log π(a|s) for the update later.
        """
        state_tensor = torch.tensor(state, dtype=torch.float).unsqueeze(0)
        
        # Get action probabilities from policy network
        probs = self.policy(state_tensor)  # e.g., [0.7, 0.2, 0.1]
        
        # 🎓 CATEGORICAL DISTRIBUTION
        # This creates a probability distribution we can sample from
        dist = Categorical(probs)
        
        # 🎓 SAMPLE an action (stochastic!)
        # Unlike DQN's argmax, we SAMPLE. This naturally explores.
        action = dist.sample()  # Returns tensor with action index (0, 1, or 2)
        
        # 🎓 STORE log π(a|s) for the policy gradient update
        # We need this to compute: ∇log π(a|s) * G
        log_prob = dist.log_prob(action)
        self.log_probs.append(log_prob)
        
        # 🎓 STORE ENTROPY for exploration bonus
        # Entropy = -sum(p * log(p)). High entropy = uniform distribution = more exploration
        entropy = dist.entropy()
        self.entropies.append(entropy)
        
        # Convert to one-hot for compatibility with game
        final_move = [0, 0, 0]
        final_move[action.item()] = 1
        return final_move
    
    def store_reward(self, reward):
        """Store reward for current step"""
        self.rewards.append(reward)
    
    def update_policy(self):
        """
        🎓 THE REINFORCE UPDATE (End of Episode)
        
        This is where the magic happens!
        
        For each step t in the episode:
            1. Compute G_t = sum of discounted future rewards from step t
            2. Compute loss = -log π(a_t|s_t) * G_t
            3. Backpropagate and update θ (policy parameters)
        
        Why the negative sign?
            - We want to MAXIMIZE expected reward
            - PyTorch minimizes loss
            - So we use: loss = -log_prob * return (to maximize)
        
        Intuition:
            - If G_t is HIGH and we took action a: increase π(a|s)
            - If G_t is LOW and we took action a: decrease π(a|s)
        """
        
        # 🎓 STEP 1: Compute Returns G_t for each timestep
        # G_t = r_t + γ*r_{t+1} + γ²*r_{t+2} + ...
        returns = []
        G = 0
        for r in reversed(self.rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        
        returns = torch.tensor(returns, dtype=torch.float)
        
        # 🎓 UPDATE BASELINE (moving average of episode returns)
        episode_return = returns[0].item()  # Total return from start
        self.baseline = self.baseline_decay * self.baseline + (1 - self.baseline_decay) * episode_return
        
        # 🎓 STEP 2: Subtract baseline from returns (VARIANCE REDUCTION!)
        # This is THE key improvement over vanilla REINFORCE.
        # Instead of "update based on raw return", we update based on
        # "how much better/worse than average was this episode?"
        advantages = returns - self.baseline
        
        # Normalize advantages (additional variance reduction)
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # 🎓 STEP 3: Compute Policy Gradient Loss with Entropy Bonus
        # loss = - sum( log π(a_t|s_t) * A_t ) - entropy_coef * entropy
        policy_loss = 0
        entropy_bonus = 0
        for log_prob, advantage, entropy in zip(self.log_probs, advantages, self.entropies):
            policy_loss += -log_prob * advantage  # Negative because we want to maximize
            entropy_bonus += entropy
        
        # 🎓 ENTROPY BONUS: Encourages exploration by penalizing deterministic policies
        # Higher entropy = more uniform action distribution = more exploration
        total_loss = policy_loss - self.entropy_coef * entropy_bonus
        
        # 🎓 STEP 4: Backprop and Update
        self.optimizer.zero_grad()
        total_loss.backward()
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.optimizer.step()
        
        # Clear episode memory
        self.log_probs = []
        self.rewards = []
        self.entropies = []
        
        return total_loss.item()

# ============================================================================
# TRAINING LOOP
# ============================================================================

def get_absolute_action(relative_move, game):
    """
    Convert relative action (straight/right/left) to absolute (up/down/left/right)
    
    🎓 RELATIVE vs ABSOLUTE ACTIONS:
    The policy outputs THREE actions: [straight, right, left]
    But the game needs FOUR absolute actions: [UP=0, DOWN=1, LEFT=2, RIGHT=3]
    
    We need to convert based on current direction:
    - If heading UP:   straight=UP,   right=RIGHT, left=LEFT
    - If heading DOWN: straight=DOWN, right=LEFT,  left=RIGHT
    - If heading LEFT: straight=LEFT, right=UP,    left=DOWN
    - If heading RIGHT:straight=RIGHT,right=DOWN,  left=UP
    """
    head = game.snake_position[-1]
    if len(game.snake_position) > 1:
        neck = game.snake_position[-2]
        if head[0] < neck[0]:   direction = 0  # UP
        elif head[0] > neck[0]: direction = 1  # DOWN
        elif head[1] < neck[1]: direction = 2  # LEFT
        else:                   direction = 3  # RIGHT
    else:
        direction = 3  # Default RIGHT

    # 🎓 DIRECTION MAPPING TABLE
    # Clockwise order: UP(0) -> RIGHT(3) -> DOWN(1) -> LEFT(2) -> UP(0)...
    # right_turn[direction] = turn 90° clockwise
    # left_turn[direction]  = turn 90° counter-clockwise
    
    right_turn = {0: 3, 3: 1, 1: 2, 2: 0}  # UP->RIGHT->DOWN->LEFT->UP
    left_turn = {0: 2, 2: 1, 1: 3, 3: 0}   # UP->LEFT->DOWN->RIGHT->UP

    if relative_move == [1, 0, 0]:    # Straight
        return direction
    elif relative_move == [0, 1, 0]:  # Right turn
        return right_turn[direction]
    elif relative_move == [0, 0, 1]:  # Left turn
        return left_turn[direction]
    return direction

def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    
    agent = REINFORCEAgent(lr=0.001, gamma=0.99)
    game = SnakeGame(board_size=7)
    
    MAX_STEPS = 1200
    print("Starting REINFORCE Training on 7x7 Board...")
    print("=" * 60)
    
    steps_in_game = 0
    while agent.n_games < 5000:  # Train for 5000 episodes
        state = agent.get_state(game)
        
        # Get action from policy (stochastic!)
        action_relative = agent.get_action(state)
        
        # Execute action
        action = get_absolute_action(action_relative, game)
        _, reward, done, info = game.step(action)
        score = info['score']
        
        steps_in_game += 1
        if steps_in_game > MAX_STEPS:
            done = True
            reward = -10
        
        # 🎓 Store reward (we'll use it at end of episode)
        agent.store_reward(reward)
        
        if done:
            # 🎓 END OF EPISODE: Update policy!
            # This is THE key difference from DQN:
            # - DQN updates every step
            # - REINFORCE updates only at episode end
            loss = agent.update_policy()
            
            game.reset()
            agent.n_games += 1
            steps_in_game = 0
            
            if score > record:
                record = score
            
            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            
            if agent.n_games % 100 == 0:
                print(f"Game {agent.n_games} | Score: {score} | Record: {record} | Mean: {mean_score:.2f}")
    
    print("=" * 60)
    print(f"Training Complete!")
    print(f"Best Score: {record}")
    print(f"Final Mean Score: {mean_score:.2f}")

if __name__ == '__main__':
    train()
