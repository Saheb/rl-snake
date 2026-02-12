"""
==============================================================================
ACTOR-CRITIC (A2C): Advantage Actor-Critic for Snake
==============================================================================

🎓 WHAT IS ACTOR-CRITIC?

Actor-Critic combines the best of both worlds:
    - ACTOR (Policy Network): Decides which action to take (like REINFORCE)
    - CRITIC (Value Network): Estimates how good a state is (like DQN's value)

The key insight:
    REINFORCE uses Monte Carlo returns (wait until episode ends)
    Actor-Critic uses TD learning (update every step!)

==============================================================================
WHY IS THIS BETTER THAN REINFORCE?
==============================================================================

REINFORCE Problem: HIGH VARIANCE
    - Must wait for entire episode to get returns
    - If episode is long, credit assignment is hard
    - "Did I get food because of move 1 or move 47?"

Actor-Critic Solution: BOOTSTRAPPING
    - Use Critic to ESTIMATE future rewards
    - Update after every single step
    - Much lower variance, faster learning!

==============================================================================
THE ADVANTAGE FUNCTION
==============================================================================

Instead of using raw returns G_t, we use the ADVANTAGE:

    A(s,a) = Q(s,a) - V(s)
           = "How much better is action 'a' than the average action?"

In practice, we estimate this using TD error:

    A(s,a) ≈ r + γV(s') - V(s)
           = "reward received + value of next state - value of current state"

If A > 0: Action was BETTER than expected → increase probability
If A < 0: Action was WORSE than expected → decrease probability

==============================================================================
COMPARISON TABLE
==============================================================================

| Aspect          | REINFORCE              | Actor-Critic (A2C)      |
|-----------------|------------------------|-------------------------|
| Update timing   | End of episode (MC)    | Every step (TD)         |
| Variance        | HIGH                   | LOWER                   |
| Bias            | NONE (unbiased)        | SOME (from bootstrapping)|
| Networks        | 1 (Policy only)        | 2 (Actor + Critic)      |
| Sample efficiency| Lower                 | Higher                  |

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
# GAME ENVIRONMENT (Same as REINFORCE)
# ============================================================================
class GameState(Enum):
    PLAYING = 1
    GAME_OVER = 2
    WIN = 3

class SnakeGame:
    def __init__(self, board_size=7):
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
# 🎓 ACTOR NETWORK (Same as Policy Network in REINFORCE)
# ============================================================================
# The Actor outputs action PROBABILITIES.
# It answers: "What action should I take in this state?"

class ActorNetwork(nn.Module):
    """
    The Actor (Policy) Network: π(a|s; θ_actor)
    
    Same as REINFORCE's PolicyNetwork:
    - Input: State features
    - Output: Action probabilities (softmax)
    """
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, output_size)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.softmax(x, dim=-1)


# ============================================================================
# 🎓 CRITIC NETWORK (NEW! This is what makes A2C different)
# ============================================================================
# The Critic estimates the VALUE of being in a state.
# It answers: "How good is this state? What's my expected total reward from here?"

class CriticNetwork(nn.Module):
    """
    The Critic (Value) Network: V(s; θ_critic)
    
    This is NEW compared to REINFORCE!
    - Input: State features  
    - Output: Single scalar value V(s)
    
    The Critic learns to predict "how much reward will I get from this state?"
    This is used to compute the ADVANTAGE = actual_reward - predicted_reward
    """
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, 1)  # Output: single value V(s)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)  # No activation - value can be any real number


# ============================================================================
# 🎓 ACTOR-CRITIC AGENT (N-STEP VERSION)
# ============================================================================

class A2CAgent:
    """
    Advantage Actor-Critic Agent with N-Step Returns
    
    🎓 WHY N-STEP?
    
    1-Step TD (vanilla A2C):
        - High bias (bootstrapping from potentially wrong value estimates)
        - Low variance
        - Can be unstable if critic is inaccurate
    
    Monte Carlo (REINFORCE):
        - No bias (uses actual returns)
        - High variance
        - Slow learning
    
    N-Step (this version):
        - Collect N steps, then update
        - Use actual rewards for N steps, then bootstrap from V(s_{t+N})
        - Balance of bias and variance!
        
    Formula:
        G_t^{(n)} = r_t + γr_{t+1} + ... + γ^{n-1}r_{t+n-1} + γ^n V(s_{t+n})
    """
    
    def __init__(self, lr_actor=0.0005, lr_critic=0.001, gamma=0.99, 
                 entropy_coef=0.05, n_steps=20):
        self.gamma = gamma
        self.n_games = 0
        self.entropy_coef = entropy_coef  # Higher = more exploration
        self.n_steps = n_steps  # Collect this many steps before updating
        
        # Storage for N-step collection
        self.states = []
        self.log_probs = []
        self.entropies = []
        self.values = []
        self.rewards = []
        self.dones = []
        
        # Networks
        self.actor = ActorNetwork(14, 256, 3)
        self.critic = CriticNetwork(14, 256)
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)
    
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

    def get_action(self, state):
        """Select action using Actor network"""
        state_tensor = torch.tensor(state, dtype=torch.float).unsqueeze(0)
        
        probs = self.actor(state_tensor)
        dist = Categorical(probs)
        action = dist.sample()
        
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        value = self.critic(state_tensor)
        
        final_move = [0, 0, 0]
        final_move[action.item()] = 1
        
        return final_move, log_prob, entropy, value
    
    def store_transition(self, state, log_prob, entropy, value, reward, done):
        """Store a single transition"""
        self.states.append(state)
        self.log_probs.append(log_prob)
        self.entropies.append(entropy)
        self.values.append(value)
        self.rewards.append(reward)
        self.dones.append(done)
    
    def should_update(self):
        """Check if we have enough steps to update"""
        return len(self.rewards) >= self.n_steps
    
    def update(self, next_state, next_done):
        """
        🎓 N-STEP A2C UPDATE
        
        1. Compute N-step returns for each stored step
        2. Calculate advantages
        3. Update Actor and Critic
        """
        # Get bootstrap value for the last state
        if next_done:
            next_value = 0
        else:
            next_state_tensor = torch.tensor(next_state, dtype=torch.float).unsqueeze(0)
            with torch.no_grad():
                next_value = self.critic(next_state_tensor).item()
        
        # 🎓 COMPUTE N-STEP RETURNS (backwards)
        # G_t = r_t + γr_{t+1} + ... + γ^(n-1)r_{t+n-1} + γ^n V(s_{t+n})
        returns = []
        G = next_value
        for i in reversed(range(len(self.rewards))):
            if self.dones[i]:
                G = 0  # Reset at episode boundaries
            G = self.rewards[i] + self.gamma * G
            returns.insert(0, G)
        
        returns = torch.tensor(returns, dtype=torch.float)
        values = torch.cat(self.values).squeeze()
        log_probs = torch.stack(self.log_probs)
        entropies = torch.stack(self.entropies)
        
        # 🎓 ADVANTAGES = Returns - Values
        advantages = returns - values.detach()
        
        # Normalize advantages
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # 🎓 LOSSES
        actor_loss = -(log_probs * advantages).mean() - self.entropy_coef * entropies.mean()
        critic_loss = F.mse_loss(values, returns)
        
        # Update Critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
        self.critic_optimizer.step()
        
        # Update Actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
        self.actor_optimizer.step()
        
        # Clear storage
        self.states = []
        self.log_probs = []
        self.entropies = []
        self.values = []
        self.rewards = []
        self.dones = []
        
        return actor_loss.item(), critic_loss.item()


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
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    
    # N-step A2C with higher entropy for exploration
    agent = A2CAgent(lr_actor=0.0005, lr_critic=0.001, gamma=0.99, 
                     entropy_coef=0.05, n_steps=20)
    game = SnakeGame(board_size=5)
    
    MAX_STEPS = 500
    print("Starting N-Step A2C Training on 5x5 Board...")
    print("=" * 60)
    
    steps_in_game = 0
    while agent.n_games < 5000:
        state = agent.get_state(game)
        
        # Get action
        action_relative, log_prob, entropy, value = agent.get_action(state)
        
        # Execute action
        action = get_absolute_action(action_relative, game)
        _, reward, done, info = game.step(action)
        score = info['score']
        
        steps_in_game += 1
        if steps_in_game > MAX_STEPS:
            done = True
            reward = -10
        
        # Store transition
        agent.store_transition(state, log_prob, entropy, value, reward, done)
        
        # 🎓 UPDATE EVERY N STEPS (not every single step!)
        if agent.should_update() or done:
            next_state = agent.get_state(game)
            agent.update(next_state, done)
        
        if done:
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

