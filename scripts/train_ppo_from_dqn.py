"""
================================================================================
PPO PRETRAINED FROM DQN EXPERT DEMONSTRATIONS
================================================================================

🎓 KNOWLEDGE DISTILLATION: DQN → PPO

This script demonstrates an advanced technique: using a trained DQN as the
"expert" to generate demonstrations for PPO pretraining.

Why combine DQN and PPO?
- DQN: Good at exploration (epsilon-greedy), learns Q-values
- PPO: Good at stable policy optimization, can generalize better

By using DQN's best episodes as demonstrations, PPO can:
1. Learn from DQN's exploration (which found good strategies)
2. Build on that with its own stable policy optimization

This is similar to how AlphaGo used Monte Carlo Tree Search to generate
training data for its neural networks!

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

# ============================================================================
# GAME ENVIRONMENT
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
# 🎓 DQN NETWORK (Dueling architecture - same as train_dqn.py)
# ============================================================================

class DuelingQNet(nn.Module):
    """
    Dueling DQN Network
    
    Separates value (V) and advantage (A) streams:
    Q(s,a) = V(s) + A(s,a) - mean(A)
    """
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.value_stream = nn.Linear(hidden_size, 1)
        self.advantage_stream = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        val = self.value_stream(x)
        adv = self.advantage_stream(x)
        if adv.dim() == 1:
            return val + (adv - adv.mean())
        else:
            return val + (adv - adv.mean(dim=1, keepdim=True))


# ============================================================================
# 🎓 STATE FEATURE EXTRACTOR (shared between DQN and PPO)
# ============================================================================

def get_reachable_count(game, start_pos):
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

def get_state_features(game):
    """Extract 14-dimensional state features"""
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
    flood_straight = get_reachable_count(game, pos_straight) / total_cells
    flood_right = get_reachable_count(game, pos_right) / total_cells
    flood_left = get_reachable_count(game, pos_left) / total_cells

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


def get_absolute_action(relative_move, game):
    """Convert relative action [straight, right, left] to absolute direction"""
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


# ============================================================================
# 🎓 DQN TRAINER (simplified for expert collection)
# ============================================================================

class DQNTrainer:
    def __init__(self, model, lr=0.001, gamma=0.9):
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.gamma = gamma
        self.criterion = nn.MSELoss()

    def train_step(self, states, actions, rewards, next_states, dones):
        states = torch.tensor(np.array(states), dtype=torch.float)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float)
        actions = torch.tensor(np.array(actions), dtype=torch.long)
        rewards = torch.tensor(np.array(rewards), dtype=torch.float)
        
        pred = self.model(states)
        
        with torch.no_grad():
            next_pred = self.model(next_states)
        
        target = pred.clone()
        for idx in range(len(dones)):
            Q_new = rewards[idx]
            if not dones[idx]:
                Q_new = rewards[idx] + self.gamma * torch.max(next_pred[idx])
            target[idx][torch.argmax(actions[idx]).item()] = Q_new
        
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        self.optimizer.step()


# ============================================================================
# 🎓 PHASE 1: TRAIN DQN AND COLLECT EXPERT DEMONSTRATIONS
# ============================================================================

def train_dqn_and_collect(board_size=8, num_games=8000, min_score=20):
    """
    🎓 DQN EXPERT COLLECTION
    
    1. Train DQN with epsilon-greedy exploration
    2. Collect trajectories from high-scoring games
    3. Return expert data for PPO pretraining
    """
    print("\n" + "=" * 60)
    print("🎓 PHASE 1: Training DQN Expert")
    print("=" * 60)
    
    game = SnakeGame(board_size=board_size)
    model = DuelingQNet(14, 256, 3)
    trainer = DQNTrainer(model, lr=0.001, gamma=0.9)
    
    memory = deque(maxlen=100_000)
    expert_trajectories = []  # Store good episodes
    
    n_games = 0
    record = 0
    total_score = 0
    epsilon = 1.0
    
    MAX_STEPS = 2000
    
    while n_games < num_games:
        game.reset()
        trajectory = []  # Current episode trajectory
        rewards = []
        done = False
        steps = 0
        
        while not done and steps < MAX_STEPS:
            state = get_state_features(game)
            
            # Epsilon-greedy action selection
            if random.random() < epsilon:
                action_idx = random.randint(0, 2)
            else:
                state_tensor = torch.tensor(state, dtype=torch.float)
                with torch.no_grad():
                    q_values = model(state_tensor)
                action_idx = torch.argmax(q_values).item()
            
            action_onehot = [0, 0, 0]
            action_onehot[action_idx] = 1
            
            # Store state and action
            trajectory.append((state.copy(), action_idx))
            
            # Execute action
            abs_action = get_absolute_action(action_onehot, game)
            _, reward, done, info = game.step(abs_action)
            rewards.append(reward)
            
            next_state = get_state_features(game)
            steps += 1
            
            if steps >= MAX_STEPS:
                done = True
                reward = -10
                rewards[-1] = reward
            
            # Store in replay memory
            memory.append((state, action_onehot, reward, next_state, done))
        
        score = info['score']
        n_games += 1
        total_score += score
        
        # Decay epsilon
        epsilon = max(0.01, 1.0 - (n_games * 0.0002))
        
        # Train from replay memory
        if len(memory) > 1000:
            batch = random.sample(memory, min(1000, len(memory)))
            states, actions, rewards_b, next_states, dones = zip(*batch)
            trainer.train_step(states, actions, rewards_b, next_states, dones)
        
        if score > record:
            record = score
        
        # 🎓 COLLECT HIGH-SCORING EPISODES AS EXPERT DATA
        if score >= min_score:
            expert_trajectories.append((trajectory, rewards))
        
        if n_games % 500 == 0:
            mean_score = total_score / n_games
            print(f"  Game {n_games}/{num_games} | Score: {score} | Record: {record} | Mean: {mean_score:.2f} | ε: {epsilon:.2f} | Experts: {len(expert_trajectories)}")
    
    print(f"\n✅ DQN Training Complete!")
    print(f"   Record: {record}")
    print(f"   Expert episodes collected: {len(expert_trajectories)}")
    
    # Convert trajectories to (state, action, return) format
    expert_data = []
    gamma = 0.99
    
    for trajectory, rewards in expert_trajectories:
        # Compute returns
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
        
        for (state, action), ret in zip(trajectory, returns):
            expert_data.append({
                'state': state,
                'action': action,
                'return': ret
            })
    
    print(f"   Total expert (state, action, return) pairs: {len(expert_data)}")
    
    return expert_data, model


# ============================================================================
# 🎓 ACTOR-CRITIC NETWORK FOR PPO
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


# ============================================================================
# 🎓 PHASE 2: PRETRAIN PPO ON DQN DEMONSTRATIONS
# ============================================================================

def pretrain_ppo(network, expert_data, epochs=100):
    """
    Pretrain PPO's critic and actor on DQN expert demonstrations
    """
    print("\n" + "=" * 60)
    print("🎓 PHASE 2: Pretraining PPO on DQN Expert Data")
    print("=" * 60)
    
    optimizer = optim.Adam(network.parameters(), lr=0.001)
    
    states = torch.tensor(np.array([d['state'] for d in expert_data]), dtype=torch.float)
    actions = torch.tensor([d['action'] for d in expert_data], dtype=torch.long)
    returns = torch.tensor([d['return'] for d in expert_data], dtype=torch.float)
    
    print(f"\nPretraining on {len(states)} DQN expert pairs...")
    print(f"Return range: [{returns.min():.2f}, {returns.max():.2f}]")
    
    batch_size = 64
    n_samples = len(states)
    
    for epoch in range(epochs):
        indices = torch.randperm(n_samples)
        total_critic_loss = 0
        total_actor_loss = 0
        correct = 0
        n_batches = 0
        
        for i in range(0, n_samples, batch_size):
            batch_idx = indices[i:i+batch_size]
            batch_states = states[batch_idx]
            batch_actions = actions[batch_idx]
            batch_returns = returns[batch_idx]
            
            probs, values = network(batch_states)
            
            # Critic loss: predict returns
            critic_loss = F.mse_loss(values.squeeze(), batch_returns)
            
            # Actor loss: behavioral cloning
            actor_loss = F.cross_entropy(probs, batch_actions)
            
            # Combined loss
            loss = 0.5 * critic_loss + actor_loss
            
            # Accuracy tracking
            predicted = probs.argmax(dim=1)
            correct += (predicted == batch_actions).sum().item()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_critic_loss += critic_loss.item()
            total_actor_loss += actor_loss.item()
            n_batches += 1
        
        if (epoch + 1) % 20 == 0:
            accuracy = correct / n_samples * 100
            print(f"  Epoch {epoch+1}/{epochs} | Critic: {total_critic_loss/n_batches:.3f} | Actor: {total_actor_loss/n_batches:.3f} | Acc: {accuracy:.1f}%")
    
    print("\n✅ Pretraining complete!")


# ============================================================================
# 🎓 PPO AGENT (same as before)
# ============================================================================

class PPOAgent:
    def __init__(self, pretrained_network, lr=0.0003, gamma=0.99, 
                 clip_epsilon=0.2, n_epochs=4, entropy_coef=0.01):
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.n_epochs = n_epochs
        self.entropy_coef = entropy_coef
        self.n_games = 0
        
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
        
        self.network = pretrained_network
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
    
    def select_action(self, state):
        state_tensor = torch.tensor(state, dtype=torch.float).unsqueeze(0)
        with torch.no_grad():
            action, log_prob, entropy, value = self.network.get_action(state_tensor)
        final_move = [0, 0, 0]
        final_move[action.item()] = 1
        return final_move, action.item(), log_prob.item(), value.item()
    
    def store(self, state, action, log_prob, reward, value, done):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)
    
    def compute_gae(self, next_value, gae_lambda=0.95):
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
        next_state_tensor = torch.tensor(next_state, dtype=torch.float).unsqueeze(0)
        with torch.no_grad():
            _, next_value = self.network(next_state_tensor)
            next_value = next_value.item()
        
        advantages = self.compute_gae(next_value)
        
        states = torch.tensor(np.array(self.states), dtype=torch.float)
        actions = torch.tensor(self.actions, dtype=torch.long)
        old_log_probs = torch.tensor(self.log_probs, dtype=torch.float)
        advantages = torch.tensor(advantages, dtype=torch.float)
        returns = advantages + torch.tensor(self.values, dtype=torch.float)
        
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
        
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []


# ============================================================================
# 🎓 PHASE 3: PPO TRAINING WITH PRETRAINED NETWORK
# ============================================================================

def train_ppo(network, board_size=8, num_games=5000):
    """Train PPO with the pretrained network"""
    print("\n" + "=" * 60)
    print("🎓 PHASE 3: PPO Training with DQN-Pretrained Network")
    print("=" * 60)
    
    game = SnakeGame(board_size=board_size)
    agent = PPOAgent(pretrained_network=network, lr=0.0003)
    
    ROLLOUT_LENGTH = 256
    MAX_STEPS = 2000
    
    total_score = 0
    record = 0
    steps_in_game = 0
    total_steps = 0
    
    print(f"\nStarting PPO Training on {board_size}x{board_size} Board...")
    print("=" * 60)
    
    while agent.n_games < num_games:
        state = get_state_features(game)
        action_relative, action_idx, log_prob, value = agent.select_action(state)
        action = get_absolute_action(action_relative, game)
        _, reward, done, info = game.step(action)
        score = info['score']
        
        steps_in_game += 1
        total_steps += 1
        
        if steps_in_game > MAX_STEPS:
            done = True
            reward = -10
        
        agent.store(state, action_idx, log_prob, reward, value, done)
        
        if total_steps % ROLLOUT_LENGTH == 0:
            next_state = get_state_features(game)
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
    print("🎉 Training Complete!")
    print(f"   Best Score: {record}")
    print(f"   Final Mean Score: {mean_score:.2f}")
    
    return record, mean_score


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "=" * 70)
    print("🎓 PPO WITH DQN EXPERT DEMONSTRATIONS")
    print("   Knowledge Distillation: DQN → PPO")
    print("=" * 70)
    
    BOARD_SIZE = 8
    
    # Phase 1: Train DQN and collect expert demos
    expert_data, dqn_model = train_dqn_and_collect(
        board_size=BOARD_SIZE,
        num_games=8000,
        min_score=20  # Only collect games with score >= 20
    )
    
    # Phase 2: Pretrain PPO on DQN expert data
    ppo_network = ActorCritic(14, 256, 3)
    pretrain_ppo(ppo_network, expert_data, epochs=100)
    
    # Phase 3: PPO training with pretrained network
    record, mean_score = train_ppo(ppo_network, board_size=BOARD_SIZE, num_games=5000)


if __name__ == '__main__':
    main()
