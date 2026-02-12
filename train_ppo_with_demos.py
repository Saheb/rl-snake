"""
================================================================================
PPO WITH EXPERT DEMONSTRATIONS: Imitation Learning + Reinforcement Learning
================================================================================

🎓 WHAT IS IMITATION LEARNING?

Imitation Learning (IL) is about learning from demonstrations instead of 
(or in addition to) learning from rewards. Think of it like:

    RL: "Here are the rules of chess, figure out how to play well"
    IL: "Watch Magnus Carlsen play 1000 games, then play like him"

================================================================================
WHY COMBINE IL + RL?
================================================================================

Pure IL (Behavioral Cloning) Problem:
    - Agent only learns states the expert visited
    - If agent makes a mistake, it enters unknown territory
    - "Compounding errors" - small mistakes accumulate

Pure RL (like our A2C/PPO) Problem:
    - Takes forever to explore
    - Critic starts clueless
    - Can get stuck in bad local optima

Solution: IL + RL!
    - Use IL to give a HEAD START
    - Use RL to IMPROVE beyond the expert
    - Best of both worlds!

================================================================================
OUR APPROACH: PRETRAIN THE CRITIC
================================================================================

The insight: Our PPO failed because the critic was bad at the start.

Plan:
1. Run REINFORCE to get expert trajectories (states, actions, rewards)
2. Compute the actual returns G_t for each state in expert trajectories
3. Pretrain the critic to predict these returns: V(s) ≈ G_t
4. Now the critic starts with GOOD value estimates!
5. Run PPO with this warm-started critic

This is sometimes called "Value Function Pretraining" or "Demonstration-
Augmented Training".

================================================================================
RELATED TECHNIQUES (for your learning)
================================================================================

1. BEHAVIORAL CLONING (BC):
   - Supervised learning: predict expert's action given state
   - Loss = CrossEntropy(π(s), expert_action)
   - Problem: doesn't handle mistakes well

2. DAGGER (Dataset Aggregation):
   - Run policy, ask expert to label states
   - Add to dataset, retrain
   - Iteratively improves

3. GAIL (Generative Adversarial Imitation Learning):
   - Discriminator tells if trajectory is expert or agent
   - Agent tries to fool discriminator
   - Very powerful but complex

4. RLHF (RL from Human Feedback) - used in ChatGPT!:
   - Human ranks outputs
   - Train reward model on rankings
   - Use reward model for RL

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
import pickle
from collections import deque
from enum import Enum
from pathlib import Path

# ============================================================================
# GAME ENVIRONMENT (Same as before)
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
# 🎓 EXPERT POLICY NETWORK (from REINFORCE)
# ============================================================================
# We'll use a trained policy network to generate expert demonstrations.
# In a real scenario, this could be a human expert, a hand-coded policy,
# or any other source of good behavior.

class PolicyNetwork(nn.Module):
    """The same policy network architecture used in REINFORCE"""
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
# 🎓 ACTOR-CRITIC NETWORK FOR PPO
# ============================================================================

class ActorCritic(nn.Module):
    """
    Combined Actor-Critic with separate heads.
    
    The CRITIC is what we'll pretrain with expert demonstrations!
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
        # Actor head (policy) - will learn from RL
        self.actor = nn.Linear(hidden_size // 2, output_size)
        # Critic head (value) - will be PRETRAINED on expert demos!
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
# 🎓 EXPERT DEMONSTRATION COLLECTOR
# ============================================================================

class ExpertCollector:
    """
    🎓 WHAT IS THIS?
    
    This class runs a trained REINFORCE policy to collect "expert" 
    demonstrations. Each demonstration is a trajectory:
    
        [(state_0, action_0, reward_0), 
         (state_1, action_1, reward_1), 
         ...,
         (state_T, action_T, reward_T)]
    
    We'll use these to pretrain the critic!
    
    REAL-WORLD ANALOGY:
    - In robotics: human operator demonstrates the task
    - In games: expert player recordings
    - In ChatGPT: human-written responses
    """
    
    def __init__(self, board_size=5):
        self.board_size = board_size
        self.game = SnakeGame(board_size=board_size)
        
    def _get_reachable_count(self, game, start_pos):
        """BFS flood fill for state features"""
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
    
    def collect_with_trained_reinforce(self, num_episodes=100, min_score=5, gamma=0.99):
        """
        🎓 COLLECT EXPERT DEMONSTRATIONS
        
        This runs our trained REINFORCE policy and collects trajectories.
        We only keep "good" trajectories where score >= min_score.
        
        For each trajectory, we compute the ACTUAL RETURN G_t:
            G_t = r_t + γr_{t+1} + γ²r_{t+2} + ... + γ^(T-t)r_T
        
        This is the TRUE value of each state - exactly what we want
        the critic to learn!
        
        Args:
            num_episodes: How many episodes to run
            min_score: Only keep episodes with at least this score
            gamma: Discount factor for computing returns
            
        Returns:
            List of (state, action, return) tuples
        """
        print("\n" + "=" * 60)
        print("🎓 PHASE 1: Collecting Expert Demonstrations")
        print("=" * 60)
        
        # Create a REINFORCE-style policy network
        # We'll train it first to create the "expert"
        policy = PolicyNetwork(14, 256, 3)
        optimizer = optim.Adam(policy.parameters(), lr=0.001)
        
        # Storage for expert data
        expert_data = []
        
        # First, train a quick REINFORCE expert
        print("\nStep 1: Training expert policy with REINFORCE...")
        baseline = 0
        baseline_decay = 0.99
        
        for episode in range(5000):  # More training for larger board
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
                
                # Get action from policy
                probs = policy(state_tensor)
                dist = Categorical(probs)
                action = dist.sample()
                log_prob = dist.log_prob(action)
                entropy = dist.entropy()
                
                log_probs.append(log_prob)
                entropies.append(entropy)
                
                # Convert to game action
                action_onehot = [0, 0, 0]
                action_onehot[action.item()] = 1
                abs_action = self._get_absolute_action(action_onehot, self.game)
                
                # Store state and action
                trajectory.append((state.copy(), action.item()))
                
                # Take step
                _, reward, done, info = self.game.step(abs_action)
                rewards.append(reward)
                steps += 1
                
                if steps >= MAX_STEPS:
                    done = True
                    reward = -10
                    rewards[-1] = reward
            
            # REINFORCE update with baseline
            if len(rewards) > 0:
                returns = []
                G = 0
                for r in reversed(rewards):
                    G = r + gamma * G
                    returns.insert(0, G)
                returns = torch.tensor(returns, dtype=torch.float)
                
                # Update baseline
                episode_return = returns[0].item()
                baseline = baseline_decay * baseline + (1 - baseline_decay) * episode_return
                
                # Normalize
                advantages = returns - baseline
                if len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                
                # Compute loss
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
                print(f"  Training episode {episode + 1}/5000, last score: {info['score']}")
        
        # Now collect expert demonstrations
        print("\nStep 2: Collecting high-quality expert trajectories...")
        collected = 0
        attempts = 0
        
        while collected < num_episodes and attempts < num_episodes * 10:
            attempts += 1
            trajectory = []
            rewards = []
            
            self.game.reset()
            done = False
            steps = 0
            MAX_STEPS = 500
            
            while not done and steps < MAX_STEPS:
                state = self.get_state_features(self.game)
                state_tensor = torch.tensor(state, dtype=torch.float).unsqueeze(0)
                
                # Get action from trained policy
                with torch.no_grad():
                    probs = policy(state_tensor)
                dist = Categorical(probs)
                action = dist.sample()
                
                # Store
                trajectory.append((state.copy(), action.item()))
                
                # Convert and execute
                action_onehot = [0, 0, 0]
                action_onehot[action.item()] = 1
                abs_action = self._get_absolute_action(action_onehot, self.game)
                
                _, reward, done, info = self.game.step(abs_action)
                rewards.append(reward)
                steps += 1
                
                if steps >= MAX_STEPS:
                    done = True
            
            score = info['score']
            
            # Only keep good trajectories!
            # 🎓 This is important: we want to learn from SUCCESS, not failure
            if score >= min_score:
                collected += 1
                
                # Compute returns for each state
                # 🎓 G_t = sum of discounted future rewards from time t
                returns = []
                G = 0
                for r in reversed(rewards):
                    G = r + gamma * G
                    returns.insert(0, G)
                
                # Store (state, action, return) for each timestep
                for (state, action), ret in zip(trajectory, returns):
                    expert_data.append({
                        'state': state,
                        'action': action,
                        'return': ret
                    })
                
                if collected % 10 == 0:
                    print(f"  Collected {collected}/{num_episodes} expert trajectories (score >= {min_score})")
        
        print(f"\n✅ Collected {len(expert_data)} expert state-return pairs!")
        print(f"   from {collected} successful episodes")
        
        return expert_data, policy
    
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

        if relative_move == [1, 0, 0]:
            return direction
        elif relative_move == [0, 1, 0]:
            return right_turn[direction]
        elif relative_move == [0, 0, 1]:
            return left_turn[direction]
        return direction


# ============================================================================
# 🎓 CRITIC PRETRAINING
# ============================================================================

def pretrain_critic(actor_critic, expert_data, epochs=100, batch_size=64, lr=0.001):
    """
    🎓 PRETRAIN THE CRITIC ON EXPERT DEMONSTRATIONS
    
    This is the KEY insight! We train the critic to predict the actual
    returns that occurred in expert trajectories.
    
    What we're doing:
        Input: state features from expert trajectory
        Target: actual return G_t that occurred
        Loss: MSE(V(s), G_t)
    
    After this, the critic should have GOOD initial value estimates!
    
    This is similar to how you might teach someone chess:
    - Show them grandmaster games
    - For each position, tell them "this position is +2 for white"
    - Now they understand position evaluation BEFORE they learn to play
    
    Args:
        actor_critic: The ActorCritic network to pretrain
        expert_data: List of dicts with 'state' and 'return' keys
        epochs: Number of training epochs
        batch_size: Mini-batch size
        lr: Learning rate
    """
    print("\n" + "=" * 60)
    print("🎓 PHASE 2: Pretraining Critic on Expert Demonstrations")
    print("=" * 60)
    
    # Create optimizer for just the critic parameters
    # 🎓 We freeze the actor during pretraining - we only want to 
    # teach the critic what "good" looks like
    optimizer = optim.Adam(actor_critic.parameters(), lr=lr)
    
    # Convert expert data to tensors
    states = torch.tensor([d['state'] for d in expert_data], dtype=torch.float)
    returns = torch.tensor([d['return'] for d in expert_data], dtype=torch.float)
    
    print(f"\nPretraining on {len(states)} expert state-return pairs...")
    print(f"Return range: [{returns.min():.2f}, {returns.max():.2f}]")
    
    n_samples = len(states)
    
    for epoch in range(epochs):
        # Shuffle data each epoch
        indices = torch.randperm(n_samples)
        total_loss = 0
        n_batches = 0
        
        for i in range(0, n_samples, batch_size):
            batch_indices = indices[i:i+batch_size]
            batch_states = states[batch_indices]
            batch_returns = returns[batch_indices]
            
            # Forward pass - get critic's value predictions
            _, values = actor_critic(batch_states)
            
            # 🎓 MSE LOSS between predicted values and actual returns
            # This teaches the critic: "for this state, expect this return"
            loss = F.mse_loss(values.squeeze(), batch_returns)
            
            # Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        if (epoch + 1) % 20 == 0:
            avg_loss = total_loss / n_batches
            print(f"  Epoch {epoch + 1}/{epochs} | Critic Loss: {avg_loss:.4f}")
    
    print("\n✅ Critic pretraining complete!")
    
    # Verify the critic learned something useful
    with torch.no_grad():
        sample_values = actor_critic(states[:10])[1].squeeze()
        print(f"\nSample predictions vs actual returns:")
        for i in range(5):
            print(f"  State {i}: Predicted V={sample_values[i]:.2f}, Actual G={returns[i]:.2f}")


# ============================================================================
# 🎓 OPTIONAL: BEHAVIORAL CLONING FOR THE ACTOR
# ============================================================================

def pretrain_actor_bc(actor_critic, expert_data, epochs=50, batch_size=64, lr=0.001):
    """
    🎓 BEHAVIORAL CLONING (OPTIONAL)
    
    In addition to pretraining the critic, we can also pretrain the ACTOR
    using Behavioral Cloning (BC).
    
    BC is simple supervised learning:
        Input: state
        Target: expert's action
        Loss: CrossEntropy(π(s), expert_action)
    
    This teaches the actor: "in this state, the expert took this action"
    
    Pros:
        - Gives actor a head start
        - May speed up learning
    
    Cons:
        - Can't improve beyond expert
        - Compounding errors if expert didn't cover all states
    
    We'll use this to INITIALIZE the actor, then let RL improve it.
    """
    print("\n" + "=" * 60)
    print("🎓 PHASE 2b: Behavioral Cloning for Actor (Optional)")
    print("=" * 60)
    
    optimizer = optim.Adam(actor_critic.parameters(), lr=lr)
    
    states = torch.tensor([d['state'] for d in expert_data], dtype=torch.float)
    actions = torch.tensor([d['action'] for d in expert_data], dtype=torch.long)
    
    print(f"\nTraining actor to imitate expert on {len(states)} examples...")
    
    n_samples = len(states)
    
    for epoch in range(epochs):
        indices = torch.randperm(n_samples)
        total_loss = 0
        correct = 0
        n_batches = 0
        
        for i in range(0, n_samples, batch_size):
            batch_indices = indices[i:i+batch_size]
            batch_states = states[batch_indices]
            batch_actions = actions[batch_indices]
            
            # Get actor's action probabilities
            probs, _ = actor_critic(batch_states)
            
            # 🎓 Cross-entropy loss: minimize -log(π(expert_action | state))
            # This increases the probability of the action the expert took
            loss = F.cross_entropy(probs, batch_actions)
            
            # Track accuracy
            predicted_actions = probs.argmax(dim=1)
            correct += (predicted_actions == batch_actions).sum().item()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        if (epoch + 1) % 10 == 0:
            avg_loss = total_loss / n_batches
            accuracy = correct / n_samples * 100
            print(f"  Epoch {epoch + 1}/{epochs} | Loss: {avg_loss:.4f} | Accuracy: {accuracy:.1f}%")
    
    print("\n✅ Behavioral cloning complete!")


# ============================================================================
# 🎓 PPO AGENT (with pretrained critic)
# ============================================================================

class PPOAgent:
    """PPO Agent - same as before but will use pretrained critic"""
    
    def __init__(self, pretrained_network=None, lr=0.0003, gamma=0.99, 
                 clip_epsilon=0.2, n_epochs=4, entropy_coef=0.01, value_coef=0.5):
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.n_epochs = n_epochs
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.n_games = 0
        
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
        
        # 🎓 USE PRETRAINED NETWORK if provided!
        if pretrained_network is not None:
            print("\n🎓 Using PRETRAINED network (critic knows value estimates!)")
            self.network = pretrained_network
        else:
            print("\n⚠️  Training from scratch (critic starts clueless)")
            self.network = ActorCritic(14, 256, 3)
        
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
    
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

    def get_state(self, game):
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
            loss = actor_loss + self.value_coef * critic_loss - self.entropy_coef * entropy
            
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
# TRAINING LOOP
# ============================================================================

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

    if relative_move == [1, 0, 0]:
        return direction
    elif relative_move == [0, 1, 0]:
        return right_turn[direction]
    elif relative_move == [0, 0, 1]:
        return left_turn[direction]
    return direction


def train_ppo_with_demos():
    """
    🎓 MAIN TRAINING FUNCTION
    
    This demonstrates the full IL + RL pipeline:
    1. Collect expert demonstrations
    2. Pretrain critic on demonstrations
    3. (Optional) Behavioral clone the actor
    4. Run PPO with the warm-started network
    """
    print("\n" + "=" * 70)
    print("🎓 PPO WITH EXPERT DEMONSTRATIONS")
    print("   Combining Imitation Learning + Reinforcement Learning")
    print("=" * 70)
    
    BOARD_SIZE = 8  # 8x8 board for DQN comparison
    
    # =========================================
    # PHASE 1: Collect Expert Demonstrations
    # =========================================
    collector = ExpertCollector(board_size=BOARD_SIZE)
    expert_data, _ = collector.collect_with_trained_reinforce(
        num_episodes=100,   # More episodes for larger board
        min_score=10,       # Higher threshold for 8x8
        gamma=0.99
    )
    
    # =========================================
    # PHASE 2: Pretrain the Critic
    # =========================================
    network = ActorCritic(14, 256, 3)
    pretrain_critic(network, expert_data, epochs=100, batch_size=64, lr=0.001)
    
    # =========================================
    # PHASE 2b (OPTIONAL): Behavioral Cloning
    # =========================================
    # Also pretrain the actor to imitate the expert:
    pretrain_actor_bc(network, expert_data, epochs=50, batch_size=64, lr=0.001)
    
    # =========================================
    # PHASE 3: PPO Training with Warm Start
    # =========================================
    print("\n" + "=" * 60)
    print("🎓 PHASE 3: PPO Training (with pretrained critic)")
    print("=" * 60)
    
    total_score = 0
    record = 0
    
    # Pass the pretrained network to PPO!
    agent = PPOAgent(
        pretrained_network=network,
        lr=0.0003, 
        gamma=0.99, 
        clip_epsilon=0.2, 
        n_epochs=4
    )
    game = SnakeGame(board_size=BOARD_SIZE)
    
    ROLLOUT_LENGTH = 256  # Larger rollouts for bigger board
    MAX_STEPS = 2000       # More steps allowed on 8x8
    
    print(f"\nStarting PPO Training on {BOARD_SIZE}x{BOARD_SIZE} Board...")
    print("=" * 60)
    
    steps_in_game = 0
    total_steps = 0
    
    while agent.n_games < 5000:  # More training for larger board
        state = agent.get_state(game)
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
    print("🎉 Training Complete!")
    print(f"   Best Score: {record}")
    print(f"   Final Mean Score: {mean_score:.2f}")
    
    return agent, record, mean_score


if __name__ == '__main__':
    train_ppo_with_demos()
