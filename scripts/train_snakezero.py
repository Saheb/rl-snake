import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from collections import deque
import math
import os
import time

# ==========================================
# 1. FAST GAME LOGIC
# ==========================================
# ==========================================
# 1. FAST GAME LOGIC
# ==========================================
class SnakeGame:
    def __init__(self, board_size=10):
        self.board_size = board_size
        self.reset()

    def reset(self):
        # Start in middle, length 3
        c = self.board_size // 2
        self.snake = deque([(c, c), (c, c-1), (c, c-2)])
        self.score = 0
        self.steps = 0
        self.done = False
        self.place_food()
        self.place_food()
        return self.get_state()

    def place_food(self):
        snake_set = set(self.snake)
        attempts = 0
        while attempts < 100:
            r, c = random.randint(0, self.board_size-1), random.randint(0, self.board_size-1)
            if (r, c) not in snake_set:
                self.food = (r, c)
                return
            attempts += 1
        # Fallback linear search
        for r in range(self.board_size):
            for c in range(self.board_size):
                if (r, c) not in snake_set:
                    self.food = (r, c)
                    return
        self.done = True # No space left

    def get_legal_moves(self):
        # 0: UP (-1, 0), 1: RIGHT (0, 1), 2: DOWN (1, 0), 3: LEFT (0, -1)
        # Prevent 180 reverse
        head = self.snake[0]
        neck = self.snake[1]
        diff = (head[0]-neck[0], head[1]-neck[1]) # current direction
        
        legal = []
        moves = [(-1,0), (0,1), (1,0), (0,-1)]
        for action, (dr, dc) in enumerate(moves):
            # Check reverse
            if (diff[0] + dr == 0) and (diff[1] + dc == 0):
                continue
            
            # Check collision (walls/body) for basic pruning?
            # AlphaZero typically allows illegal moves but masks them, or learns to avoid.
            # Here we'll return all non-reverse moves, but mark game over if hit.
            legal.append(action)
        return legal

    def step(self, action):
        if self.done: return self.get_state(), 0, True

        self.steps += 1
        moves = [(-1,0), (0,1), (1,0), (0,-1)]
        dr, dc = moves[action]
        head_r, head_c = self.snake[0]
        new_head = (head_r + dr, head_c + dc)

        # 1. Wall collision
        if not (0 <= new_head[0] < self.board_size and 0 <= new_head[1] < self.board_size):
            self.done = True
            return self.get_state(), -1.0, True

        # 2. Body collision
        if new_head in self.snake and new_head != self.snake[-1]:
            self.done = True
            return self.get_state(), -1.0, True

        # Move snake
        self.snake.appendleft(new_head)
        
        # 3. Eat food
        if new_head == self.food:
            self.score += 1
            if len(self.snake) >= self.board_size * self.board_size:
                self.done = True # Win
                return self.get_state(), 1.0, True
            self.place_food()
            reward = 1.0
        else:
            self.snake.pop()
            reward = 0.0
            
            # Starvation/Loop penalty
            if self.steps > self.board_size * self.board_size * 2:
                self.done = True
                reward = -0.5

        return self.get_state(), reward, self.done

    def get_state_tensor(self):
        # EGOCENTRIC VIEW
        # Return 7x7 grid centered on head
        # View Range 3 means +/- 3 cells from head -> 7x7
        # Channels: 0: Head, 1: Body, 2: Food, 3: Wall/Boundary (Implicitly handled or explicit?)
        # Let's stick to 3 channels for now:
        # Channel 0: Objects (1=Body, 1=Head) -> Actually let's keep separate
        # But wait, we need to show WALLS. 
        # Best way for ConvNet:
        # Ch 0: Self (Head + Body)
        # Ch 1: Food
        # Ch 2: Walls (1.0 if wall/out of bounds)
        
        view_range = 3
        view_size = 2 * view_range + 1 # 7
        
        state = np.zeros((3, view_size, view_size), dtype=np.float32)
        head_r, head_c = self.snake[0]
        
        # We iterate over the 7x7 window
        for i in range(view_size):
            for j in range(view_size):
                # Calculate corresponding board coordinate
                r = head_r - view_range + i
                c = head_c - view_range + j
                
                # Check bounds (Walls)
                if not (0 <= r < self.board_size and 0 <= c < self.board_size):
                    state[2, i, j] = 1.0 # Wall channel
                    continue
                
                # It's on the board
                # Food
                if (r, c) == self.food:
                    state[1, i, j] = 1.0
                
                # Snake Body/Head
                # We need to check if (r, c) is in self.snake
                # This is O(N) inside O(49) -> OK for small snake
                # Optimization: convert snake to set if slow, but len < 100 usually
                if (r, c) in self.snake:
                     state[0, i, j] = 1.0

        return state

    def get_state(self):
        # Returns (Grid, Compass)
        grid = self.get_state_tensor()
        
        head = self.snake[0]
        # Vector from head to food
        dy = self.food[0] - head[0]
        dx = self.food[1] - head[1]
        
        # Normalize
        dist = math.sqrt(dy*dy + dx*dx)
        if dist == 0: 
            compass = np.zeros(2, dtype=np.float32)
        else:
            compass = np.array([dy/dist, dx/dist], dtype=np.float32)
            
        return (grid, compass)

    def clone(self):
        new_game = SnakeGame(self.board_size)
        new_game.snake = deque(self.snake)
        new_game.food = self.food
        new_game.score = self.score
        new_game.steps = self.steps
        new_game.done = self.done
        return new_game

# ==========================================
# 2. NEURAL NETWORK
# ==========================================
class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        res = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += res
        return F.relu(x)

class SnakeNet(nn.Module):
    def __init__(self, board_size=10, num_channels=3, num_res_blocks=3, num_filters=64):
        super().__init__()
        self.board_size = board_size 
        # Observation is ALWAYS 7x7 (view_range=3)
        self.view_size = 7
        
        # Initial conv
        self.conv_in = nn.Conv2d(num_channels, num_filters, 3, padding=1)
        self.bn_in = nn.BatchNorm2d(num_filters)
        
        # Residual tower
        self.res_blocks = nn.ModuleList([ResBlock(num_filters) for _ in range(num_res_blocks)])
        
        # Policy Head
        self.policy_conv = nn.Conv2d(num_filters, 2, 1)
        self.policy_bn = nn.BatchNorm2d(2)
        # Flatten size: 2*7*7=98 + 2(compass) = 100
        self.policy_fc = nn.Linear(98 + 2, 4)

        # Value Head
        self.value_conv = nn.Conv2d(num_filters, 1, 1)
        self.value_bn = nn.BatchNorm2d(1)
        # Flatten size: 1*7*7=49 + 2(compass) = 51
        self.value_fc1 = nn.Linear(49 + 2, 64)
        self.value_fc2 = nn.Linear(64, 1)

    def forward(self, grid, compass):
        # grid: (B, 3, 7, 7)
        # compass: (B, 2)
        x = grid
        x = F.relu(self.bn_in(self.conv_in(x)))
        for block in self.res_blocks:
            x = block(x)
            
        # Policy
        p = F.relu(self.policy_bn(self.policy_conv(x)))
        p = p.view(p.size(0), -1) # (B, 98)
        p = torch.cat([p, compass], dim=1) # (B, 100)
        pi = self.policy_fc(p)
        pi = F.softmax(pi, dim=1)
        
        # Value
        v = F.relu(self.value_bn(self.value_conv(x)))
        v = v.view(v.size(0), -1) # (B, 49)
        v = torch.cat([v, compass], dim=1) # (B, 51)
        v = F.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v))
        
        return pi, v

# ==========================================
# 3. MCTS ENGINE
# ==========================================
class MCTSNode:
    def __init__(self, state, parent=None, prior_prob=1.0):
        self.state = state
        self.parent = parent
        self.children = {} # action -> node
        self.prior_prob = prior_prob
        self.visit_count = 0
        self.value_sum = 0

    def is_leaf(self):
        return len(self.children) == 0

    def value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count

class MCTS:
    def __init__(self, model, num_simulations=50, c_puct=1.0, device='cpu'):
        self.model = model
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.device = device

    def search(self, game):
        # Root node
        root = MCTSNode(game.clone())
        
        # Expand root immediately to get priors
        grid, compass = game.get_state()
        grid_tensor = torch.tensor(grid, dtype=torch.float32).unsqueeze(0).to(self.device)
        compass_tensor = torch.tensor(compass, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            policy_logits, _ = self.model(grid_tensor, compass_tensor)
            policy_probs = F.softmax(policy_logits, dim=1).cpu().numpy()[0]
        
        # Add Dirichlet noise to root for exploration
        noise = np.random.dirichlet([0.3] * 4)
        policy_probs = 0.75 * policy_probs + 0.25 * noise
        
        legal_moves = game.get_legal_moves()
        for action in legal_moves:
            root.children[action] = MCTSNode(None, parent=root, prior_prob=policy_probs[action])

        for _ in range(self.num_simulations):
            node = root
            sim_game = game.clone() 

            # 1. Selection
            while not node.is_leaf():
                action, node = self._select_child(node)
                sim_game.step(action)

            # 2. Expansion & Evaluation
            if not sim_game.done:
                grid, compass = sim_game.get_state()
                grid_tensor = torch.tensor(grid, dtype=torch.float32).unsqueeze(0).to(self.device)
                compass_tensor = torch.tensor(compass, dtype=torch.float32).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    policy_logits, value = self.model(grid_tensor, compass_tensor)
                    policy_probs = F.softmax(policy_logits, dim=1).cpu().numpy()[0]
                    value = value.item()
                
                legal_moves = sim_game.get_legal_moves()
                for action in legal_moves:
                    node.children[action] = MCTSNode(None, parent=node, prior_prob=policy_probs[action])
            else:
                # Terminal state value
                # If won: +1, if lost: -1 (approx)
                # We use the reward from the last step, but values are usually [-1, 1]
                # Step reward gives +1 for food, -1 for death.
                # Let's verify what step() returned for the last move.
                # Ideally MCTS simulates until terminal or depth limit.
                # Here we trust the scalar value from the last step transition if we want exact rewards?
                # Actually, step() returns intermediate reward.
                # For simplicity in this demo:
                # Win = 1, Loss = -1, Tie/Timeout = 0
                if sim_game.score >= sim_game.board_size**2: value = 1.0
                else: value = -1.0

            # 3. Backpropagation
            self._backpropagate(node, value)
        
        # Return visit counts -> policy
        counts = np.zeros(4)
        for action, child in root.children.items():
            counts[action] = child.visit_count
        
        if np.sum(counts) == 0: return np.ones(4) / 4.0 # Fallback
        return counts / np.sum(counts)

    def _select_child(self, node):
        best_score = -float('inf')
        best_action = -1
        best_child = None

        for action, child in node.children.items():
            u = self.c_puct * child.prior_prob * math.sqrt(node.visit_count) / (1 + child.visit_count)
            # AlphaZero value is from perspective of current player. Snake is 1-player.
            # So standard UCB: Q + U
            score = child.value() + u
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child
        
        return best_action, best_child

    def _backpropagate(self, node, value):
        while node:
            node.visit_count += 1
            node.value_sum += value
            node = node.parent

import json

# ==========================================
# 4. TRAINING LOOP
# ==========================================
class SnakeZeroTrainer:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Running on {self.device}")
        
        self.model = SnakeNet(board_size=10).to(self.device) # MCTS model is now board-invariant
        # Initialize MCTS with dummy board, it will adapt
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.mcts = MCTS(self.model, num_simulations=50, device=self.device) # Low sims for speed
        self.replay_buffer = deque(maxlen=5000)
        self.batch_size = 64
        self.game_log = []
        self.recorded_games = []

    def self_play(self, episode):
        # TRAIN ON 10x10 BOARD with Compass!
        game = SnakeGame(board_size=10)
        states = []
        game_history = []
        
        # Recording data for visualization
        replay = {
            "episode": episode,
            "initial_snake": list(game.snake),
            "initial_food": game.food,
            "steps": [],
            "score": 0
        }
        
        while not game.done:
            # Get MCTS policy
            action_probs = self.mcts.search(game)
            
            # Sample action (exploration) or greedy (exploitation)
            # Usually temp=1 for first N moves, then temp=0
            if len(game_history) < 10:
                action = np.random.choice(4, p=action_probs)
            else:
                action = np.argmax(action_probs)
            
            # Store state and target policy
            state_train = game.get_state()
            states.append((state_train, action_probs))
            
            # Step
            _, reward, done = game.step(action)
            game_history.append(reward)
            
            # Record step (action, food_eaten?)
            replay["steps"].append({
                "action": int(action),
                "snake": [list(x) for x in game.snake],
                "food": game.food
            })
            
        replay["score"] = game.score
        
        # Save all games for full learning history analysis
        self.recorded_games.append(replay)
        
        # Save live progress for visualization
        try:
             with open("assets/snakezero_live.json", "w") as f:
                json.dump(self.recorded_games, f)
        except Exception as e:
            print(f"Error saving live log: {e}")
            
        # Game Over - Compute Value Targets
            
        # Game Over - Compute Value Targets
        # Outcome: Win (+1), Loss (-1), or discounted returns?
        # AlphaZero uses strictly { -1, 0, 1 } for 2-player.
        # For Snake, we can use: +1 (Win), -1 (Death), scaled score?
        # Let's try simple: Reward if len > X? Or just standard -1 for death?
        # Let's use computed returns from the game.
        # Simple approach: Final result propagated back.
        # If score increased significantly -> +1, else -1?
        # Let's try: Value = Discounted Return
        
        returns = []
        G = 0
        if game.score >= 3: # Arbitrary "good" threshold for micro-board
            final_val = 1.0 
        else:
            final_val = -1.0 # Died early
            
        # Simply assign final value to all steps (Monte Carlo outcome)
        # Or Td-lambda. Let's stick to AlphaZero style: Outcome.
        # If score > mean_score -> +1 else -1? 
        # For tabula rasa, let's just use final return.
        
        for state, probs in states:
            self.replay_buffer.append((state, probs, final_val))
            
        return game.score, game.steps

    def train_step(self):
        if len(self.replay_buffer) < self.batch_size: return 0
        
        batch = random.sample(self.replay_buffer, self.batch_size)
        
        grids = np.array([x[0][0] for x in batch])
        compasses = np.array([x[0][1] for x in batch])
        
        grid_batch = torch.tensor(grids, dtype=torch.float32).to(self.device)
        compass_batch = torch.tensor(compasses, dtype=torch.float32).to(self.device)
        
        policy_target = torch.tensor(np.array([x[1] for x in batch]), dtype=torch.float32).to(self.device)
        value_target = torch.tensor(np.array([x[2] for x in batch]), dtype=torch.float32).view(-1, 1).to(self.device)
        
        self.optimizer.zero_grad()
        p_logits, v_pred = self.model(grid_batch, compass_batch)
        
        # Loss: Value MSE + Policy CrossEntropy
        value_loss = F.mse_loss(v_pred, value_target)
        policy_loss = -torch.mean(torch.sum(policy_target * p_logits, dim=1))
        
        loss = value_loss + policy_loss
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

    def evaluate(self, num_games=10, board_size=4):
        total_score = 0
        for _ in range(num_games):
            game = SnakeGame(board_size=board_size)
            while not game.done:
                # Use policy network directly for fast eval, or MCTS for strong eval
                # Using MCTS for consistency with training policy distribution
                action_probs = self.mcts.search(game)
                action = np.argmax(action_probs)
                game.step(action)
            total_score += game.score
        return total_score / num_games

    def test_generalization(self):
        # Zero-shot transfer to larger boards
        sizes = [6, 8, 10]
        results = {}
        for sz in sizes:
            avg_score = self.evaluate(num_games=3, board_size=sz)
            results[sz] = avg_score
        return results

    def run(self, num_episodes=50):
        print(f"Starting SnakeZero training for {num_episodes} episodes...")
        best_avg_score = -1
        
        for e in range(num_episodes):
            game_score, game_steps = self.self_play(e)
            loss = self.train_step()
            
            self.game_log.append(game_score)
            rolling_avg = sum(self.game_log[-10:]) / min(len(self.game_log), 10)
            
            if e % 10 == 0:
                eval_score = self.evaluate(num_games=5, board_size=4)
                gen_scores = self.test_generalization()
                gen_str = ", ".join([f"{k}x{k}:{v:.1f}" for k, v in gen_scores.items()])
                
                print(f"Ep {e}: Score {game_score}, Steps {game_steps}, Loss {loss:.4f}, Rolling {rolling_avg:.1f}, Eval {eval_score:.1f}, [GEN]: {gen_str}")
                
                if eval_score > best_avg_score:
                    best_avg_score = eval_score
                    torch.save(self.model.state_dict(), "checkpoints/snakezero_4x4_invariant.pth")
                    print(f"  Moves >> New Best Model Saved! ({best_avg_score:.1f})")

        # Save final
        torch.save(self.model.state_dict(), "checkpoints/snakezero_6x6_final.pth")
        print("Final model saved.")
        
        # Save recorded games
        with open("assets/snakezero_games.json", "w") as f:
            json.dump(self.recorded_games, f)
        print(f"Saved {len(self.recorded_games)} replay games to snakezero_games.json")

if __name__ == "__main__":
    trainer = SnakeZeroTrainer()
    print("Starting SnakeZero training for 2000 episodes...")
    trainer.run(num_episodes=2000)
