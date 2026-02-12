import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt
from test_improvements import SnakeGame  # Reusing the existing game logic

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

    def save(self, file_name='model.pth'):
        torch.save(self.state_dict(), file_name)

# ============================================================================
# Q-TRAINER
# ============================================================================
import collections

# ============================================================================
# Q-TRAINER
# ============================================================================
# ============================================================================
# Q-TRAINER
# ============================================================================
class QTrainer:
    def __init__(self, model, target_model, lr, gamma, n_steps=1):
        self.lr = lr
        self.gamma = gamma
        self.n_steps = n_steps
        self.model = model
        self.target_model = target_model # Target Network
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
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
            done = (done, )

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
                gamma_n = self.gamma ** self.n_steps
                Q_new = reward[idx] + gamma_n * next_pred_target[idx][next_best_actions[idx]]

            target[idx][torch.argmax(action[idx]).item()] = Q_new
    
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        self.optimizer.step()

# ============================================================================
# DQN AGENT
# ============================================================================
class DQNAgent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0 
        self.gamma = 0.9 
        self.memory = deque(maxlen=100_000) 
        
        # N-Step Learning
        self.n_steps = 4
        self.n_step_buffer = deque(maxlen=self.n_steps)
        
        # Dueling Network (Input 14)
        self.model = DuelingQNet(14, 256, 3) 
        self.target_model = DuelingQNet(14, 256, 3)
        self.target_model.load_state_dict(self.model.state_dict())
        self.target_model.eval()
        
        self.trainer = QTrainer(self.model, self.target_model, lr=0.001, gamma=self.gamma, n_steps=self.n_steps)

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
        max_search = len(game.snake_position) * 3 
        
        while queue and count < max_search:
            curr = queue.popleft()
            count += 1
            cx, cy = curr
            neighbors = [(cx-1, cy), (cx+1, cy), (cx, cy-1), (cx, cy+1)]
            for nx, ny in neighbors:
                if (0 <= nx < game.board_size and 
                    0 <= ny < game.board_size and 
                    (nx, ny) not in obstacles and 
                    (nx, ny) not in visited):
                    visited.add((nx, ny))
                    queue.append((nx, ny))
        return count

    def get_state(self, game):
        # ... (Identical to previous, omitting for brevity in diff but assuming kept)
        # Note: Since replace_file_content replaces exact range, I must ensure get_state remains.
        # However, the user prompt implies just updating the Agent init and memory. 
        # I will replace the START (init) and END (remember/train) and LEAVE get_state if possible?
        # No, I must provide contiguous block.
        # Re-implementing get_state exactly as before since I'm replacing the whole class block.
        
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
        if dir_u:   idx = 0
        elif dir_r: idx = 1
        elif dir_d: idx = 2
        else:       idx = 3 # dir_l
        
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

    def remember(self, state, action, reward, next_state, done):
        # Apply N-Step Buffer logic
        self.n_step_buffer.append((state, action, reward, next_state, done))
        
        if len(self.n_step_buffer) < self.n_steps:
            return # Buffer not full yet
            
        # Compute N-step Reward
        # R = r_0 + gamma*r_1 + gamma^2*r_2 ...
        R, next_s, d = self._get_n_step_info()
        state_0, action_0 = self.n_step_buffer[0][:2]
        
        self.memory.append((state_0, action_0, R, next_s, d))
        
        return state_0, action_0, R, next_s, d # Return computed transition for online training

    def _get_n_step_info(self):
        R = 0
        for i, transition in enumerate(self.n_step_buffer):
            r = transition[2]
            R += r * (self.gamma ** i)
            if transition[4]: # done is True
                return R, transition[3], True # next_state is terminal state of this step
        
        # If no done, next_state is the next_state of the LAST item in buffer
        return R, self.n_step_buffer[-1][3], False

    def train_long_memory(self):
        if len(self.memory) > 1000:
            mini_batch = random.sample(self.memory, 1000) 
        else:
            mini_batch = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_batch)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        # N-step online training (Optional, but helps latency)
        # We need to construct the N-step transition immediately if possible, but we can't look 3 steps ahead instantly.
        # Standard approach: Simple DQN does 1-step online. N-step usually relies on replay.
        # We will use the valid transition processed by remember().
        pass # We delegate online training to the training loop using the returned transition

    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
        # Epsilon decay: 1.0 -> 0.01 slower decay now
        # Decay over ~2000 games for the long 16k run
        self.epsilon = max(0.01, 1.0 - (self.n_games * 0.0005)) 
        final_move = [0,0,0]
        
        if random.random() < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move

def get_absolute_action(move, game):
    # move is [straight, right, left]
    # We need to convert this relative move to absolute action (0: up, 1: right, 2: down, 3: left)
    
    # Determine current direction
    head = game.snake_position[-1]
    if len(game.snake_position) > 1:
        neck = game.snake_position[-2]
        if head[0] < neck[0]:
            current_dir = 0 # UP
        elif head[0] > neck[0]:
            current_dir = 2 # DOWN
        elif head[1] < neck[1]:
            current_dir = 3 # LEFT
        else:
            current_dir = 1 # RIGHT
    else:
        # Default start direction (e.g., right) or infer from last action if passed
        current_dir = 1 # Assume growing right initially

    # Clockwise order: [UP, RIGHT, DOWN, LEFT] -> [0, 1, 2, 3]
    clock_wise = [0, 1, 2, 3]
    idx = clock_wise.index(current_dir)

    if np.array_equal(move, [1, 0, 0]): # Straight
        new_dir = clock_wise[idx]
    elif np.array_equal(move, [0, 1, 0]): # Right turn
        new_dir = clock_wise[(idx + 1) % 4]
    else: # [0, 0, 1] Left turn
        new_dir = clock_wise[(idx - 1) % 4]
        
    return new_dir

def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = DQNAgent()
    game = SnakeGame(board_size=8) # Increased to 8x8
    
    MAX_STEPS = 2000 # Increased step limit for larger board
    print("Starting Rainbow DQN Training (Dueling + 3-Step) on 8x8 Board...")
    
    steps_in_game = 0
    while agent.n_games < 16000: # Train for 16000 games
        # get old state
        state_old = agent.get_state(game)

        # get move
        final_move = agent.get_action(state_old)
        
        # perform move and get new state
        action = get_absolute_action(final_move, game)
        _, reward, done, info = game.step(action)
        score = info['score']
        
        steps_in_game += 1
        if steps_in_game > MAX_STEPS:
            done = True
            reward = -10 # Penalize timeout
        
        state_new = agent.get_state(game)

        # remember returns None if buffer not full, or tuple if N-step ready
        n_step_transition = agent.remember(state_old, final_move, reward, state_new, done)
        
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
                agent.memory.append((state_0, action_0, R, next_s, d))
                agent.train_short_memory(state_0, action_0, R, next_s, d)
            
            # train long memory, plot result
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()
            steps_in_game = 0 # Reset counter
            
            # Update Target Network every 50 games
            if agent.n_games % 50 == 0:
                agent.update_target_network()

            if score > record:
                record = score
                agent.model.save()

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            
            if agent.n_games % 100 == 0:
                print(f"Game {agent.n_games} | Score: {score} | Record: {record} | Mean Score: {mean_score:.2f} | Epsilon: {agent.epsilon:.2f}")

    print("\nTraining Complete!")
    print(f"Best Score: {record}")
    print(f"Final Mean Score: {mean_score:.2f}")

if __name__ == '__main__':
    train()
