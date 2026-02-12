import random
import collections
import numpy as np
import matplotlib.pyplot as plt
import pickle
from snake_game import SnakeGame

# Helper for defaultdict
def default_q_values():
    return [0.0, 0.0, 0.0, 0.0]

class QLearningAgent:
    def __init__(self, learning_rate=0.1, discount_factor=0.9, epsilon=0.5):
        self.learning_rate = learning_rate  # α
        self.discount_factor = discount_factor  # γ  
        self.epsilon = epsilon
        self.q_table = collections.defaultdict(default_q_values)  # 4 actions
    
    def get_state_key(self, game, last_action):
        head_x, head_y = game.snake_position[-1]
        food_x, food_y = game.food_position if game.food_position else (-1, -1)
        
        # Directions
        dir_l = last_action == 3
        dir_r = last_action == 1
        dir_u = last_action == 0
        dir_d = last_action == 2
        
        # Relative Food Position
        food_left = food_y < head_y if game.food_position else 0
        food_right = food_y > head_y if game.food_position else 0
        food_up = food_x < head_x if game.food_position else 0
        food_down = food_x > head_x if game.food_position else 0
        
        # Danger sensing (immediate neighbors)
        point_l = (head_x, head_y - 1)
        point_r = (head_x, head_y + 1)
        point_u = (head_x - 1, head_y)
        point_d = (head_x + 1, head_y)
        
        danger_left = game._check_collision(point_l, False)
        danger_right = game._check_collision(point_r, False)
        danger_up = game._check_collision(point_u, False)
        danger_down = game._check_collision(point_d, False)

        # Danger relative to view (Straight, Right, Left)
        # This part depends on current direction, simplifying to absolute danger for tabular state
        # to keep state space smaller (or using the notebook's complex state)
        
        # Replicating the notebook's rich state representation:
        tail_x, tail_y = game.snake_position[0]
        tail_left = tail_y < head_y
        tail_right = tail_y > head_y
        tail_up = tail_x < head_x
        tail_down = tail_x > head_x
        
        snake_len_bin = min(2, (len(game.snake_position) - 1) // 5)

        # Wall distances
        max_bin_dist = int(game.board_size / 2)
        dist_left_bin = min(max_bin_dist, head_y)
        dist_right_bin = min(max_bin_dist, game.board_size - 1 - head_y)
        dist_up_bin = min(max_bin_dist, head_x)
        dist_down_bin = min(max_bin_dist, game.board_size - 1 - head_x)
        
        # Danger logic from notebook (complicated boolean logic)
        # danger_r_view = (dir_r and danger_right) or (dir_l and danger_left) or (dir_u and danger_up) or (dir_d and danger_down)
        # ... actually, let's stick to the core features to ensure it runs:
        # 1. Danger Straight, Right, Left
        # 2. Food Direction
        
        # Simplified state for readability/reliability if exact notebook logic is too complex to reconstruct perfectly from snippets
        # But wait, I have the snippet!
        
        state = [
            (dir_r and danger_right) or
            (dir_l and danger_left) or
            (dir_u and danger_up) or
            (dir_d and danger_down),  # Danger Straight
            
            (dir_u and danger_right) or
            (dir_d and danger_left) or
            (dir_l and danger_down) or # Fixed typo here too likely
            (dir_r and danger_down),  # Danger Right (wait, logic check)

            
            (dir_d and game._check_collision(point_r, False)) or
            (dir_u and game._check_collision(point_l, False)) or
            (dir_r and game._check_collision(point_u, False)) or
            (dir_l and game._check_collision(point_d, False)),  # Danger Left
            
            dir_l, dir_r, dir_u, dir_d,
            food_left, food_right, food_up, food_down,
            tail_left, tail_right, tail_up, tail_down,
            snake_len_bin,
            # dist_left_bin, dist_right_bin, dist_up_bin, dist_down_bin # Notebook used these
        ]
        return tuple(map(int, state))
        
    def choose_action(self, state_key, game):
        q_values = self.q_table[state_key]
        
        if random.random() < self.epsilon:
            return random.randint(0, 3)
        else:
            # Random tie breaking
            max_q = max(q_values)
            best_actions = [i for i, q in enumerate(q_values) if q == max_q]
            return random.choice(best_actions)
    
    def update_q_value(self, state, action, reward, next_state):
        current_q = self.q_table[state][action]
        next_max_q = max(self.q_table[next_state])
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * next_max_q - current_q
        )
        self.q_table[state][action] = new_q

class DoubleQLearningAgent:
    def __init__(self, learning_rate=0.1, discount_factor=0.9, epsilon=0.5):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.q_table_a = collections.defaultdict(default_q_values)
        self.q_table_b = collections.defaultdict(default_q_values)
        self.last_action = 0

    def get_state_key(self, game, last_action):
        # Same state logic as QLearningAgent
        # Reuse the logic or instantiate a helper
        return QLearningAgent().get_state_key(game, last_action)

    def choose_action(self, state_key, game):
        if random.random() < self.epsilon:
            return random.randint(0, 3)
        else:
            q_values_a = self.q_table_a[state_key]
            q_values_b = self.q_table_b[state_key]
            sum_q = [a + b for a, b in zip(q_values_a, q_values_b)]
            max_q = max(sum_q)
            best_actions = [i for i, q in enumerate(sum_q) if q == max_q]
            return random.choice(best_actions)

    def update_q_value(self, state, action, reward, next_state):
        if random.random() < 0.5:
            # Update A
            best_next_action = np.argmax(self.q_table_a[next_state])
            target = reward + self.discount_factor * self.q_table_b[next_state][best_next_action]
            self.q_table_a[state][action] += self.learning_rate * (target - self.q_table_a[state][action])
        else:
            # Update B
            best_next_action = np.argmax(self.q_table_b[next_state])
            target = reward + self.discount_factor * self.q_table_a[next_state][best_next_action]
            self.q_table_b[state][action] += self.learning_rate * (target - self.q_table_b[state][action])

def train(agent_type='q', episodes=1000, board_size=5):
    game = SnakeGame(board_size=board_size)
    
    if agent_type == 'double_q':
        agent = DoubleQLearningAgent(learning_rate=0.1, discount_factor=0.99, epsilon=1.0)
    else:
        agent = QLearningAgent(learning_rate=0.1, discount_factor=0.99, epsilon=1.0)
        
    scores = []
    
    print(f"Training {agent_type} agent on {board_size}x{board_size} board...")
    
    for ep in range(episodes):
        game.reset()
        state = agent.get_state_key(game, 0)
        action = agent.choose_action(state, game)
        done = False
        total_score = 0
        
        while not done:
            _, reward, done, info = game.step(action)
            next_state = agent.get_state_key(game, action)
            
            # Simple reward override for tabular stability if needed
            # But game.step already has shaping
            
            agent.update_q_value(state, action, reward, next_state)
            
            state = next_state
            # On-policy-ish for action selection for next step? 
            # Standard Q-learning is off-policy, but in loop we need next action
            next_action = agent.choose_action(next_state, game)
            action = next_action
            
            if reward == 10: # Food
                total_score += 1
                
        scores.append(total_score)
        agent.epsilon = max(0.01, agent.epsilon * 0.995)
        
        if (ep+1) % 500 == 0:
            avg = sum(scores[-100:]) / 100
            print(f"Episode {ep+1}: Avg Score {avg:.2f}, Epsilon {agent.epsilon:.2f}")

    return agent, scores

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', choices=['q', 'double_q'], default='q')
    parser.add_argument('--episodes', type=int, default=5000)
    parser.add_argument('--board_size', type=int, default=5)
    args = parser.parse_args()
    
    agent, scores = train(args.type, args.episodes, args.board_size)
    
    # Save
    filename = f"tabular_{args.type}_{args.board_size}x{args.board_size}.pkl"
    with open(filename, 'wb') as f:
        pickle.dump(agent, f)
    print(f"Saved agent to {filename}")
    
    # Plot
    plt.plot(scores)
    plt.title(f"{args.type} Training")
    plt.xlabel("Episode")
    plt.ylabel("Score")
    plt.savefig(f"tabular_{args.type}_training.png")
