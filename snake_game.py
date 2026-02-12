import numpy as np
import random as r
from enum import Enum
import collections

EMPTY = 0
SNAKE = 1
FOOD = 2

class GameState(Enum):
    PLAYING = "playing"     # Game is ongoing (includes moments right after food collection)
    GAME_OVER = "game_over" # Snake collision with wall or itself
    WIN = "win"            # Board is full/max score reached


class SnakeGame:
    def __init__(self, board_size=10, seed=None):
        self.board_size = board_size
        if seed is not None:
            r.seed(seed)
        self.board = np.zeros((board_size, board_size))
        # Initialize snake in the middle
        self.snake_position = collections.deque([(board_size//2, board_size//2)])  # ordered!
        self.food_position = self._pick_food_position()
        self._update_board_food()
        self._update_board_snake()
        self.game_state = GameState.PLAYING
        self.score = 0
        self.head_history = collections.deque(maxlen=4) # Tracks the last 4 head positions

    def _pick_food_position(self):
        available_positions = set((x, y) 
            for x in range(self.board_size) 
            for y in range(self.board_size)) - set(self.snake_position)
        
        if not available_positions:
            return None  # Game is won - board is full
            
        return r.choice(list(available_positions))
    
    def _clear_board(self):
        self.board = np.zeros((self.board_size, self.board_size))
    
    def _update_board_snake(self):
        for x, y in self.snake_position:
            self.board[x][y] = SNAKE

    def _update_board_food(self):
        if self.food_position:
            x, y = self.food_position
            self.board[x][y] = FOOD

    def reset(self, seed=None):
        if seed is not None:
            r.seed(seed)
        self.game_state = GameState.PLAYING
        self.score = 0
        self.snake_position = collections.deque([(self.board_size // 2, self.board_size // 2)])
        self.food_position = self._pick_food_position()
        self._clear_board()
        self._update_board_snake()
        self._update_board_food()
        self.head_history = collections.deque(maxlen=4) # Tracks the last 4 head positions
        return self.get_state()
    
    def step(self, action):
        """
        Execute one time step within the environment
        Args:
            action: int (0: up, 1: right, 2: down, 3: left)
        """
        head_x, head_y = self.snake_position[-1]
        
        # Calculate old distance for reward shaping
        old_dist = 0
        if self.food_position:
            old_dist = abs(head_x - self.food_position[0]) + abs(head_y - self.food_position[1])
            
        new_head = self._move_snake(action)
        will_eat_food = (new_head == self.food_position)

        # --- Check for loops/waffling ---
        is_biting_neck = len(self.snake_position) > 1 and new_head == self.snake_position[-2]
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
            
            # Tiny step penalty
            reward = -0.01
            
            # Reward shaping based on distance
            if self.food_position:
                new_dist = abs(new_head[0] - self.food_position[0]) + abs(new_head[1] - self.food_position[1])
                if new_dist < old_dist:
                    reward += 0.1 # Moved closer
                else:
                    reward -= 0.1 # Moved further
            
            # Penalize waffling/loops
            if is_biting_neck or is_waffling:
                reward -= 0.5 

            self.head_history.append(new_head)
            done = False
        
        self._clear_board()
        self._update_board_snake()
        self._update_board_food()

        return self.get_state(), reward, done, {
            "score": self.score,
            "game_state": self.game_state.value,
            "snake_length": len(self.snake_position)
        }
    
    def _move_snake(self, action):
        """
        Calculate new head position based on action
        Args:
            action: int (0: up, 1: right, 2: down, 3: left)
        Returns:
            tuple: (x, y) coordinates of new head position
        """
        head_x, head_y = self.snake_position[-1]
        
        if action == 0:    # up
            new_head = (head_x - 1, head_y)
        elif action == 1:  # right
            new_head = (head_x, head_y + 1)
        elif action == 2:  # down
            new_head = (head_x + 1, head_y)
        else:             # left
            new_head = (head_x, head_y - 1)
        
        return new_head

    def _check_collision(self, new_head, will_eat_food=False):
        x, y = new_head
        
        # Check wall collision
        if x < 0 or x >= self.board_size or y < 0 or y >= self.board_size:
            return True
        
        # For self collision, exclude tail if not eating food (tail moves away)
        snake_body_to_check = list(self.snake_position)
        if not will_eat_food:
            snake_body_to_check = snake_body_to_check[1:] # Exclude tail
        
        if new_head in snake_body_to_check:
            return True
        
        return False
    
    def get_state(self):
        """Return current board state"""
        return np.copy(self.board)

    def print_board(self):
        """Print a visual representation of the current game state"""
        symbols = {EMPTY: '-', SNAKE: '*', FOOD: '@'}
        for row in self.board:
            print(' '.join(symbols[int(cell)] for cell in row))
        print(f"Score: {self.score}, Snake length: {len(self.snake_position)}")
        print()
