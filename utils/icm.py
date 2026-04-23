import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class InverseDynamicsModel(nn.Module):
    """Inverse dynamics model that predicts action from state transition"""

    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        # Input: current_state + next_state (2 * state_dim)
        # Output: action probabilities
        self.network = nn.Sequential(
            nn.Linear(2 * state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, state, next_state):
        """Predict action given current and next state"""
        x = torch.cat([state, next_state], dim=1)
        return self.network(x)


class ForwardDynamicsModel(nn.Module):
    """Forward dynamics model that predicts next state from current state and action"""

    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        # Input: current_state + action
        # Output: next_state prediction
        self.network = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim),
        )

    def forward(self, state, action):
        """Predict next state given current state and action"""
        x = torch.cat([state, action], dim=1)
        return self.network(x)


class ICM:
    """Intrinsic Curiosity Module implementation"""

    def __init__(self, state_dim, action_dim, lr=0.001, eta=0.01):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.eta = eta  # Intrinsic reward scaling factor

        # Initialize models
        self.inverse_model = InverseDynamicsModel(state_dim, action_dim)
        self.forward_model = ForwardDynamicsModel(state_dim, action_dim)

        # Optimizers
        self.inverse_optimizer = torch.optim.Adam(
            self.inverse_model.parameters(), lr=lr
        )
        self.forward_optimizer = torch.optim.Adam(
            self.forward_model.parameters(), lr=lr
        )

        # Loss functions
        self.inverse_criterion = nn.CrossEntropyLoss()
        self.forward_criterion = nn.MSELoss()

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.inverse_model.to(self.device)
        self.forward_model.to(self.device)

    def compute_intrinsic_reward(self, state, action, next_state):
        """Compute intrinsic reward based on prediction error"""
        state = torch.tensor(np.array(state), dtype=torch.float).unsqueeze(0).to(self.device)
        action = torch.tensor(np.array(action), dtype=torch.float).unsqueeze(0).to(self.device)
        next_state = torch.tensor(np.array(next_state), dtype=torch.float).unsqueeze(0).to(self.device)

        # Forward model prediction
        predicted_next_state = self.forward_model(state, action)

        # Intrinsic reward is the squared error of forward model prediction
        intrinsic_reward = self.eta * torch.mean(
            (predicted_next_state - next_state) ** 2
        )

        return intrinsic_reward.item()

    def train_step(self, state, action, next_state):
        """Train ICM models on a single experience"""
        state = torch.tensor(np.array(state), dtype=torch.float).unsqueeze(0).to(self.device)
        action = torch.tensor(np.array(action), dtype=torch.float).unsqueeze(0).to(self.device)
        next_state = torch.tensor(np.array(next_state), dtype=torch.float).unsqueeze(0).to(self.device)

        # Train forward model
        self.forward_optimizer.zero_grad()
        predicted_next_state = self.forward_model(state, action)
        forward_loss = self.forward_criterion(predicted_next_state, next_state)
        forward_loss.backward()
        self.forward_optimizer.step()

        # Train inverse model
        self.inverse_optimizer.zero_grad()
        # Convert true action from one-hot shape (1, 3) to class index shape (1,)
        target_action = torch.argmax(action, dim=1)
        predicted_action_logits = self.inverse_model(state, next_state)
        inverse_loss = self.inverse_criterion(predicted_action_logits, target_action)
        inverse_loss.backward()
        self.inverse_optimizer.step()

        intrinsic_reward = self.eta * torch.mean(
            (predicted_next_state - next_state) ** 2
        )
        return intrinsic_reward.item()

    def train_batch(self, states, actions_one_hot, next_states):
        """Train ICM models on a batch of experiences"""
        # states, actions_one_hot, next_states are already tensors on self.device

        # Train forward model
        self.forward_optimizer.zero_grad()
        predicted_next_states = self.forward_model(states, actions_one_hot)
        forward_loss = self.forward_criterion(predicted_next_states, next_states)
        forward_loss.backward()
        self.forward_optimizer.step()

        # Train inverse model
        self.inverse_optimizer.zero_grad()
        # Convert true action from one-hot shape (batch, 3) to class index shape (batch,)
        target_actions = torch.argmax(actions_one_hot, dim=1)
        predicted_action_logits = self.inverse_model(states, next_states)
        inverse_loss = self.inverse_criterion(predicted_action_logits, target_actions)
        inverse_loss.backward()
        self.inverse_optimizer.step()

        return forward_loss.item(), inverse_loss.item()

    def save_models(self, path_prefix):
        """Save ICM model weights"""
        torch.save(self.inverse_model.state_dict(), f"{path_prefix}_inverse.pth")
        torch.save(self.forward_model.state_dict(), f"{path_prefix}_forward.pth")

    def load_models(self, path_prefix):
        """Load ICM model weights"""
        self.inverse_model.load_state_dict(torch.load(f"{path_prefix}_inverse.pth"))
        self.forward_model.load_state_dict(torch.load(f"{path_prefix}_forward.pth"))
