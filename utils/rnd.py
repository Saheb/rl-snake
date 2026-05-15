import torch
import torch.nn as nn
import numpy as np


class RunningMeanStd:
    """Welford online algorithm for tracking mean and variance of a scalar stream."""

    def __init__(self, epsilon=1e-4):
        self.mean = 0.0
        self.var = 1.0
        self.count = epsilon

    def update(self, x: np.ndarray):
        batch_mean = x.mean()
        batch_var = x.var()
        batch_count = x.size

        total = self.count + batch_count
        delta = batch_mean - self.mean
        new_mean = self.mean + delta * batch_count / total
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        new_var = (m_a + m_b + delta ** 2 * self.count * batch_count / total) / total

        self.mean = new_mean
        self.var = new_var
        self.count = total

    @property
    def std(self):
        return max(np.sqrt(self.var), 1e-8)


class RNDNetwork(nn.Module):
    """Shared conv → flat architecture for both target and predictor."""

    def __init__(self, board_size, embedding_dim=256, ch1=32, ch2=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, ch1, kernel_size=3, padding=1), nn.LeakyReLU(0.01),
            nn.Conv2d(ch1, ch2, kernel_size=3, padding=1), nn.LeakyReLU(0.01),
            nn.Flatten(),
            nn.Linear(ch2 * board_size * board_size, embedding_dim),
        )

    def forward(self, x):
        return self.net(x)


class RND:
    """
    Random Network Distillation exploration bonus.

    Usage:
        rnd = RND(board_size=10)
        intrinsic = rnd.intrinsic_reward(state_tensor)   # (B,) numpy
        rnd.update(state_tensor)                          # train predictor
    """

    def __init__(self, board_size, embedding_dim=256, lr=1e-4, beta=0.05, device=None):
        self.beta = beta
        self.device = device or torch.device("cpu")
        self.reward_rms = RunningMeanStd()

        self.target = RNDNetwork(board_size, embedding_dim).to(self.device)
        self.predictor = RNDNetwork(board_size, embedding_dim).to(self.device)

        # Target is never trained.
        for p in self.target.parameters():
            p.requires_grad_(False)

        self.optimizer = torch.optim.Adam(self.predictor.parameters(), lr=lr)

    def intrinsic_reward(self, states: torch.Tensor) -> np.ndarray:
        """Return normalized intrinsic rewards for a batch of states."""
        with torch.no_grad():
            tgt = self.target(states)
            pred = self.predictor(states)
            errors = ((pred - tgt) ** 2).mean(dim=1).cpu().numpy()

        self.reward_rms.update(errors)
        return (errors / self.reward_rms.std).clip(-5, 5)

    def update(self, states: torch.Tensor) -> float:
        """Train predictor to match target; return mean loss."""
        tgt = self.target(states).detach()
        pred = self.predictor(states)
        loss = ((pred - tgt) ** 2).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()
