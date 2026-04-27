import numpy as np
import random
from utils.sum_tree import SumTree

class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay (PER) buffer using a SumTree.
    Extracts transitions based on their absolute Temporal Difference (TD) error.
    """
    def __init__(self, capacity=100000, alpha=0.6, beta=0.4, beta_increment_per_sampling=0.001, priority_cap=1.0):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment_per_sampling = beta_increment_per_sampling
        
        # Hard ceiling on TD-error priority — prevents death states dominating the batch
        self.priority_cap = priority_cap
        
        # Epsilon prevents zero-probability sampling for perfectly predicted states
        self.epsilon = 0.01 
        
        # Keep track of the absolute highest priority so new experiences get reviewed immediately
        self.max_priority = 1.0 
        
        self.tree = SumTree(capacity)

        # --- Instrumentation: track terminal-state buffer & sampling composition ---
        # Updated in sample(); inspected by the training loop each checkpoint.
        self.last_sample_terminal_count = 0
        self.last_sample_total = 0

    def __len__(self):
        return self.tree.n_entries

    def add(self, transition):
        # New experiences are pushed with max_priority to guarantee they are sampled at least once
        priority = self.max_priority
        self.tree.add(priority, transition)

    def sample(self, batch_size):
        batch = []
        indices = []
        is_weights = np.empty(batch_size, dtype=np.float32)

        # Calculate the size of each priority segment
        segment = self.tree.total_priority() / batch_size
        
        # Anneal beta towards 1.0 (Importance Sampling bias correction becomes stronger later in training)
        self.beta = np.min([1.0, self.beta + self.beta_increment_per_sampling])
        
        # Calculate max weight for normalization
        if self.tree.n_entries > 0:
            p_min_raw = np.min(self.tree.tree[-self.tree.capacity:][self.tree.tree[-self.tree.capacity:] > 0])
            total_p = self.tree.total_priority()
            p_min = p_min_raw / total_p if total_p > 0 else 1e-8
            p_min = max(p_min, 1e-8)
            max_weight = (p_min * self.tree.n_entries) ** (-self.beta)
        else:
            max_weight = 1.0

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            
            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get_leaf(s)
            
            if data is None:
                continue

            # Calculate Importance Sampling (IS) weight to correct the bias created by non-uniform sampling
            sampling_probability = p / self.tree.total_priority()
            is_weight = np.power(self.tree.n_entries * sampling_probability, -self.beta) / max_weight
            
            indices.append(idx)
            batch.append(data)
            is_weights[i] = is_weight

        # Instrumentation: count terminals in this sampled batch (data[4] == done flag).
        self.last_sample_total = len(batch)
        self.last_sample_terminal_count = sum(1 for t in batch if t[4])

        return batch, indices, is_weights

    def buffer_terminal_frac(self):
        """Fraction of currently stored transitions that are terminal (done=True). O(capacity) walk."""
        n = self.tree.n_entries
        if n == 0:
            return 0.0
        terminal = 0
        for transition in self.tree.data[:n]:
            if transition is not None and transition[4]:
                terminal += 1
        return terminal / n

    def last_sample_terminal_frac(self):
        if self.last_sample_total == 0:
            return 0.0
        return self.last_sample_terminal_count / self.last_sample_total

    def update_priorities(self, indices, errors):
        # Update tree priorities based on absolute TD error
        for idx, error in zip(indices, errors):
            # Add epsilon to prevent 0 priority, raise to the power of alpha to scale priority impact.
            # PRIORITY ROOF: Cap the max priority so death states (high TD error) cannot monopolize
            # the training batches. Without this cap, the network would only ever practice dying,
            # dragging all Q-values negative ("Existence is pain" collapse).
            raw_priority = (abs(error) + self.epsilon) ** self.alpha
            priority = min(raw_priority, self.priority_cap)
            self.tree.update(idx, priority)
            
            # Keep tracking the global maximum priority to assign to new experiences
            self.max_priority = max(self.max_priority, priority)
