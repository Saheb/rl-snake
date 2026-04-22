import numpy as np

class SumTree:
    """
    A binary tree data structure where the parent's value is the sum of its children.
    Used for $O(\log N)$ operations in Prioritized Experience Replay.
    """
    def __init__(self, capacity):
        self.capacity = capacity
        # The tree itself (internal nodes + leaves)
        self.tree = np.zeros(2 * capacity - 1)
        # Array to hold the actual transitions (state, action, reward, next_state, done)
        self.data = np.zeros(capacity, dtype=object)
        self.write_ptr = 0
        self.n_entries = 0

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total_priority(self):
        # Root node holds the sum of all priorities
        return self.tree[0]

    def add(self, priority, data):
        # Add a priority score and its corresponding experience to the tree
        idx = self.write_ptr + self.capacity - 1

        self.data[self.write_ptr] = data
        self.update(idx, priority)

        self.write_ptr += 1
        if self.write_ptr >= self.capacity:
            self.write_ptr = 0

        if self.n_entries < self.capacity:
            self.n_entries += 1

    def update(self, idx, priority):
        # Update the priority of a leaf and propagate the change up the tree
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)

    def get_leaf(self, s):
        # Get a leaf node based on a random sample 's'
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return (idx, self.tree[idx], self.data[data_idx])
