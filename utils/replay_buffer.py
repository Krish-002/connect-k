from __future__ import annotations

import numpy as np


class _SumTree:
    """
    Binary sum-tree for O(log n) priority updates and sampling.

    Leaf nodes (indices capacity-1 .. 2*capacity-2) store individual
    priorities; internal nodes store the sum of their subtree.
    """

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self._tree = np.zeros(2 * capacity, dtype=np.float64)
        self._write = 0       # next leaf to overwrite
        self._size = 0

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    @property
    def total(self) -> float:
        return float(self._tree[1])  # root holds the grand total

    @property
    def size(self) -> int:
        return self._size

    def set(self, leaf_idx: int, priority: float) -> None:
        """Update priority at leaf_idx (0-based) and propagate to root."""
        node = leaf_idx + self.capacity  # convert to tree index
        delta = priority - self._tree[node]
        self._tree[node] = priority
        # Propagate upward
        node >>= 1
        while node >= 1:
            self._tree[node] += delta
            node >>= 1

    def add(self, priority: float) -> int:
        """Write priority to the next leaf, return its 0-based leaf index."""
        leaf_idx = self._write
        self.set(leaf_idx, priority)
        self._write = (self._write + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)
        return leaf_idx

    def get(self, value: float) -> tuple[int, float]:
        """
        Find the leaf whose cumulative prefix sum first exceeds `value`.
        Returns (leaf_idx, priority).
        """
        node = 1  # start at root
        while node < self.capacity:
            left = node * 2
            if value <= self._tree[left]:
                node = left
            else:
                value -= self._tree[left]
                node = left + 1
        leaf_idx = node - self.capacity
        return leaf_idx, float(self._tree[node])

    def max_priority(self) -> float:
        return float(self._tree[self.capacity: self.capacity + self._size].max())


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay buffer (Schaul et al., 2015).

    Transitions are stored as (state, action, reward, next_state, done).
    Sampling probability for transition i:
        P(i) = p_i^alpha / sum_k p_k^alpha
    Importance-sampling weights:
        w_i = (N * P(i))^{-beta} / max_j w_j
    """

    _EPS: float = 1e-6   # added to |td_error| so no priority is zero

    def __init__(self, capacity: int, alpha: float = 0.6, beta: float = 0.4) -> None:
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta

        self._tree = _SumTree(capacity)

        # Storage arrays — pre-allocated once we see the first state shape.
        self._states: np.ndarray | None = None
        self._next_states: np.ndarray | None = None
        self._actions = np.zeros(capacity, dtype=np.int64)
        self._rewards = np.zeros(capacity, dtype=np.float32)
        self._dones = np.zeros(capacity, dtype=np.float32)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Add a transition with maximum current priority."""
        if self._states is None:
            shape = state.shape
            self._states = np.zeros((self.capacity, *shape), dtype=np.float32)
            self._next_states = np.zeros((self.capacity, *shape), dtype=np.float32)

        # New transitions get the current max priority so they are sampled soon.
        priority = self._tree.max_priority() if self._tree.size > 0 else 1.0
        leaf_idx = self._tree.add(priority ** self.alpha)

        self._states[leaf_idx] = state
        self._actions[leaf_idx] = action
        self._rewards[leaf_idx] = reward
        self._next_states[leaf_idx] = next_state
        self._dones[leaf_idx] = float(done)

    def sample(
        self, batch_size: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[int]]:
        """
        Sample a batch using prioritized sampling.

        Returns:
            states       (batch_size, *state_shape) float32
            actions      (batch_size,)              int64
            rewards      (batch_size,)              float32
            next_states  (batch_size, *state_shape) float32
            dones        (batch_size,)              float32
            weights      (batch_size,)              float32  IS corrections
            indices      list[int]                           leaf indices for priority updates
        """
        if self._tree.size < batch_size:
            raise ValueError(
                f"Buffer has only {self._tree.size} transitions, need {batch_size}."
            )

        indices: list[int] = []
        priorities: list[float] = []

        # Stratified sampling: divide total priority into batch_size segments.
        segment = self._tree.total / batch_size
        for i in range(batch_size):
            lo, hi = segment * i, segment * (i + 1)
            value = np.random.uniform(lo, hi)
            leaf_idx, priority = self._tree.get(value)
            indices.append(leaf_idx)
            priorities.append(priority)

        # Importance-sampling weights: w_i = (N * P(i))^{-beta}
        n = self._tree.size
        probs = np.array(priorities, dtype=np.float64) / self._tree.total
        weights = (n * probs) ** (-self.beta)
        weights = (weights / weights.max()).astype(np.float32)  # normalise to [0, 1]

        idx = np.array(indices, dtype=np.int64)
        return (
            self._states[idx],
            self._actions[idx],
            self._rewards[idx],
            self._next_states[idx],
            self._dones[idx],
            weights,
            indices,
        )

    def update_priorities(self, indices: list[int], td_errors: np.ndarray) -> None:
        """Recompute priorities from TD errors and update the sum-tree."""
        for leaf_idx, td_error in zip(indices, td_errors):
            priority = (abs(float(td_error)) + self._EPS) ** self.alpha
            self._tree.set(leaf_idx, priority)

    def update_beta(self, beta: float) -> None:
        """Anneal beta towards 1 over training."""
        self.beta = beta

    def __len__(self) -> int:
        return self._tree.size


# ----------------------------------------------------------------------
# Sanity check
# ----------------------------------------------------------------------

if __name__ == "__main__":
    STATE_SHAPE = (3, 6, 7)
    CAPACITY = 2048
    BATCH_SIZE = 32
    N_PUSH = 1000

    buf = PrioritizedReplayBuffer(capacity=CAPACITY, alpha=0.6, beta=0.4)

    # Push random transitions
    for _ in range(N_PUSH):
        s = np.random.rand(*STATE_SHAPE).astype(np.float32)
        a = np.random.randint(0, 7)
        r = float(np.random.choice([-1.0, 0.0, 1.0]))
        ns = np.random.rand(*STATE_SHAPE).astype(np.float32)
        d = bool(np.random.rand() < 0.1)
        buf.push(s, a, r, ns, d)

    print(f"Buffer size after {N_PUSH} pushes: {len(buf)}")

    # Sample a batch
    states, actions, rewards, next_states, dones, weights, indices = buf.sample(BATCH_SIZE)

    print(f"\nSample shapes:")
    print(f"  states:      {states.shape}   dtype={states.dtype}")
    print(f"  actions:     {actions.shape}    dtype={actions.dtype}")
    print(f"  rewards:     {rewards.shape}    dtype={rewards.dtype}")
    print(f"  next_states: {next_states.shape}   dtype={next_states.dtype}")
    print(f"  dones:       {dones.shape}    dtype={dones.dtype}")
    print(f"  weights:     {weights.shape}    dtype={weights.dtype}")
    print(f"  indices:     {len(indices)} leaf indices")

    print(f"\nIS weight stats:  min={weights.min():.4f}  max={weights.max():.4f}  mean={weights.mean():.4f}")
    assert weights.max() <= 1.0 + 1e-6, "Weights should be normalised to [0, 1]"
    assert weights.min() > 0.0, "All weights should be positive"

    # Update priorities with fake TD errors and re-sample to confirm it runs
    fake_td = np.random.rand(BATCH_SIZE).astype(np.float32)
    buf.update_priorities(indices, fake_td)
    _, _, _, _, _, weights2, _ = buf.sample(BATCH_SIZE)
    print(f"\nAfter priority update — weight stats:  min={weights2.min():.4f}  max={weights2.max():.4f}")

    # Beta annealing
    buf.update_beta(0.7)
    print(f"\nbeta updated to {buf.beta}")

    print("\nAll checks passed.")
