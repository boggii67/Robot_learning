"""Data storage for RL. Implementing these yourself is part of the learning!

ReplayBuffer  -> off-policy algorithms (DQN, SAC, TD3): store ALL past transitions,
                 train on random mini-batches (breaks correlation between consecutive samples).
RolloutBuffer -> on-policy algorithms (PPO, A2C): store only the CURRENT policy's last
                 N steps, compute advantages, train, then throw the data away.
"""

import numpy as np
import torch


class ReplayBuffer:
    """Fixed-size circular buffer of transitions (s, a, r, s', done)."""

    def __init__(self, capacity, obs_shape, action_shape, action_dtype=np.float32):
        self.capacity = capacity
        self.ptr = 0  # where the next transition goes
        self.size = 0  # how many transitions are stored

        # TODO: allocate numpy arrays of length `capacity` for:
        #   obs, next_obs (shape obs_shape), actions (action_shape, action_dtype),
        #   rewards (float), dones (float)
        # Hint: np.zeros((capacity, *obs_shape), dtype=np.float32)
        raise NotImplementedError("ReplayBuffer.__init__")

    def add(self, obs, action, reward, next_obs, done):
        # TODO: write the transition at index self.ptr, then advance ptr circularly
        #       (overwrite the oldest data when full) and update self.size.
        # Question to think about: should `done` be `terminated` or `terminated or truncated`?
        raise NotImplementedError("ReplayBuffer.add")

    def sample(self, batch_size, beta=None):
        # TODO: pick `batch_size` random indices in [0, self.size) and return a dict of
        #       torch tensors: {"obs", "actions", "rewards", "next_obs", "dones"}
        #       plus "weights": torch.ones(batch_size) and "indices": the numpy index array.
        #       (weights/indices are only used by PER; returning them here means the DQN training
        #        code can treat both buffers the same way. `beta` is ignored here.)
        raise NotImplementedError("ReplayBuffer.sample")

    def __len__(self):
        return self.size


class NStepBuffer:
    """Turns 1-step transitions into n-step transitions (Rainbow: --n-step N).

    Instead of bootstrapping after one step, sum n real rewards first:
        R_t^(n) = r_t + gamma r_{t+1} + ... + gamma^(n-1) r_{t+n-1}
        target  = R_t^(n) + gamma^n * max_a Q_target(s_{t+n}, a) * (1 - done)
    -> rewards propagate n times faster, with less bias from a still-wrong Q (but more variance).

    Usage in the training loop (sits in front of the replay buffer):
        for transition in nstep.append(obs, action, reward, next_obs, terminated, truncated):
            replay_buffer.add(*transition)

    append() returns a list of ready transitions (obs, action, n_step_return, next_obs, done):
        - usually [] (still collecting) or 1 transition (the oldest one in the queue is complete)
        - at episode end (terminated or truncated) ALL remaining transitions are flushed, with shorter
          returns, and the queue is emptied.
        - done = terminated of the LAST step included (a time-limit truncation is not a real end!)
    Simplification to think about: flushed transitions after a truncation contain k < n rewards,
    but the target will still use gamma^n. How could you fix that?
    """

    def __init__(self, n, gamma):
        self.n = n
        self.gamma = gamma
        # TODO: a queue for the last n transitions (hint: collections.deque)
        raise NotImplementedError("NStepBuffer.__init__")

    def append(self, obs, action, reward, next_obs, terminated, truncated):
        raise NotImplementedError("NStepBuffer.append")


class SumTree:
    """Binary tree where each parent stores the sum of its children; leaves store priorities.

    Lets PER sample index i with probability p_i / sum(p) in O(log N) instead of O(N):
    draw u ~ Uniform(0, total), then walk down from the root: go left if u <= left child's sum,
    otherwise subtract the left sum and go right.

    Array layout (capacity leaves, 2 * capacity - 1 nodes): node k has children 2k+1 and 2k+2,
    the leaf for data index i is node i + capacity - 1.
    """

    def __init__(self, capacity):
        self.capacity = capacity
        # TODO: self.tree = np.zeros(2 * capacity - 1)
        raise NotImplementedError("SumTree.__init__")

    @property
    def total(self):
        # TODO: the root node
        raise NotImplementedError("SumTree.total")

    def __getitem__(self, idx):
        """Priority of data index idx."""
        raise NotImplementedError("SumTree.__getitem__")

    def update(self, idx, priority):
        # TODO: set the leaf of data index idx, then propagate the change up to the root
        raise NotImplementedError("SumTree.update")

    def find(self, value):
        """Return the data index whose cumulative-priority interval contains `value` (0 <= value <= total)."""
        raise NotImplementedError("SumTree.find")


class PrioritizedReplayBuffer(ReplayBuffer):
    """Prioritized Experience Replay (Rainbow: --per). Schaul et al. 2015, https://arxiv.org/abs/1511.05952

    Sample transitions with large TD error more often - they have the most to teach:
        P(i) = p_i^alpha / sum_k p_k^alpha,      p_i = |td_error_i| + eps
    That biases the gradient, so correct with importance-sampling weights:
        w_i = (N * P(i))^(-beta) / max_j w_j     (beta annealed from ~0.4 to 1 during training)
    and use loss = mean(w_i * loss_i).

    New transitions get the current max priority, so each one is sampled at least once.
    Store p_i^alpha in the SumTree (not p_i).
    """

    def __init__(self, capacity, obs_shape, action_shape, action_dtype=np.float32, alpha=0.6, eps=1e-6):
        super().__init__(capacity, obs_shape, action_shape, action_dtype)
        self.alpha = alpha
        self.eps = eps
        self.max_priority = 1.0
        # TODO: self.tree = SumTree(capacity)
        raise NotImplementedError("PrioritizedReplayBuffer.__init__")

    def add(self, obs, action, reward, next_obs, done):
        # TODO: remember the index (self.ptr) BEFORE calling super().add(...), then set its
        #       tree value to self.max_priority ** self.alpha (max_priority is a raw p, without alpha)
        raise NotImplementedError("PrioritizedReplayBuffer.add")

    def sample(self, batch_size, beta=0.4):
        # TODO: stratified sampling: split [0, total) into batch_size equal segments, draw one
        #       value uniformly per segment, tree.find() it. Then compute P(i), the IS weights,
        #       and return the same dict as ReplayBuffer.sample (with real "weights" and "indices").
        raise NotImplementedError("PrioritizedReplayBuffer.sample")

    def update_priorities(self, indices, td_errors):
        """indices: numpy int array from sample(); td_errors: numpy array of |TD errors|."""
        # TODO: p = (|td| + eps) ** alpha -> tree.update; keep self.max_priority up to date
        raise NotImplementedError("PrioritizedReplayBuffer.update_priorities")


class RolloutBuffer:
    """Stores one rollout of `n_steps` from the current policy (for PPO / A2C)."""

    def __init__(self, n_steps, obs_shape, action_shape):
        self.n_steps = n_steps
        # TODO: allocate arrays for obs, actions, log_probs, rewards, dones, values
        #       and later advantages, returns
        raise NotImplementedError("RolloutBuffer.__init__")

    def add(self, obs, action, log_prob, reward, done, value):
        raise NotImplementedError("RolloutBuffer.add")

    def compute_returns_and_advantages(self, last_value, gamma=0.99, gae_lambda=0.95):
        """Generalized Advantage Estimation (GAE), the heart of PPO.

        delta_t = r_t + gamma * V(s_{t+1}) * (1 - done_t) - V(s_t)
        A_t     = delta_t + gamma * lambda * (1 - done_t) * A_{t+1}     (computed backwards!)
        R_t     = A_t + V(s_t)                                           (target for the critic)
        """
        # TODO: loop backwards over the rollout and fill self.advantages and self.returns
        raise NotImplementedError("RolloutBuffer.compute_returns_and_advantages")

    def get_minibatches(self, batch_size):
        # TODO: shuffle indices and yield dicts of torch tensors of size batch_size
        raise NotImplementedError("RolloutBuffer.get_minibatches")
