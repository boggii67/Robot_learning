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

    def sample(self, batch_size):
        # TODO: pick `batch_size` random indices in [0, self.size) and return a dict of
        #       torch tensors: {"obs", "actions", "rewards", "next_obs", "dones"}
        raise NotImplementedError("ReplayBuffer.sample")

    def __len__(self):
        return self.size


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
