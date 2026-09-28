"""Tests for rl/common/buffers.py.   Run:  pytest tests/test_buffers.py -v"""

import numpy as np
import pytest
import torch

from rl.common.buffers import NStepBuffer, PrioritizedReplayBuffer, ReplayBuffer, SumTree


def fill(buffer, n, obs_dim=2):
    for i in range(n):
        buffer.add(np.full(obs_dim, i, dtype=np.float32), i % 2, float(i), np.full(obs_dim, i + 1, dtype=np.float32), False)


# ---------------------------------------------------------------- step 0: ReplayBuffer


def test_replay_buffer_wraps_around():
    buf = ReplayBuffer(5, (2,), (), action_dtype=np.int64)
    fill(buf, 7)
    assert len(buf) == 5, "size must stop at capacity"
    batch = buf.sample(200)
    seen = set(batch["obs"][:, 0].tolist())
    assert seen == {2.0, 3.0, 4.0, 5.0, 6.0}, "the two oldest transitions (0, 1) must be overwritten"


def test_replay_buffer_sample_format():
    buf = ReplayBuffer(10, (2,), (), action_dtype=np.int64)
    fill(buf, 10)
    batch = buf.sample(4)
    for key in ["obs", "actions", "rewards", "next_obs", "dones", "weights"]:
        assert isinstance(batch[key], torch.Tensor), f"{key} should be a torch tensor"
    assert batch["obs"].shape == (4, 2) and batch["obs"].dtype == torch.float32
    assert batch["actions"].shape == (4,) and batch["actions"].dtype == torch.int64
    assert batch["rewards"].shape == (4,) and batch["dones"].shape == (4,)
    assert torch.all(batch["weights"] == 1)
    assert len(batch["indices"]) == 4
    # transitions must stay consistent: next_obs = obs + 1, reward = obs in fill()
    assert torch.allclose(batch["next_obs"], batch["obs"] + 1)
    assert torch.allclose(batch["rewards"], batch["obs"][:, 0])


# ---------------------------------------------------------------- step 3: NStepBuffer


def step(nstep, t, reward, terminated=False, truncated=False):
    return nstep.append(f"s{t}", t, reward, f"s{t + 1}", terminated, truncated)


def test_nstep_returns_and_flush_on_termination():
    nstep = NStepBuffer(n=3, gamma=0.5)
    out = []
    out += step(nstep, 0, 1.0)
    out += step(nstep, 1, 2.0)
    assert out == [], "nothing is ready before n transitions were collected"
    out += step(nstep, 2, 3.0)
    assert len(out) == 1
    obs, action, ret, next_obs, done = out[0]
    assert (obs, action, next_obs, done) == ("s0", 0, "s3", False)
    assert ret == pytest.approx(1 + 0.5 * 2 + 0.25 * 3)  # 2.75

    out = step(nstep, 3, 4.0, terminated=True)
    # at the episode end all remaining transitions are flushed, each ending in the terminal state
    assert [o[0] for o in out] == ["s1", "s2", "s3"]
    assert [o[2] for o in out] == pytest.approx([2 + 0.5 * 3 + 0.25 * 4, 3 + 0.5 * 4, 4.0])
    assert all(o[3] == "s4" and o[4] is True for o in out)

    # queue must be empty for the next episode
    assert step(nstep, 10, 1.0) == []


def test_nstep_truncation_is_not_done():
    nstep = NStepBuffer(n=3, gamma=0.5)
    step(nstep, 0, 1.0)
    out = step(nstep, 1, 1.0, truncated=True)
    assert len(out) == 2
    assert all(o[4] is False or o[4] == 0 for o in out), "a time limit is not a real terminal state"


# ---------------------------------------------------------------- step 4: SumTree + PER


def test_sum_tree_total_and_update():
    tree = SumTree(4)
    for i, p in enumerate([1.0, 2.0, 3.0, 4.0]):
        tree.update(i, p)
    assert tree.total == pytest.approx(10.0)
    assert tree[2] == pytest.approx(3.0)
    tree.update(2, 0.5)
    assert tree.total == pytest.approx(7.5), "changing a leaf must propagate to the root"


def test_sum_tree_find():
    tree = SumTree(4)
    for i, p in enumerate([1.0, 2.0, 3.0, 4.0]):
        tree.update(i, p)
    # cumulative intervals: [0,1) -> 0, [1,3) -> 1, [3,6) -> 2, [6,10] -> 3
    assert tree.find(0.5) == 0
    assert tree.find(1.5) == 1
    assert tree.find(2.9) == 1
    assert tree.find(3.5) == 2
    assert tree.find(9.99) == 3


def test_sum_tree_proportional_non_power_of_two():
    rng = np.random.default_rng(0)
    priorities = np.array([1.0, 0.0, 3.0, 2.0, 4.0, 0.5])
    tree = SumTree(len(priorities))
    for i, p in enumerate(priorities):
        tree.update(i, p)
    counts = np.bincount([tree.find(u) for u in rng.uniform(0, tree.total, 20_000)], minlength=len(priorities))
    assert counts[1] == 0, "zero priority must never be sampled"
    np.testing.assert_allclose(counts / counts.sum(), priorities / priorities.sum(), atol=0.015)


def test_per_samples_proportional_to_priority():
    buf = PrioritizedReplayBuffer(4, (2,), (), action_dtype=np.int64, alpha=1.0, eps=0.0)
    fill(buf, 4)
    buf.update_priorities(np.arange(4), np.array([1.0, 2.0, 3.0, 4.0]))
    idx = np.concatenate([np.asarray(buf.sample(64, beta=0.4)["indices"]) for _ in range(300)])
    freq = np.bincount(idx, minlength=4) / len(idx)
    np.testing.assert_allclose(freq, [0.1, 0.2, 0.3, 0.4], atol=0.02)


def test_per_importance_weights():
    buf = PrioritizedReplayBuffer(4, (2,), (), action_dtype=np.int64, alpha=1.0, eps=0.0)
    fill(buf, 4)
    buf.update_priorities(np.arange(4), np.array([1.0, 2.0, 3.0, 4.0]))
    batch = buf.sample(64, beta=1.0)
    w = batch["weights"].numpy()
    idx = np.asarray(batch["indices"])
    assert w.max() <= 1.0 + 1e-6, "weights are normalized by their max"
    probs = np.array([0.1, 0.2, 0.3, 0.4])
    # w_i / w_j = (P_j / P_i) ** beta
    i, j = 0, int(np.argmax(idx != idx[0]))
    assert w[i] / w[j] == pytest.approx((probs[idx[j]] / probs[idx[i]]) ** 1.0, rel=1e-4)


def test_per_new_transitions_get_max_priority():
    buf = PrioritizedReplayBuffer(8, (2,), (), action_dtype=np.int64, alpha=1.0, eps=0.0)
    fill(buf, 2)
    buf.update_priorities(np.array([0, 1]), np.array([5.0, 1.0]))
    fill(buf, 1)  # index 2
    assert buf.tree[2] == pytest.approx(5.0)
