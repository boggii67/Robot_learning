"""Tests for the learning rules in rl/algos/dqn.py.   Run:  pytest tests/test_dqn.py -v"""

import torch

from rl.algos.dqn import compute_td_target, project_distribution

# Next-state Q-values chosen so that vanilla and Double DQN give different targets:
# online net prefers action 1, target net prefers action 0.
ONLINE_Q = torch.tensor([[1.0, 5.0]])
TARGET_Q = torch.tensor([[3.0, 2.0]])


def targets(double, done=0.0):
    return compute_td_target(lambda _: ONLINE_Q, lambda _: TARGET_Q, rewards=torch.tensor([1.0]),
                             next_obs=torch.zeros(1, 4), dones=torch.tensor([done]), gamma=0.5, double=double)


# ---------------------------------------------------------------- steps 0 + 1


def test_vanilla_target():
    # 1 + 0.5 * max(3, 2)
    torch.testing.assert_close(targets(double=False), torch.tensor([2.5]))


def test_double_target():
    # online argmax = action 1, evaluated by the target net: 1 + 0.5 * 2
    torch.testing.assert_close(targets(double=True), torch.tensor([2.0]))


def test_terminal_has_no_bootstrap():
    torch.testing.assert_close(targets(double=False, done=1.0), torch.tensor([1.0]))


# ---------------------------------------------------------------- step 6: C51 projection

SUPPORT = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0])  # v_min 0, v_max 4, delta_z 1


def one_hot(i):
    p = torch.zeros(1, 5)
    p[0, i] = 1.0
    return p


def project(probs, reward, done, gamma):
    return project_distribution(probs, torch.tensor([reward]), torch.tensor([done]), SUPPORT, gamma)


def test_projection_exact_atom():
    # 1 + 1.0 * z_1 = 2 lands exactly on atom 2 (the l == u edge case!)
    torch.testing.assert_close(project(one_hot(1), 1.0, 0.0, 1.0), one_hot(2))


def test_projection_splits_between_atoms():
    # 0 + 0.5 * z_3 = 1.5 -> half on atom 1, half on atom 2
    torch.testing.assert_close(project(one_hot(3), 0.0, 0.0, 0.5), torch.tensor([[0.0, 0.5, 0.5, 0.0, 0.0]]))


def test_projection_terminal_ignores_next_state():
    # done: Tz = r = 2.5 for every atom -> split between atoms 2 and 3
    uniform = torch.full((1, 5), 0.2)
    torch.testing.assert_close(project(uniform, 2.5, 1.0, 0.9), torch.tensor([[0.0, 0.0, 0.5, 0.5, 0.0]]))


def test_projection_clips_to_support():
    torch.testing.assert_close(project(one_hot(0), 10.0, 0.0, 0.9), one_hot(4))


def test_projection_batch_sums_to_one():
    torch.manual_seed(0)
    probs = torch.softmax(torch.randn(16, 5), dim=-1)
    m = project_distribution(probs, torch.randn(16), (torch.rand(16) < 0.3).float(), SUPPORT, 0.97)
    assert m.shape == (16, 5)
    torch.testing.assert_close(m.sum(-1), torch.ones(16))
