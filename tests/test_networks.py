"""Tests for the network components in rl/common/networks.py.   Run:  pytest tests/test_networks.py -v"""

import torch
import torch.nn.functional as F

from rl.common.networks import DuelingHead, NoisyLinear, QNetwork

# ---------------------------------------------------------------- step 2: DuelingHead


def test_dueling_shape_and_identifiability():
    torch.manual_seed(0)
    head = DuelingHead(16, n_actions=4)
    x = torch.randn(8, 16)
    q = head(x)
    assert q.shape == (8, 4)
    # mean over actions of Q = V, because the mean advantage is subtracted
    torch.testing.assert_close(q.mean(dim=1), head.value(x).squeeze(-1))


def test_dueling_with_atoms():
    head = DuelingHead(16, n_actions=4, n_atoms=11)
    x = torch.randn(8, 16)
    out = head(x)
    assert out.shape == (8, 4, 11)
    torch.testing.assert_close(out.mean(dim=1), head.value(x).view(8, 11))


# ---------------------------------------------------------------- step 5: NoisyLinear


def test_noisy_linear_shapes_and_params():
    layer = NoisyLinear(5, 3)
    assert layer.weight_mu.shape == (3, 5) and layer.weight_sigma.shape == (3, 5)
    assert layer.bias_mu.shape == (3,) and layer.bias_sigma.shape == (3,)
    assert layer(torch.randn(7, 5)).shape == (7, 3)
    trainable = {n for n, p in layer.named_parameters()}
    assert trainable == {"weight_mu", "weight_sigma", "bias_mu", "bias_sigma"}, "epsilon must be a buffer"


def test_noisy_linear_noise_changes_in_train_mode():
    torch.manual_seed(0)
    layer = NoisyLinear(5, 3).train()
    x = torch.randn(4, 5)
    y1 = layer(x)
    torch.testing.assert_close(layer(x), y1, msg="same noise -> same output until reset_noise()")
    layer.reset_noise()
    assert not torch.allclose(layer(x), y1), "reset_noise() must draw new noise"


def test_noisy_linear_eval_uses_means():
    layer = NoisyLinear(5, 3).eval()
    x = torch.randn(4, 5)
    torch.testing.assert_close(layer(x), F.linear(x, layer.weight_mu, layer.bias_mu))


def test_noisy_linear_init():
    layer = NoisyLinear(100, 10, sigma0=0.5)
    assert layer.weight_mu.abs().max() <= 1 / 100 ** 0.5 + 1e-6
    torch.testing.assert_close(layer.weight_sigma, torch.full((10, 100), 0.5 / 100 ** 0.5))


# ---------------------------------------------------------------- all flags together


def test_qnetwork_all_variants_build():
    obs = torch.randn(3, 8)
    for dueling in (False, True):
        for noisy in (False, True):
            for n_atoms in (1, 51):
                net = QNetwork(8, 4, dueling=dueling, noisy=noisy, n_atoms=n_atoms)
                assert net.q_values(obs).shape == (3, 4)
                if n_atoms > 1:
                    torch.testing.assert_close(net.dist(obs).sum(-1), torch.ones(3, 4))
