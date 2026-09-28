import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def mlp(in_dim, out_dim, hidden=(64, 64), activation=nn.Tanh, out_activation=None):
    """A plain fully-connected network: in_dim -> hidden... -> out_dim.

    Examples:
        q_net  = mlp(obs_dim, n_actions, hidden=(128, 128), activation=nn.ReLU)   # DQN
        policy = mlp(obs_dim, n_actions)                                          # logits for REINFORCE / PPO
        critic = mlp(obs_dim, 1)                                                  # state value V(s)
    """
    layers = []
    last = in_dim
    for h in hidden:
        layers += [nn.Linear(last, h), activation()]
        last = h
    layers.append(nn.Linear(last, out_dim))
    if out_activation is not None:
        layers.append(out_activation())
    return nn.Sequential(*layers)


# ---------------------------------------------------------------------------------------------
# Rainbow components. Implement these yourself! Tests: pytest tests/test_networks.py
# ---------------------------------------------------------------------------------------------


class NoisyLinear(nn.Module):
    """Linear layer with learnable, factorized Gaussian noise on weights and biases (Rainbow: --noisy).

    Paper: "Noisy Networks for Exploration", Fortunato et al. 2017, https://arxiv.org/abs/1706.10295

        y = (mu_w + sigma_w * eps_w) x + (mu_b + sigma_b * eps_b)

    The network learns HOW MUCH noise to use (sigma) -> exploration without epsilon-greedy.
    Factorized noise: draw eps_in (in_features) and eps_out (out_features) from N(0, 1), apply
    f(x) = sign(x) * sqrt(|x|), then eps_w = outer(f(eps_out), f(eps_in)) and eps_b = f(eps_out).

    Required attribute names (the tests and QNetwork.mean_sigma use them):
        weight_mu, weight_sigma  (nn.Parameter, shape [out, in])
        bias_mu, bias_sigma      (nn.Parameter, shape [out])
        weight_epsilon, bias_epsilon (register_buffer - not trained, but saved/moved with the model)
    Behavior:
        self.training == True  -> use noisy weights
        self.training == False -> use only the means (agent.eval() for deterministic evaluation)
    """

    def __init__(self, in_features, out_features, sigma0=0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sigma0 = sigma0
        # TODO: create the 4 parameters and 2 buffers, then call reset_parameters() and reset_noise()
        raise NotImplementedError("NoisyLinear.__init__")

    def reset_parameters(self):
        # TODO (paper, sec. 3.2, factorized case):
        #   mu     ~ Uniform(-1/sqrt(in), 1/sqrt(in))
        #   sigma  = sigma0 / sqrt(in)
        raise NotImplementedError("NoisyLinear.reset_parameters")

    @staticmethod
    def _scale_noise(size):
        x = torch.randn(size)
        return x.sign() * x.abs().sqrt()

    def reset_noise(self):
        # TODO: draw new factorized noise into weight_epsilon and bias_epsilon (use .copy_())
        raise NotImplementedError("NoisyLinear.reset_noise")

    def forward(self, x):
        # TODO: F.linear(x, weight, bias) with noisy or mean weights depending on self.training
        raise NotImplementedError("NoisyLinear.forward")


class DuelingHead(nn.Module):
    """Dueling architecture (Rainbow: --dueling). Wang et al. 2016, https://arxiv.org/abs/1511.06581

    Split Q into a state value and per-action advantages:
        Q(s, a) = V(s) + A(s, a) - mean_a' A(s, a')
    Subtracting the mean makes V and A identifiable (otherwise you could add c to V and subtract it from A).
    Intuition: in many states the action barely matters; V can learn that from every sample.

    Required attributes: self.value (in_dim -> n_atoms), self.advantage (in_dim -> n_actions * n_atoms),
    both built with `linear_cls` (nn.Linear or NoisyLinear).
    Output: [B, n_actions] if n_atoms == 1, else [B, n_actions, n_atoms] (combined with C51: the
    formula is applied per atom on the logits, before the softmax).
    """

    def __init__(self, in_dim, n_actions, n_atoms=1, linear_cls=nn.Linear):
        super().__init__()
        self.n_actions = n_actions
        self.n_atoms = n_atoms
        # TODO: self.value = ..., self.advantage = ...
        raise NotImplementedError("DuelingHead.__init__")

    def forward(self, x):
        # TODO: reshape value to [B, 1, n_atoms], advantage to [B, n_actions, n_atoms], combine,
        #       and squeeze the atom dim if n_atoms == 1
        raise NotImplementedError("DuelingHead.forward")


class QNetwork(nn.Module):
    """Builds the Q-network for any combination of flags. (Wiring only - already implemented.)

    body:  obs -> hidden layers (ReLU)
    head:  plain / dueling, with nn.Linear or NoisyLinear, 1 or n_atoms outputs per action

    forward(obs) -> Q-values [B, A]            (normal)
                 -> logits  [B, A, n_atoms]    (C51: a distribution over returns per action)
    """

    def __init__(self, obs_dim, n_actions, hidden=(128, 128), dueling=False, noisy=False,
                 n_atoms=1, v_min=-10.0, v_max=10.0):
        super().__init__()
        self.n_actions = n_actions
        self.n_atoms = n_atoms
        linear_cls = NoisyLinear if noisy else nn.Linear

        layers, last = [], obs_dim
        for i, h in enumerate(hidden):
            # Rainbow makes all fully-connected layers noisy; we keep the first layer plain.
            layers += [(nn.Linear if i == 0 else linear_cls)(last, h), nn.ReLU()]
            last = h
        self.body = nn.Sequential(*layers)

        if dueling:
            self.head = DuelingHead(last, n_actions, n_atoms, linear_cls)
        else:
            self.head = linear_cls(last, n_actions * n_atoms)

        # C51 support: the fixed return values z_i the distribution puts probability on
        self.register_buffer("support", torch.linspace(v_min, v_max, n_atoms))

    def forward(self, obs):
        out = self.head(self.body(obs))
        if self.n_atoms > 1:
            return out.view(-1, self.n_actions, self.n_atoms)
        return out.view(-1, self.n_actions)

    def dist(self, obs, log=False):
        """C51 only: probabilities (or log-probabilities) over atoms, [B, A, n_atoms]."""
        logits = self(obs)
        return F.log_softmax(logits, dim=-1) if log else F.softmax(logits, dim=-1)

    def q_values(self, obs):
        """Expected Q-values [B, A] - works for every variant."""
        if self.n_atoms > 1:
            return (self.dist(obs) * self.support).sum(-1)
        return self(obs)

    def reset_noise(self):
        for m in self.modules():
            if isinstance(m, NoisyLinear):
                m.reset_noise()

    def mean_sigma(self):
        """Average |sigma| of all noisy layers - watch it shrink as the agent explores less."""
        sigmas = [m.weight_sigma.abs().mean().item() for m in self.modules() if isinstance(m, NoisyLinear)]
        return sum(sigmas) / len(sigmas) if sigmas else 0.0
