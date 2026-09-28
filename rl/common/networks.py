import torch.nn as nn


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
