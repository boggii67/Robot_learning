"""Soft Actor-Critic (Haarnoja et al. 2018). Off-policy, continuous actions.

Run:  python -m rl.algos.sac --env-id Pendulum-v1        (then: HalfCheetah-v5, Reacher-v5)

Like DQN (replay buffer, target networks) but for continuous actions, plus an entropy bonus:
the agent maximizes  reward + alpha * entropy  -> it explores by staying as random as possible
while still solving the task.
Components:
    - actor: Gaussian policy, action = tanh(sample)  (tanh squashes into [-1, 1]; rescale to env bounds)
    - two critics Q1, Q2 + their target copies (take the min -> less overestimation)
Update (per gradient step):
    target  = r + gamma * (1 - terminated) * (min(Q1_t, Q2_t)(s', a') - alpha * log pi(a'|s')),  a' ~ pi(s')
    critic loss = MSE(Q1(s,a), target) + MSE(Q2(s,a), target)
    actor loss  = (alpha * log pi(a~|s) - min(Q1, Q2)(s, a~)).mean(),  a~ ~ pi(s) (reparameterized!)
    soft target update: theta_t <- tau * theta + (1 - tau) * theta_t
    (optional) automatic alpha tuning towards a target entropy of -act_dim

Paper: https://arxiv.org/abs/1812.05905
Later: add Hindsight Experience Replay (HER) to solve FetchReach / FetchPush (sparse rewards).
"""

import argparse

import numpy as np
import torch
import torch.nn as nn

from rl.common.buffers import ReplayBuffer
from rl.common.env_utils import describe_env, make_env
from rl.common.logger import Logger
from rl.common.networks import mlp
from rl.common.utils import make_run_dir, save_agent, set_seed


class Agent(nn.Module):
    """The actor. Critics live in train() since they aren't needed to act."""

    def __init__(self, observation_space, action_space):
        super().__init__()
        obs_dim = int(np.prod(observation_space.shape))
        act_dim = int(np.prod(action_space.shape))
        self.net = mlp(obs_dim, 2 * act_dim, hidden=(256, 256), activation=nn.ReLU)  # mean and log_std
        self.register_buffer("action_scale", torch.tensor((action_space.high - action_space.low) / 2, dtype=torch.float32))
        self.register_buffer("action_bias", torch.tensor((action_space.high + action_space.low) / 2, dtype=torch.float32))

    def sample(self, obs):
        """Returns (action, log_prob) with the tanh-squashing correction:
        log_prob = Normal.log_prob(u) - log(1 - tanh(u)^2 + 1e-6), summed over action dims."""
        # TODO
        raise NotImplementedError

    def act(self, obs, deterministic=True):
        # TODO: deterministic -> tanh(mean) * action_scale + action_bias
        raise NotImplementedError


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--env-id", default="Pendulum-v1")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--total-steps", type=int, default=100_000)
    p.add_argument("--buffer-size", type=int, default=1_000_000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--learning-starts", type=int, default=5_000)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--tau", type=float, default=0.005)
    p.add_argument("--actor-lr", type=float, default=3e-4)
    p.add_argument("--critic-lr", type=float, default=1e-3)
    p.add_argument("--alpha", type=float, default=0.2)
    p.add_argument("--autotune", action="store_true", help="learn alpha automatically")
    p.add_argument("--video", action="store_true")
    return p.parse_args()


def train(args):
    set_seed(args.seed)
    run_dir = make_run_dir("sac", args.env_id, args.seed)
    env = make_env(args.env_id, args.seed, video_dir=f"{run_dir}/videos" if args.video else None)
    describe_env(env)
    logger = Logger(run_dir)
    agent = Agent(env.observation_space, env.action_space)

    # TODO: critics q1, q2 = mlp(obs_dim + act_dim, 1, hidden=(256, 256), activation=nn.ReLU)
    # TODO: target critics, optimizers, replay buffer
    # TODO: main loop: random actions before learning_starts, then agent.sample(); store; update
    raise NotImplementedError("SAC training loop")

    save_agent(agent, run_dir)
    env.close()
    logger.close()


if __name__ == "__main__":
    train(parse_args())
