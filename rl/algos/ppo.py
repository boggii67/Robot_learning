"""Proximal Policy Optimization (Schulman et al. 2017). Continuous actions version.

Run:  python -m rl.algos.ppo --env-id Pendulum-v1        (then: Hopper-v5, HalfCheetah-v5)

Actor-critic: the actor pi(a|s) outputs a Gaussian (mean from a network, learned log_std),
the critic V(s) estimates state values.
Loop:
    1. Collect n_steps with the current policy into a RolloutBuffer (store log_probs + values)
    2. Compute advantages with GAE (see RolloutBuffer.compute_returns_and_advantages)
    3. For n_epochs, over minibatches:
         ratio      = exp(new_log_prob - old_log_prob)
         policy_loss = -min(ratio * A, clip(ratio, 1-eps, 1+eps) * A).mean()
         value_loss  = (V(s) - returns)^2 .mean()
         loss = policy_loss + vf_coef * value_loss - ent_coef * entropy
    4. Throw the data away (on-policy!) and repeat
The clip keeps each update small -> stable training.

Paper: https://arxiv.org/abs/1707.06347
Very useful: "The 37 Implementation Details of PPO" https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/
"""

import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal

from rl.common.buffers import RolloutBuffer
from rl.common.env_utils import describe_env, make_env
from rl.common.logger import Logger
from rl.common.networks import mlp
from rl.common.utils import make_run_dir, save_agent, set_seed


class Agent(nn.Module):
    def __init__(self, observation_space, action_space):
        super().__init__()
        obs_dim = int(np.prod(observation_space.shape))
        act_dim = int(np.prod(action_space.shape))
        self.actor_mean = mlp(obs_dim, act_dim)
        self.actor_log_std = nn.Parameter(torch.zeros(act_dim))
        self.critic = mlp(obs_dim, 1)

    def get_value(self, obs):
        return self.critic(obs).squeeze(-1)

    def get_action_and_value(self, obs, action=None):
        """Returns (action, log_prob, entropy, value). If `action` is given, evaluates that action
        instead of sampling - needed in the update step to get the NEW log_prob of OLD actions."""
        # TODO: build Normal(mean, std), sample if action is None,
        #       log_prob summed over action dims, entropy summed over action dims
        raise NotImplementedError

    def act(self, obs, deterministic=True):
        # TODO: return the mean action (deterministic) as a numpy array
        raise NotImplementedError


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--env-id", default="Pendulum-v1")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--total-steps", type=int, default=1_000_000)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--n-steps", type=int, default=2048, help="rollout length per update")
    p.add_argument("--n-epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip-eps", type=float, default=0.2)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--ent-coef", type=float, default=0.0)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--video", action="store_true")
    return p.parse_args()


def train(args):
    set_seed(args.seed)
    run_dir = make_run_dir("ppo", args.env_id, args.seed)
    env = make_env(args.env_id, args.seed, video_dir=f"{run_dir}/videos" if args.video else None)
    # Tip: MuJoCo envs train much better with observation/reward normalization:
    #   env = gym.wrappers.NormalizeObservation(env); env = gym.wrappers.NormalizeReward(env)
    describe_env(env)
    logger = Logger(run_dir)
    agent = Agent(env.observation_space, env.action_space)
    optimizer = torch.optim.Adam(agent.parameters(), lr=args.lr, eps=1e-5)

    # TODO: buffer = RolloutBuffer(...)
    # TODO: outer loop over updates (total_steps // n_steps):
    #   - collect n_steps (clip actions to the action space before env.step!)
    #   - compute last value, GAE
    #   - n_epochs x minibatch updates with the clipped loss
    #   - log losses, approx KL, clip fraction
    raise NotImplementedError("PPO training loop")

    save_agent(agent, run_dir)
    env.close()
    logger.close()


if __name__ == "__main__":
    train(parse_args())
