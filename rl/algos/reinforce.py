"""REINFORCE (Williams 1992): the simplest policy-gradient algorithm.

Run:  python -m rl.algos.reinforce --env-id CartPole-v1

Instead of learning Q-values, directly learn a policy pi(a|s) (a network outputting action logits).
    1. Play one full episode with the current policy, record log pi(a_t|s_t) and r_t
    2. Compute the discounted return from each step:  G_t = r_t + gamma * G_{t+1}
    3. loss = -sum_t log pi(a_t|s_t) * G_t        ("make actions that led to high return more likely")
Improvements to try afterwards (this is how you get to A2C):
    - normalize G_t (subtract mean, divide by std) -> much lower variance
    - subtract a learned baseline V(s_t) instead: advantage = G_t - V(s_t)
"""

import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from rl.common.env_utils import describe_env, make_env
from rl.common.logger import Logger
from rl.common.networks import mlp
from rl.common.utils import make_run_dir, save_agent, set_seed


class Agent(nn.Module):
    def __init__(self, observation_space, action_space):
        super().__init__()
        obs_dim = int(np.prod(observation_space.shape))
        self.policy = mlp(obs_dim, action_space.n)  # outputs logits

    def get_action(self, obs):
        """Sample an action and return (action, log_prob) - used during training."""
        # TODO: logits -> Categorical(logits=...) -> sample -> return action.item(), dist.log_prob(action)
        raise NotImplementedError

    def act(self, obs, deterministic=True):
        # TODO: for evaluation: argmax of logits (deterministic) or a sample
        raise NotImplementedError


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--env-id", default="CartPole-v1")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--episodes", type=int, default=1_000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--video", action="store_true")
    return p.parse_args()


def train(args):
    set_seed(args.seed)
    run_dir = make_run_dir("reinforce", args.env_id, args.seed)
    env = make_env(args.env_id, args.seed, video_dir=f"{run_dir}/videos" if args.video else None)
    describe_env(env)
    logger = Logger(run_dir)
    agent = Agent(env.observation_space, env.action_space)
    optimizer = torch.optim.Adam(agent.parameters(), lr=args.lr)

    global_step = 0
    for episode in range(args.episodes):
        obs, info = env.reset(seed=args.seed + episode)
        log_probs, rewards = [], []
        done = False
        while not done:
            action, log_prob = agent.get_action(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            log_probs.append(log_prob)
            rewards.append(reward)
            done = terminated or truncated
            global_step += 1
        logger.log_episode(info, global_step)

        # TODO: compute returns G_t (loop backwards over rewards)
        # TODO: loss = -(torch.stack(log_probs) * returns).sum(); optimizer step
        # logger.log_scalar("losses/policy_loss", loss.item(), global_step)

    save_agent(agent, run_dir)
    env.close()
    logger.close()


if __name__ == "__main__":
    train(parse_args())
