"""Deep Q-Network (Mnih et al. 2015). Discrete actions only.

Run:  python -m rl.algos.dqn --env-id CartPole-v1
Goal: CartPole return of 500 (the maximum) within ~50k-100k steps.

Core idea: learn Q(s, a) = expected return after taking action a in state s.
    target = r + gamma * max_a' Q_target(s', a') * (1 - terminated)
    loss   = (Q(s, a) - target)^2      (or Huber loss)
Two tricks make it stable:
    1. Replay buffer  -> train on random old transitions, not just the latest one
    2. Target network -> a slowly updated copy of Q used to compute the target
Exploration: epsilon-greedy (random action with prob. epsilon, decayed over time).

Paper: https://www.nature.com/articles/nature14236
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
    def __init__(self, observation_space, action_space):
        super().__init__()
        obs_dim = int(np.prod(observation_space.shape))
        n_actions = action_space.n
        # TODO: create self.q_net, a network obs_dim -> n_actions (one Q-value per action)
        #       e.g. mlp(obs_dim, n_actions, hidden=(128, 128), activation=nn.ReLU)
        raise NotImplementedError("Agent.__init__")

    def forward(self, obs):
        return self.q_net(obs)

    def act(self, obs, deterministic=True):
        # TODO: convert obs to a float tensor, return argmax_a Q(obs, a) as a Python int
        raise NotImplementedError("Agent.act")


def linear_schedule(start, end, duration, step):
    """Epsilon decay: goes linearly from `start` to `end` over `duration` steps."""
    slope = (end - start) / duration
    return max(slope * step + start, end)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--env-id", default="CartPole-v1")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--total-steps", type=int, default=100_000)
    p.add_argument("--lr", type=float, default=2.5e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--buffer-size", type=int, default=10_000)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--learning-starts", type=int, default=1_000, help="fill buffer before training")
    p.add_argument("--train-every", type=int, default=10, help="gradient step every N env steps")
    p.add_argument("--target-update-every", type=int, default=500)
    p.add_argument("--eps-start", type=float, default=1.0)
    p.add_argument("--eps-end", type=float, default=0.05)
    p.add_argument("--eps-decay-fraction", type=float, default=0.5)
    p.add_argument("--video", action="store_true")
    return p.parse_args()


def train(args):
    set_seed(args.seed)
    run_dir = make_run_dir("dqn", args.env_id, args.seed)
    env = make_env(args.env_id, args.seed, video_dir=f"{run_dir}/videos" if args.video else None)
    describe_env(env)
    logger = Logger(run_dir)

    agent = Agent(env.observation_space, env.action_space)
    # TODO: create target network (a copy of agent with the same weights) and an Adam optimizer
    # TODO: create ReplayBuffer(args.buffer_size, env.observation_space.shape, (), action_dtype=np.int64)

    obs, info = env.reset(seed=args.seed)
    for global_step in range(args.total_steps):
        # 1. Choose action with epsilon-greedy
        epsilon = linear_schedule(args.eps_start, args.eps_end,
                                  args.eps_decay_fraction * args.total_steps, global_step)
        # TODO: with prob. epsilon take env.action_space.sample(), else agent.act(obs)
        action = env.action_space.sample()

        # 2. Step the environment
        next_obs, reward, terminated, truncated, info = env.step(action)
        logger.log_episode(info, global_step)

        # 3. Store transition
        # TODO: buffer.add(...)

        obs = next_obs
        if terminated or truncated:
            obs, info = env.reset()

        # 4. Learn
        if global_step > args.learning_starts and global_step % args.train_every == 0:
            # TODO: sample a batch, compute TD target with the TARGET network (under torch.no_grad()),
            #       compute Q(s, a) for the taken actions (hint: .gather), MSE loss, optimizer step
            # logger.log_scalar("losses/q_loss", loss.item(), global_step)
            # logger.log_scalar("charts/epsilon", epsilon, global_step)
            pass

        # 5. Update target network
        if global_step % args.target_update_every == 0:
            # TODO: target_net.load_state_dict(agent.state_dict())
            pass

    save_agent(agent, run_dir)
    env.close()
    logger.close()


if __name__ == "__main__":
    train(parse_args())
