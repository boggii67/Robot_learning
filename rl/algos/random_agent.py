"""A random agent. Learns nothing, but shows the full pipeline every algorithm uses.

Run:  python -m rl.algos.random_agent --env-id CartPole-v1
Its returns are your BASELINE: if your trained agent isn't clearly better, it didn't learn.
"""

import argparse

import torch.nn as nn

from rl.common.env_utils import describe_env, make_env
from rl.common.evaluation import evaluate
from rl.common.logger import Logger
from rl.common.utils import make_run_dir, save_agent, save_config, set_seed


class Agent(nn.Module):
    """Every algorithm defines an Agent with this interface, so scripts/play.py can load any of them."""

    def __init__(self, observation_space, action_space, config=None):
        super().__init__()
        self.action_space = action_space

    def act(self, obs, deterministic=False):
        return self.action_space.sample()


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--env-id", default="CartPole-v1")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--total-steps", type=int, default=20_000)
    p.add_argument("--experiment", default=None, help="group runs under runs/<experiment>/ for plotting")
    p.add_argument("--name", default="random")
    p.add_argument("--eval-every", type=int, default=5_000)
    p.add_argument("--eval-episodes", type=int, default=5)
    p.add_argument("--video", action="store_true", help="record videos to the run dir")
    return p.parse_args(argv)


def train(args):
    set_seed(args.seed)
    run_dir = make_run_dir(args.name, args.env_id, args.seed, args.experiment)
    save_config(args, run_dir)
    env = make_env(args.env_id, args.seed, video_dir=f"{run_dir}/videos" if args.video else None)
    env.action_space.seed(args.seed)
    describe_env(env)
    logger = Logger(run_dir)
    agent = Agent(env.observation_space, env.action_space)

    obs, info = env.reset(seed=args.seed)
    for global_step in range(args.total_steps):
        if global_step % args.eval_every == 0:
            res = evaluate(agent, args.env_id, args.eval_episodes, seed=10_000 + args.seed)
            logger.log_scalar("eval/return", res["return"], global_step)

        action = agent.act(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)
        logger.log_episode(info, global_step)

        # <- a learning algorithm would store (obs, action, reward, next_obs, terminated) and update here

        obs = next_obs
        if terminated or truncated:
            obs, info = env.reset()

    res = evaluate(agent, args.env_id, args.eval_episodes, seed=10_000 + args.seed)
    logger.log_scalar("eval/return", res["return"], args.total_steps)
    save_agent(agent, run_dir)
    env.close()
    logger.close()


if __name__ == "__main__":
    train(parse_args())
