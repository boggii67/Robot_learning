"""Watch an environment in a live window, with random actions or a trained agent.

Examples:
    python scripts/play.py --env-id HalfCheetah-v5                         # random actions
    python scripts/play.py --env-id CartPole-v1 --algo dqn --model runs/<run>/model.pt
    python scripts/play.py --env-id LunarLander-v3 --algo dqn --model runs/<experiment>/<variant>/s0/model.pt
    python scripts/play.py --env-id Hopper-v5 --video-dir videos/hopper    # save mp4 instead of window
"""

import argparse
import importlib
import inspect
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rl.common.env_utils import describe_env, make_env  # noqa: E402
from rl.common.utils import load_agent, load_config  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env-id", default="CartPole-v1")
    p.add_argument("--algo", default="random_agent", help="module name in rl/algos/")
    p.add_argument("--model", default=None, help="path to model.pt")
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--video-dir", default=None, help="record mp4s instead of opening a window")
    args = p.parse_args()

    env = make_env(args.env_id, render_mode="human", video_dir=args.video_dir, video_every=1)
    describe_env(env)

    Agent = importlib.import_module(f"rl.algos.{args.algo}").Agent
    # config.json next to model.pt says how the network was built (e.g. --dueling --c51)
    config = load_config(os.path.dirname(args.model)) if args.model else None
    if config is not None and "config" in inspect.signature(Agent).parameters:
        agent = Agent(env.observation_space, env.action_space, config=config)
    else:
        agent = Agent(env.observation_space, env.action_space)
    if args.model:
        load_agent(agent, args.model)
    agent.eval()

    for ep in range(args.episodes):
        obs, info = env.reset()
        done, ep_return = False, 0.0
        while not done:
            with torch.no_grad():
                action = agent.act(np.asarray(obs, dtype=np.float32), deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_return += reward
            done = terminated or truncated
        print(f"Episode {ep}: return = {ep_return:.2f}")
    env.close()


if __name__ == "__main__":
    main()
