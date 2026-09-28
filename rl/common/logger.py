"""Logging to TensorBoard. View with:  tensorboard --logdir runs"""

from collections import deque

import numpy as np
from torch.utils.tensorboard import SummaryWriter


class Logger:
    def __init__(self, run_dir, print_every_episodes=10):
        self.writer = SummaryWriter(run_dir)
        self.recent_returns = deque(maxlen=print_every_episodes)
        self.print_every = print_every_episodes
        self.episodes = 0

    def log_scalar(self, name, value, step):
        """E.g. log_scalar("losses/q_loss", loss.item(), global_step)"""
        self.writer.add_scalar(name, value, step)

    def log_episode(self, info, step):
        """Call after every env.step(); logs the return when an episode has finished.

        Works with RecordEpisodeStatistics, which puts info["episode"] at episode end.
        """
        if "episode" not in info:
            return
        ep_return = float(info["episode"]["r"])
        ep_length = int(info["episode"]["l"])
        self.writer.add_scalar("charts/episodic_return", ep_return, step)
        self.writer.add_scalar("charts/episodic_length", ep_length, step)
        self.recent_returns.append(ep_return)
        self.episodes += 1
        if self.episodes % self.print_every == 0:
            print(f"step {step:>8} | episode {self.episodes:>5} | "
                  f"mean return (last {len(self.recent_returns)}): {np.mean(self.recent_returns):8.2f}")

    def close(self):
        self.writer.close()
