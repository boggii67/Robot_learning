"""Logging to TensorBoard and to progress.csv (for scripts/plot.py).

View live curves with:  tensorboard --logdir runs
"""

import csv
import os
from collections import deque

import numpy as np
from torch.utils.tensorboard import SummaryWriter


class Logger:
    def __init__(self, run_dir, print_every_episodes=10):
        self.writer = SummaryWriter(run_dir)
        # Long format (one row per value) so any metric can be added without changing the file layout.
        self.csv_file = open(os.path.join(run_dir, "progress.csv"), "w", newline="")
        self.csv = csv.writer(self.csv_file)
        self.csv.writerow(["step", "name", "value"])
        self.recent_returns = deque(maxlen=print_every_episodes)
        self.print_every = print_every_episodes
        self.episodes = 0

    def log_scalar(self, name, value, step):
        """E.g. log_scalar("losses/q_loss", loss.item(), global_step)"""
        value = float(value)
        self.writer.add_scalar(name, value, step)
        self.csv.writerow([step, name, value])

    def log_episode(self, info, step):
        """Call after every env.step(); logs the return when an episode has finished.

        Works with RecordEpisodeStatistics, which puts info["episode"] at episode end.
        """
        if "episode" not in info:
            return
        ep_return = float(info["episode"]["r"])
        self.log_scalar("charts/episodic_return", ep_return, step)
        self.log_scalar("charts/episodic_length", int(info["episode"]["l"]), step)
        self.recent_returns.append(ep_return)
        self.episodes += 1
        if self.episodes % self.print_every == 0:
            self.csv_file.flush()
            print(f"step {step:>8} | episode {self.episodes:>5} | "
                  f"mean return (last {len(self.recent_returns)}): {np.mean(self.recent_returns):8.2f}",
                  flush=True)

    def close(self):
        self.writer.close()
        self.csv_file.close()
