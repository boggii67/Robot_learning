import os
import random
import time

import numpy as np
import torch


def set_seed(seed):
    """Make runs reproducible. RL is very noisy -> always compare several seeds!"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_run_dir(algo, env_id, seed):
    """runs/<algo>_<env>_<seed>_<timestamp>/ holds logs, videos and the model of one run."""
    name = f"{algo}_{env_id}_s{seed}_{time.strftime('%Y%m%d-%H%M%S')}"
    path = os.path.join("runs", name)
    os.makedirs(path, exist_ok=True)
    return path


def save_agent(agent, run_dir):
    path = os.path.join(run_dir, "model.pt")
    torch.save(agent.state_dict(), path)
    print(f"Saved model to {path}")
    return path


def load_agent(agent, path):
    agent.load_state_dict(torch.load(path, map_location="cpu"))
    return agent
