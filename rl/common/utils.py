import json
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


def make_run_dir(algo, env_id, seed, experiment=None):
    """Where logs, videos and the model of one run go.

    Without experiment: runs/<algo>_<env>_s<seed>_<timestamp>/
    With experiment:    runs/<experiment>/<algo>/s<seed>/   (grouped for scripts/plot.py)
    """
    if experiment:
        path = os.path.join("runs", experiment, algo, f"s{seed}")
    else:
        path = os.path.join("runs", f"{algo}_{env_id}_s{seed}_{time.strftime('%Y%m%d-%H%M%S')}")
    os.makedirs(path, exist_ok=True)
    return path


def save_config(args, run_dir):
    """Save all hyperparameters - needed later to rebuild the network (play.py) and to know what you ran."""
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)


def load_config(run_dir):
    path = os.path.join(run_dir, "config.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def save_agent(agent, run_dir):
    path = os.path.join(run_dir, "model.pt")
    torch.save(agent.state_dict(), path)
    print(f"Saved model to {path}")
    return path


def load_agent(agent, path):
    agent.load_state_dict(torch.load(path, map_location="cpu"))
    return agent
