"""Greedy evaluation: measure how good the policy is WITHOUT exploration.

Training returns are misleading for comparisons: an epsilon-greedy agent takes random actions,
a noisy-net agent has noisy weights. Evaluation removes both (agent.eval() + deterministic=True),
so all DQN variants are compared fairly.
"""

import numpy as np
import torch

from rl.common.env_utils import make_env


def evaluate(agent, env_id, n_episodes=5, seed=10_000, gamma=0.99):
    """Run `n_episodes` greedy episodes on a fresh env.

    Returns a dict:
        return            mean undiscounted episode return (the score)
        return_std        std over episodes
        and, if the agent has q_values(obs) (all DQN variants):
        q_mean            mean over visited states of the predicted max_a Q(s, a)
        discounted_return mean over the same states of the ACTUAL discounted return-to-go
    q_mean > discounted_return means the agent overestimates its values (what Double DQN fixes).
    Note: at a time-limit truncation the actual return-to-go is cut off, so it is slightly too low.
    """
    env = make_env(env_id, seed)
    has_q = hasattr(agent, "q_values")
    was_training = agent.training
    agent.eval()

    returns, q_preds, returns_to_go = [], [], []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        rewards, done = [], False
        while not done:
            obs_arr = np.asarray(obs, dtype=np.float32)
            with torch.no_grad():
                if has_q:
                    q_preds.append(agent.q_values(torch.as_tensor(obs_arr).unsqueeze(0)).max().item())
                action = agent.act(obs_arr, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            rewards.append(float(reward))
            done = terminated or truncated
        returns.append(sum(rewards))
        g, ep_rtg = 0.0, []
        for r in reversed(rewards):
            g = r + gamma * g
            ep_rtg.append(g)
        returns_to_go.extend(reversed(ep_rtg))

    env.close()
    agent.train(was_training)
    result = {"return": float(np.mean(returns)), "return_std": float(np.std(returns))}
    if has_q:
        result["q_mean"] = float(np.mean(q_preds))
        result["discounted_return"] = float(np.mean(returns_to_go))
    return result
