"""DQN and the Rainbow improvements, each switchable by a flag.

    python -m rl.algos.dqn --env-id CartPole-v1                        # vanilla DQN
    python -m rl.algos.dqn --double --dueling                          # DQN + 2 improvements
    python -m rl.algos.dqn --double --dueling --per --n-step 3 --noisy --c51   # Rainbow

Vanilla DQN (Mnih et al. 2015, https://www.nature.com/articles/nature14236):
learn Q(s, a) = expected return after taking action a in state s.
    target = r + gamma * max_a' Q_target(s', a') * (1 - terminated)
    loss   = (Q(s, a) - target)^2      (or Huber loss)
Two tricks make it stable:
    1. Replay buffer  -> train on random old transitions, not just the latest one
    2. Target network -> a slowly updated copy of Q used to compute the target
Exploration: epsilon-greedy (random action with prob. epsilon, decayed over time).

Rainbow (Hessel et al. 2017, https://arxiv.org/abs/1710.02298) combines 6 improvements:
    --double    Double DQN           less overestimation of Q-values           (this file)
    --dueling   Dueling network      Q = V + A - mean(A)                       (networks.py)
    --n-step N  n-step returns       faster reward propagation                 (buffers.py)
    --per       Prioritized replay   learn more from surprising transitions    (buffers.py)
    --noisy     Noisy nets           learned exploration instead of epsilon    (networks.py)
    --c51       Distributional RL    learn the distribution of returns         (this file)

Suggested order: vanilla -> double -> dueling -> n-step -> per -> noisy -> c51.
Test each component with pytest before training (tests/), then compare with scripts/run_experiment.py.

What's already written: the training loop, logging, evaluation, network wiring (QNetwork).
What you write: every function with a TODO below.
"""

import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from rl.common.buffers import NStepBuffer, PrioritizedReplayBuffer, ReplayBuffer
from rl.common.env_utils import describe_env, make_env
from rl.common.evaluation import evaluate
from rl.common.logger import Logger
from rl.common.networks import QNetwork
from rl.common.utils import make_run_dir, save_agent, save_config, set_seed

NET_DEFAULTS = {"hidden": [128, 128], "dueling": False, "noisy": False, "c51": False,
                "n_atoms": 51, "v_min": -100.0, "v_max": 100.0}


class Agent(nn.Module):
    """Wraps the QNetwork. `config` is the dict of args (or config.json) that decides the architecture."""

    def __init__(self, observation_space, action_space, config=None):
        super().__init__()
        cfg = {**NET_DEFAULTS, **(config or {})}
        self.q_net = QNetwork(
            obs_dim=int(np.prod(observation_space.shape)),
            n_actions=action_space.n,
            hidden=tuple(cfg["hidden"]),
            dueling=cfg["dueling"],
            noisy=cfg["noisy"],
            n_atoms=cfg["n_atoms"] if cfg["c51"] else 1,
            v_min=cfg["v_min"],
            v_max=cfg["v_max"],
        )

    def q_values(self, obs):
        """Expected Q-values [B, A] for a batch of observations (tensor)."""
        return self.q_net.q_values(obs)

    def act(self, obs, deterministic=True):
        """Greedy action for a single observation (numpy array) -> Python int."""
        # TODO (step 0): convert obs to a float tensor with a batch dim, return argmax of q_values
        raise NotImplementedError("Agent.act")


# -------------------------------------------------------------------------------------------------
# Step 0: vanilla DQN
# -------------------------------------------------------------------------------------------------


def select_action(agent, obs, epsilon, action_space):
    """Epsilon-greedy: random action with prob. epsilon, else agent.act(obs).
    (With --noisy the loop passes epsilon=0: the noise in the weights does the exploring.)"""
    # TODO (step 0)
    raise NotImplementedError("select_action")


def store_transition(buffer, nstep, obs, action, reward, next_obs, terminated, truncated):
    """Put a transition into the replay buffer.
    Without n-step (nstep is None): buffer.add(obs, action, reward, next_obs, done).
        Which `done` is right: terminated, or terminated or truncated? (Hint: what does
        (1 - done) in the target mean for a CartPole episode cut off at step 500?)
    With n-step (step 3): pass the transition through nstep.append(...) and add what comes out."""
    # TODO (step 0, extend in step 3)
    raise NotImplementedError("store_transition")


def compute_td_target(q_fn, target_q_fn, rewards, next_obs, dones, gamma, double=False):
    """TD target y = r + gamma * Q_target(s', a*) * (1 - done), shape [B]. Use torch.no_grad()!

    q_fn / target_q_fn: callables obs -> Q-values [B, A] (online and target network)
    gamma: already gamma ** n_step for n-step returns
    Vanilla (step 0):  a* = argmax_a Q_target(s', a)   -> the same net selects AND evaluates the action
                                                        -> max over noisy estimates is biased upwards
    Double (step 1):   a* = argmax_a Q_online(s', a)   -> online net selects, target net evaluates
                       (van Hasselt et al. 2015, https://arxiv.org/abs/1509.06461)
    """
    # TODO (step 0, extend in step 1)
    raise NotImplementedError("compute_td_target")


def dqn_loss(agent, target_agent, batch, args):
    """Loss for the non-distributional variants.

    Returns (loss_per_sample [B], td_abs [B] as numpy - the new PER priorities, q_mean float)
    Steps: y = compute_td_target(...); q = Q(s, a) of the taken actions (hint: .gather);
           loss_per_sample = F.smooth_l1_loss(q, y, reduction="none")   (Huber: robust to big errors)
    """
    # TODO (step 0)
    raise NotImplementedError("dqn_loss")


def update(agent, target_agent, optimizer, buffer, args, beta):
    """One gradient step. Returns a dict of metrics to log, e.g. {"loss": ..., "q_mean": ...}.

    1. batch = buffer.sample(args.batch_size, beta)
    2. loss_per_sample, td_abs, q_mean = c51_loss(...) if args.c51 else dqn_loss(...)
    3. loss = (batch["weights"] * loss_per_sample).mean()      (weights are all 1 without PER)
    4. optimizer.zero_grad(); loss.backward(); clip_grad_norm_(agent.parameters(), 10); optimizer.step()
    5. step 4 (--per):   buffer.update_priorities(batch["indices"], td_abs)
    6. step 5 (--noisy): draw new noise for agent.q_net and target_agent.q_net (reset_noise)
    """
    # TODO (step 0, extend in steps 4 and 5)
    raise NotImplementedError("update")


# -------------------------------------------------------------------------------------------------
# Step 6: distributional RL (C51). Bellemare et al. 2017, https://arxiv.org/abs/1707.06887
# Instead of the expected return Q(s, a), learn a probability distribution over returns on fixed
# atoms z_i = v_min + i * delta_z (the network's `support`). Q(s, a) = sum_i p_i(s, a) * z_i.
# -------------------------------------------------------------------------------------------------


def project_distribution(next_probs, rewards, dones, support, gamma):
    """Project the target distribution back onto the support.

    next_probs: [B, N] probabilities of the next state's chosen action (from the target net)
    rewards, dones: [B];  support: [N];  gamma: already gamma ** n_step
    Each atom z_j moves to Tz_j = clip(r + gamma * (1 - done) * z_j, v_min, v_max), which usually
    lies BETWEEN two atoms -> split its probability to the neighbours l = floor(b), u = ceil(b)
    with b = (Tz_j - v_min) / delta_z, in proportion to the distance:
        m_l += p_j * (u - b),   m_u += p_j * (b - l)
    Careful: when b is an exact integer, l == u and both terms are 0 -> the mass disappears!
    Returns m: [B, N], each row sums to 1. (Hint: scatter_add_ or index_add_ on a flattened view.)
    """
    # TODO (step 6)
    raise NotImplementedError("project_distribution")


def c51_loss(agent, target_agent, batch, args):
    """Cross-entropy between the projected target distribution and the predicted one.

    Returns (loss_per_sample [B], priorities [B] numpy, q_mean float), like dqn_loss.
    1. with no_grad: next action a* (argmax of expected Q; online net if --double, else target net),
       next_probs = target_agent.q_net.dist(next_obs)[range(B), a*], m = project_distribution(...)
    2. log_p = agent.q_net.dist(obs, log=True)[range(B), actions]
    3. loss_per_sample = -(m * log_p).sum(-1)       (also a good PER priority, instead of |td|)
    """
    # TODO (step 6)
    raise NotImplementedError("c51_loss")


# -------------------------------------------------------------------------------------------------
# Training loop (already written)
# -------------------------------------------------------------------------------------------------


def linear_schedule(start, end, duration, step):
    """Linear decay from `start` to `end` over `duration` steps (epsilon, PER beta)."""
    slope = (end - start) / duration
    return max(slope * step + start, end) if end < start else min(slope * step + start, end)


def variant_name(args):
    if args.name:
        return args.name
    parts = ["dqn"]
    if args.double:
        parts.append("double")
    if args.dueling:
        parts.append("dueling")
    if args.n_step > 1:
        parts.append(f"nstep{args.n_step}")
    if args.per:
        parts.append("per")
    if args.noisy:
        parts.append("noisy")
    if args.c51:
        parts.append("c51")
    return "+".join(parts)


def parse_args(argv=None):
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    g = p.add_argument_group("general")
    g.add_argument("--env-id", default="CartPole-v1")
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--total-steps", type=int, default=100_000)
    g.add_argument("--experiment", default=None, help="group runs under runs/<experiment>/ for plotting")
    g.add_argument("--name", default=None, help="variant name (default: built from the flags)")
    g.add_argument("--eval-every", type=int, default=5_000)
    g.add_argument("--eval-episodes", type=int, default=5)
    g.add_argument("--log-every", type=int, default=500, help="log training metrics every N steps")
    g.add_argument("--video", action="store_true")

    d = p.add_argument_group("DQN")
    d.add_argument("--hidden", type=int, nargs="+", default=[128, 128])
    d.add_argument("--lr", type=float, default=2.5e-4)
    d.add_argument("--gamma", type=float, default=0.99)
    d.add_argument("--buffer-size", type=int, default=50_000)
    d.add_argument("--batch-size", type=int, default=128)
    d.add_argument("--learning-starts", type=int, default=1_000, help="fill buffer before training")
    d.add_argument("--train-every", type=int, default=4, help="gradient step every N env steps")
    d.add_argument("--target-update-every", type=int, default=500)
    d.add_argument("--eps-start", type=float, default=1.0)
    d.add_argument("--eps-end", type=float, default=0.05)
    d.add_argument("--eps-decay-fraction", type=float, default=0.5)

    r = p.add_argument_group("Rainbow")
    r.add_argument("--double", action="store_true")
    r.add_argument("--dueling", action="store_true")
    r.add_argument("--n-step", type=int, default=1)
    r.add_argument("--per", action="store_true")
    r.add_argument("--per-alpha", type=float, default=0.6)
    r.add_argument("--per-beta-start", type=float, default=0.4)
    r.add_argument("--noisy", action="store_true")
    r.add_argument("--c51", action="store_true")
    r.add_argument("--n-atoms", type=int, default=51)
    r.add_argument("--v-min", type=float, default=-100.0, help="C51 support; should cover the returns")
    r.add_argument("--v-max", type=float, default=100.0)
    return p.parse_args(argv)


def train(args):
    set_seed(args.seed)
    variant = variant_name(args)
    run_dir = make_run_dir(variant, args.env_id, args.seed, args.experiment)
    save_config(args, run_dir)
    print(f"Variant: {variant}  ->  {run_dir}")

    env = make_env(args.env_id, args.seed, video_dir=f"{run_dir}/videos" if args.video else None)
    describe_env(env)
    logger = Logger(run_dir)

    agent = Agent(env.observation_space, env.action_space, vars(args))
    target_agent = Agent(env.observation_space, env.action_space, vars(args))
    target_agent.load_state_dict(agent.state_dict())
    optimizer = torch.optim.Adam(agent.parameters(), lr=args.lr)

    buffer_cls = PrioritizedReplayBuffer if args.per else ReplayBuffer
    buffer_kwargs = {"alpha": args.per_alpha} if args.per else {}
    buffer = buffer_cls(args.buffer_size, env.observation_space.shape, (), action_dtype=np.int64, **buffer_kwargs)
    nstep = NStepBuffer(args.n_step, args.gamma) if args.n_step > 1 else None

    obs, info = env.reset(seed=args.seed)
    for global_step in range(args.total_steps):
        if global_step % args.eval_every == 0:
            log_evaluation(agent, args, logger, global_step)

        epsilon = 0.0 if args.noisy else linear_schedule(
            args.eps_start, args.eps_end, args.eps_decay_fraction * args.total_steps, global_step)
        action = select_action(agent, np.asarray(obs, dtype=np.float32), epsilon, env.action_space)

        next_obs, reward, terminated, truncated, info = env.step(action)
        logger.log_episode(info, global_step)
        store_transition(buffer, nstep, obs, action, reward, next_obs, terminated, truncated)

        obs = next_obs
        if terminated or truncated:
            obs, info = env.reset()

        if global_step >= args.learning_starts and global_step % args.train_every == 0:
            beta = linear_schedule(args.per_beta_start, 1.0, args.total_steps, global_step)
            metrics = update(agent, target_agent, optimizer, buffer, args, beta)
            if global_step % args.log_every < args.train_every:
                for key, value in (metrics or {}).items():
                    logger.log_scalar(f"train/{key}", value, global_step)
                logger.log_scalar("charts/epsilon", epsilon, global_step)
                if args.per:
                    logger.log_scalar("charts/per_beta", beta, global_step)
                if args.noisy:
                    logger.log_scalar("charts/noisy_sigma", agent.q_net.mean_sigma(), global_step)

        # Hard target update: copy the online weights every N steps
        if global_step % args.target_update_every == 0:
            target_agent.load_state_dict(agent.state_dict())

    log_evaluation(agent, args, logger, args.total_steps)
    save_agent(agent, run_dir)
    env.close()
    logger.close()


def log_evaluation(agent, args, logger, step):
    res = evaluate(agent, args.env_id, args.eval_episodes, seed=10_000 + args.seed, gamma=args.gamma)
    for key, value in res.items():
        logger.log_scalar(f"eval/{key}", value, step)
    print(f"[eval] step {step:>8} | return {res['return']:8.2f} ± {res['return_std']:.2f}", flush=True)


if __name__ == "__main__":
    train(parse_args())
