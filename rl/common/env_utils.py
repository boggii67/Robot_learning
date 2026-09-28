"""Creating environments: the "world" the agent lives in.

Every Gymnasium env follows the same loop:

    obs, info = env.reset(seed=...)
    while True:
        action = agent.act(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:   # episode over
            obs, info = env.reset()

- terminated: the task ended "for real" (robot fell, goal reached) -> no future reward.
- truncated:  we hit a time limit -> the episode was cut off, the state still has a future value.
  This distinction matters when you compute bootstrapped targets (DQN, SAC, PPO)!
"""

import gymnasium as gym

try:
    # Registers the robot-arm envs (FetchReach, FetchPush, ...). Optional.
    import gymnasium_robotics

    gym.register_envs(gymnasium_robotics)
except ImportError:
    pass


def make_env(env_id, seed=0, render_mode=None, video_dir=None, video_every=50):
    """Create a single env with episode statistics and optional video recording.

    Args:
        env_id: e.g. "CartPole-v1", "Pendulum-v1", "HalfCheetah-v5", "FetchReach-v4"
        render_mode: None (fast, no graphics), "human" (live window), or "rgb_array"
        video_dir: if set, every `video_every`-th episode is saved as an .mp4 here
    """
    if video_dir is not None:
        render_mode = "rgb_array"  # video recording needs pixel frames
    env = gym.make(env_id, render_mode=render_mode)

    # Adds info["episode"] = {"r": total_return, "l": length} at the end of each episode.
    env = gym.wrappers.RecordEpisodeStatistics(env)

    if video_dir is not None:
        env = gym.wrappers.RecordVideo(
            env, video_dir, episode_trigger=lambda ep: ep % video_every == 0, disable_logger=True
        )

    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    return env


def describe_env(env):
    """Print what the agent sees and what it can do. Always look at this for a new env!"""
    print(f"Env:               {env.spec.id}")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space:      {env.action_space}")
    if isinstance(env.action_space, gym.spaces.Discrete):
        print(f"  -> discrete, {env.action_space.n} actions (DQN / REINFORCE / PPO work)")
    elif isinstance(env.action_space, gym.spaces.Box):
        print(f"  -> continuous, range [{env.action_space.low}, {env.action_space.high}]"
              " (PPO / SAC / TD3 work, DQN does not)")
    print(f"Max episode steps: {env.spec.max_episode_steps}")
