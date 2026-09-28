# RL from Scratch

Implementing reinforcement learning algorithms myself in PyTorch, trained on
[Gymnasium](https://gymnasium.farama.org/) and [MuJoCo](https://mujoco.org/) robot simulations.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

## Usage

```bash
source .venv/bin/activate

# Train (each algorithm is a runnable module)
python -m rl.algos.random_agent --env-id CartPole-v1
python -m rl.algos.dqn --env-id CartPole-v1 --video

# Watch training curves
tensorboard --logdir runs          # open http://localhost:6006

# Watch an env / a trained agent live
python scripts/play.py --env-id HalfCheetah-v5
python scripts/play.py --env-id CartPole-v1 --algo dqn --model runs/<run>/model.pt
```

Every run creates `runs/<algo>_<env>_s<seed>_<time>/` with TensorBoard logs, `model.pt` and (with `--video`) mp4s.

## Structure

```
rl/common/     shared infrastructure
  env_utils.py   make_env (stats + video wrappers), describe_env
  logger.py      TensorBoard + console logging
  networks.py    mlp() helper
  buffers.py     ReplayBuffer, RolloutBuffer (TODO: implement)
  utils.py       seeding, run dirs, save/load
rl/algos/      one file per algorithm, all expose the same `Agent` interface
scripts/play.py  visualize any env + agent
```

## Roadmap

| # | Algorithm | Envs | Status | Result |
|---|---|---|---|---|
| 0 | Random baseline | CartPole-v1 | done | ~22 return |
| 1 | DQN | CartPole-v1, LunarLander-v3 | TODO | |
| 2 | REINFORCE → A2C | CartPole-v1 | TODO | |
| 3 | PPO | Pendulum-v1, Hopper-v5, HalfCheetah-v5 | TODO | |
| 4 | SAC (/ TD3) | Pendulum-v1, HalfCheetah-v5, Reacher-v5 | TODO | |
| 5 | SAC + HER | FetchReach-v4, FetchPush-v4 | TODO | |
| 6 | Imitation learning with LeRobot | gym-pusht, gym-aloha | TODO | |

## Debugging tips

- Compare against the random baseline and against published curves (e.g. CleanRL's benchmarks).
- Always run 3+ seeds before concluding something works or doesn't.
- Start small: an algorithm that can't solve CartPole/Pendulum won't solve HalfCheetah.
- Log everything: losses, Q-values, entropy, gradient norms. Exploding Q-values = bug in the target.
- Reference implementations: [CleanRL](https://github.com/vwxyzjn/cleanrl), [Spinning Up](https://spinningup.openai.com/).
