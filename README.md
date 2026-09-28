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
  logger.py      TensorBoard + progress.csv + console logging
  networks.py    mlp(), QNetwork + Rainbow parts (TODO: NoisyLinear, DuelingHead)
  buffers.py     replay / n-step / prioritized / rollout buffers (TODO: implement)
  utils.py       seeding, run dirs, save/load
  evaluation.py  greedy evaluation (+ predicted Q vs. actual return)
rl/algos/      one file per algorithm, all expose the same `Agent` interface
scripts/
  play.py            visualize any env + agent
  run_experiment.py  run variants x seeds in parallel
  plot.py            learning curves, summary bars, Q-estimate plots
tests/         pytest checks for every component you implement
```

## DQN → Rainbow

`rl/algos/dqn.py` has a flag for each Rainbow improvement. Implement them in this order and check each
with its tests before training:

| # | Flag | Implement | Test |
|---|---|---|---|
| 0 | – | `ReplayBuffer`, `Agent.act`, `select_action`, `store_transition`, `compute_td_target`, `dqn_loss`, `update` | `pytest tests/test_buffers.py -k replay`, `tests/test_dqn.py -k "vanilla or terminal"` |
| 1 | `--double` | Double DQN in `compute_td_target` | `-k double` |
| 2 | `--dueling` | `DuelingHead` | `-k dueling` |
| 3 | `--n-step 3` | `NStepBuffer`, extend `store_transition` | `-k nstep` |
| 4 | `--per` | `SumTree`, `PrioritizedReplayBuffer`, extend `update` | `-k "sum_tree or per"` |
| 5 | `--noisy` | `NoisyLinear`, noise reset in `update` | `-k noisy` |
| 6 | `--c51` | `project_distribution`, `c51_loss` | `-k projection` |

```bash
pytest -v                                        # all component tests
python -m rl.algos.dqn --env-id CartPole-v1      # one quick run

# Compare variants: 5 seeds each, 8 in parallel. Finished runs are skipped, so add variants over time.
python scripts/run_experiment.py --experiment lunar-add-one --preset add-one --only dqn dqn+double \
    --env-id LunarLander-v3 --total-steps 300000 --seeds 5
python scripts/plot.py runs/lunar-add-one
```

Presets: `add-one` (DQN + each single improvement), `cumulative` (adds them one after another),
`ablation` (Rainbow minus each, as in the paper), `custom --variants "double,dueling" "mine=--double --lr 1e-3"`.
C51 needs `--v-min/--v-max` to cover the returns (CartPole: `-- --v-min 0 --v-max 100`).

## Roadmap

| # | Algorithm | Envs | Status | Result |
|---|---|---|---|---|
| 0 | Random baseline | CartPole-v1 | done | ~22 return |
| 1 | DQN → Rainbow (6 improvements) | CartPole-v1, LunarLander-v3 | TODO | |
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
