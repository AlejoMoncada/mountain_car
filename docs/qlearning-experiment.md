# Reproducible Q-learning experiment

Use the runner for one **fresh** tabular Q-learning run. It never loads a prior
checkpoint, so its outputs describe the requested run rather than mixed history.

## Run

```bash
uv run python scripts/run_qlearning_experiment.py \
  --episodes 20000 --seed 20250308 \
  --eval-seed 30250308 --eval-episodes 100 \
  --output-dir artifacts/qlearning/seed-20250308 \
  --save-path saves/qlearning_seed-20250308.pkl
```

The defaults are the same as this command. The runner refuses a nonempty output
directory, an existing checkpoint, and overlapping training/evaluation seed
ranges; choose a new path to run again.

## What the protocol fixes

Training starts with a new agent, seeds NumPy once, and resets training episode
`i` with `seed + i - 1`. Evaluation uses a separate consecutive seed range,
selects greedy actions, and does not update the Q-table. The CSV files record
one row per episode, including seed, return, steps, terminal/truncation status,
epsilon used during that episode (before its end-of-episode decay), and elapsed
time. `summary.json` derives its statistics and success
counts from those rows; the PNG plots raw training returns and a 100-episode
rolling mean.

| Artifact | Use |
| --- | --- |
| `training_episodes.csv` | Learning trajectory under exploratory actions |
| `evaluation_episodes.csv` | Fixed-seed greedy policy assessment |
| `summary.json` | Configuration, versions, durations, statistics, and separate best returns |
| `training_curve.png` | Training return and rolling-100 visualization |

## Observed baseline — seed 20250308

This completed fresh run used 20,000 training episodes with training seed
`20250308`, followed by 100 greedy evaluation episodes from the disjoint base
seed `30250308`. Training took **35.5071 seconds**.

| Topic | Observed value |
| --- | --- |
| Greedy evaluation mean return | **-166.88** |
| Greedy evaluation standard deviation | **23.10812844** |
| Greedy evaluation successes | **84/100 (84%)** |
| Best greedy evaluation return | **-128** |
| Best exploratory training return | **-89** |

The agent used a 20-by-20 discretization: **400 possible states** and **3
actions**. Its fixed hyperparameters were `n_bins=20`, `lr=0.1`, `gamma=0.99`,
`epsilon_start=1.0`, `epsilon_end=0.01`, and `epsilon_decay=0.9995`.

### Evidence

- [Run summary](../artifacts/qlearning/seed-20250308/summary.json)
- [Training episodes CSV](../artifacts/qlearning/seed-20250308/training_episodes.csv)
- [Evaluation episodes CSV](../artifacts/qlearning/seed-20250308/evaluation_episodes.csv)
- [Training curve PNG](../artifacts/qlearning/seed-20250308/training_curve.png)

Independent recomputation from the CSV files matched the recorded summary. A
checkpoint loaded after the run also reproduced all 100 recorded greedy
evaluation episodes.

### What the curve shows

The curve begins near the `-200` time limit, improves after the early episodes,
and then remains highly variable. Its rolling 100-episode mean reaches roughly
the low `-120`s around the middle of training but later falls back; the final
portion is not a stable convergence trend. Individual exploratory episodes can
look better than the evaluated greedy policy, which is why the `-89` best
training return must not be compared as though it were an evaluation result.

## Interpret results carefully

MountainCar gives `-1` at each step. Therefore a return is negative, and a
**less negative** return is better; a `terminated` episode means the car reached
the flag, while `truncated` normally means the time limit ended it. The best
training return is an observed exploratory trajectory, not the same quantity as
the best greedy evaluation return.

This is one-seed baseline evidence only: it is not a robust performance
estimate, does not establish that the task is solved, and does not compare
Q-learning with DQN yet. The reported policy is the final saved policy, not a
checkpoint selected for its best observed result. The seed protocol supports
repeatable fresh runs only: saved checkpoints intentionally do not store global
NumPy or environment RNG state, so resumed training has no exact-replay
guarantee. Record observed output from a completed run rather than predicting
outcomes in advance.
