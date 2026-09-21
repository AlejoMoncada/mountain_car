# Random exploration diagnosis for DQN

A reproducible measurement of this checkout's `MountainCar-v0` environment found **0 flag reaches in 500 uniformly random episodes**. All 500 episodes ended at the time limit. This is evidence for the exploration diagnosis; it does not change `dqn.py` or any agent behaviour.

## Measurement method

Run from `actividad_2`:

```bash
uv run python scripts/measure_random_exploration.py
```

The script created one Gymnasium `MountainCar-v0` environment and played 500 episodes. For episode index `i`, it called `env.reset(seed=20250310 + i)`, seeded the action space with that same seed, and selected every action with `env.action_space.sample()`. Seeding both sources makes the initial state and the uniform action sequence reproducible without selecting actions through an agent.

For every episode, [`episodes.csv`](../artifacts/dqn/random-exploration/seed-20250310/episodes.csv) records the episode index, seed, return, steps, terminal flags, final position, maximum position, action counts for actions 0, 1, and 2, and the episode's `longest_identical_action_run`, `longest_run_action`, and `longest_run_start_step`. These run fields make the summary's longest-run value and context recomputable from the CSV alone. [`summary.json`](../artifacts/dqn/random-exploration/seed-20250310/summary.json) records the aggregate and runtime metadata.

## Observed result

These are our own measurements of **this environment and version**: `MountainCar-v0` under Gymnasium `1.3.0` on Python `3.11.13`. They are not reference figures copied from an exercise or a theoretical simulation.

| Measure | Observed value |
| --- | ---: |
| Episodes | 500 |
| Terminated (flag reached) | 0 |
| Truncated (time limit) | 500 |
| Success rate | 0.0 (0.0000%) |
| Mean return | -200.0 |
| Mean final position | -0.5169559219777584 |
| Maximum final position | -0.2282896488904953 |
| Mean maximum position | -0.3888909391760826 |
| Maximum maximum position | -0.1683390736579895 |
| Longest identical consecutive-action run | 11 actions: action 0, episode index 146, starting at step 44 |

A zero success count is the expected finding **if it occurs** under this measurement; it must not be manufactured. In this run it did occur: the recorded count is 0, and the table reports the actual values from the generated artifacts.

## What the result implies

Uniform, independently sampled per-step actions did not reach the flag in this 500-episode sample. Every trajectory used the full 200-step time limit and none terminated by reaching the goal. The longest same-action run observed anywhere in the sample was only 11 actions.

MountainCar needs a sustained push in one direction, and then sustained pushing in the other direction, to build momentum by rocking between the slopes. A fresh, independent action at every step often interrupts those pushes. The measurement therefore supports investigating exploration that can produce sustained action sequences; it does not by itself prove a required run length or select a particular DQN fix.

## Reasoning note, not measurement

For independent uniform draws over three actions, the probability that a **particular** fixed window of `k` actions is a prescribed direction (all left or all right) is `3^-k`. If either pushing direction is acceptable, the probability for that fixed window is `2 * 3^-k`. For illustration, with `k = 20` it is:

```text
2 / 3^20 = 5.735943981584852e-10
```

This calculation is reasoning, not a measurement of the environment. It does not account for overlapping windows, the momentum threshold, or state-dependent trajectory dynamics. The observed evidence is the CSV and JSON artifacts above, including the actual longest run of 11.
