![CI](https://github.com/emiliomunozai/mountain_car/actions/workflows/ci.yml/badge.svg?branch=main)

## Entrega — Actividad 2 (Alejandro Moncada)

> **Documento de entrega:** [ENTREGA-ACTIVIDAD-2.md](ENTREGA-ACTIVIDAD-2.md) · **Comparación de métodos:** [docs/comparacion-qlearning-dqn.md](docs/comparacion-qlearning-dqn.md)

Este repositorio, derivado del repositorio base del curso, contiene la solución de la Actividad 2: un agente **Q-Learning tabular** con discretización del estado y un agente **DQN**, ambos entrenados y comparados sobre `MountainCar-v0`.

### Resultados medidos

| Métrica (evaluación voraz, 100 episodios) | Q-Learning | DQN |
|---|---|---|
| Retorno promedio | −166,88 | **−112,34** |
| Episodios con bandera alcanzada | 84 / 100 | **89 / 100** |
| Mejor retorno | −128 | **−84** |
| Episodios de entrenamiento | 20.000 | 2.500 |
| Duración del entrenamiento | **35,5 s** | 3 m 40 s |

Línea base medida: con acciones completamente aleatorias la bandera se alcanzó **0 veces en 500 episodios**, lo que motivó la corrección de exploración descrita en [docs/dqn-exploration-diagnosis.md](docs/dqn-exploration-diagnosis.md).

### Cómo ejecutar

```bash
uv sync                                                          # Python 3.11
uv run python -m unittest discover -s tests -p 'test_*.py' -v     # 23 pruebas
uv run python scripts/run_qlearning_experiment.py                 # Q-Learning  (~35 s)
uv run python scripts/measure_random_exploration.py               # línea base aleatoria
uv run python scripts/run_dqn_experiment.py                       # DQN         (~4 min)
```

Cada experimento escribe `training_episodes.csv`, `evaluation_episodes.csv`, `summary.json` y `training_curve.png` en su carpeta de `artifacts/`, con semillas fijas y disjuntas entre entrenamiento y evaluación.

### Documentación y evidencia

| Contenido | Ubicación |
|---|---|
| Informe de entrega | `ENTREGA-ACTIVIDAD-2.md` |
| Comparación Q-Learning frente a DQN | `docs/comparacion-qlearning-dqn.md` |
| Guía conceptual de Q-Learning | `docs/qlearning-foundations.md` |
| Guía conceptual de DQN | `docs/dqn-foundations.md` |
| Diagnóstico de exploración | `docs/dqn-exploration-diagnosis.md` |
| Protocolo y resultados tabulares | `docs/qlearning-experiment.md` |
| Evidencia de resultados | `artifacts/qlearning/seed-20250308/`, `artifacts/dqn/seed-20250310/` |
| Esquemas del estudiante | `docs/esquemas/` |

---

A hands-on repo for understanding how Reinforcement Learning works.
Train, inspect, and visualise RL agents on [MountainCar-v0](https://gymnasium.farama.org/environments/classic_control/mountain_car/) (or any other Gymnasium environment).

**This repo is a set of exercises.** The CLI, training loops and persistence are
written; the algorithms themselves are left as marked `EXERCISE` stubs for you
to fill in. Start with **[EXERCISES.md](EXERCISES.md)**.

## MountainCar-v0 environment

An under-powered car sits in a valley. Its engine is too weak to drive straight
up the right-hand hill, so the only way out is to rock back and forth and build
up momentum. The goal is to reach the flag at position `0.5`.

### State (observation) — 2 continuous values

| Index | Variable | Description | Range |
|:---:|---|---|---|
| 0 | position | Position of the car along the x-axis | -1.2 to 0.6 |
| 1 | velocity | Velocity of the car | -0.07 to 0.07 |

### Actions — 3 discrete

| Value | Action |
|:---:|---|
| 0 | Accelerate to the left |
| 1 | Don't accelerate |
| 2 | Accelerate to the right |

### Rewards

| Event | Reward |
|---|---|
| Every step taken | **-1** |
| Reaching the flag (position >= 0.5) | episode ends |

The reward is `-1` per step and nothing else, so the total return is simply the
negative of the episode length: **less negative is better**. Episodes are cut
off after 200 steps, which gives a floor of `-200` for a policy that never
reaches the flag. Anything around `-110` or better is considered solved.

This flat reward is what makes MountainCar interesting: there is no gradient to
follow toward the goal, so the agent has to stumble onto the flag by
exploration before it can learn anything at all.

## Install

```bash
uv sync
```

## Usage

All commands are exposed through the `mountaincar` CLI:

```bash
uv run mountaincar <command>
```

| Command | What it does |
|---|---|
| `version` | Show the package version |
| `list` | List the agents and whether each has a save file |
| `inspect` | Print the state/action spaces and some random transitions |
| `init <agent>` | Create a new, untrained agent and save it |
| `train <agent>` | Train an agent (resumes from its save if one exists) |
| `load <agent>` | Print a saved agent's info, optionally evaluate it |
| `sim <agent>` | Play episodes with a trained agent, printed step by step |
| `render <agent>` | Play episodes in a graphical window |
| `delete <agent>` | Delete an agent's save file |

`<agent>` is either `qlearning` or `dqn`.

### Example session

```bash
# See what the environment looks like
uv run mountaincar inspect --steps 3

# Train the tabular agent
uv run mountaincar train qlearning --episodes 10000

# How did it do?
uv run mountaincar load qlearning --eval

# Watch it drive
uv run mountaincar render qlearning --episodes 3
```

### Reproducible Q-learning experiment

For a fresh run that records per-episode CSV files, a JSON summary, and a
training curve, use the [experiment protocol](docs/qlearning-experiment.md):

```bash
uv run python scripts/run_qlearning_experiment.py \
  --output-dir artifacts/qlearning/seed-20250308 \
  --save-path saves/qlearning_seed-20250308.pkl
```

It starts a new agent instead of resuming a CLI checkpoint and refuses to
overwrite existing artifacts. The protocol uses separate training and greedy
evaluation seed ranges; see the document for interpretation and limitations.

## Agents

Both agents live in `src/mountain_car/agents/` and are written from scratch
(no Stable-Baselines3 or similar), so every part of the algorithm is visible --
and, in this repo, **partly left for you to write**. See [EXERCISES.md](EXERCISES.md).

### `qlearning` — tabular Q-Learning

The observation is only 2-dimensional and the environment publishes hard bounds
for both dimensions, so the state space is discretised into an
`n_bins x n_bins` grid (400 states by default) and stored in a plain Q-table.

Defaults: `n_bins=20`, `lr=0.1`, `gamma=0.99`, epsilon `1.0 -> 0.01` decaying by
`0.9995` per episode. See [Q-Learning foundations](docs/qlearning-foundations.md)
for the MDP, discretization, action-selection, and TD-update concepts, and the
[experiment protocol](docs/qlearning-experiment.md) for how to record observed
training and greedy-evaluation results without pre-claiming an outcome.

### `dqn` — Deep Q-Network

A small MLP on the raw 2-D observation, trained with experience replay and a
target network. A correct implementation scores about `-106` and reaches the
flag in 100/100 episodes, after roughly 2500 episodes (~5 min on CPU) -- better
than the tabular agent, and past the conventional "solved" threshold of `-110`.

Getting there takes more than transcribing the DQN pseudocode. MountainCar has
a reward structure that defeats the textbook version of the algorithm, and
Exercise 3 is about finding out how and why. That exercise ships with a ladder
of progressive clues, so it is a guided investigation rather than a wall.

> A note on hardware: none of this needs a GPU. The network is tiny and the
> batches are small, so a gradient step costs about 0.5 ms on CPU and the
> bottleneck is stepping the environment, not matrix multiplication. On a GPU
> this would most likely be *slower*, because per-kernel launch overhead would
> dominate work this small.

## Project layout

```
src/mountain_car/
├── cli.py              # argparse CLI, one command per function
└── agents/
    ├── qlearning.py    # tabular Q-Learning
    └── dqn.py          # DQN: QNetwork, ReplayBuffer, DQNAgent
saves/                  # agent save files land here
EXERCISES.md            # the exercises: what to implement, in what order
```
