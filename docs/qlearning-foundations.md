# Q-Learning foundations for MountainCar

This guide explains the tabular Q-Learning slice in this repository. It explains
what each calculation means; it does **not** report a trained-policy result.

## Quick path

1. `QLearningAgent` turns MountainCar's continuous observation into a grid key.
2. Training chooses exploratory or greedy actions, then applies one TD update.
3. Evaluation calls `predict(..., deterministic=True)`, so it never explores.

## The learning problem: an MDP

A Markov decision process (MDP) models interaction as `(S, A, P, R, gamma)`:

| Part | Meaning in MountainCar |
|---|---|
| `S` | State: position and velocity observed from the environment. |
| `A` | Actions: accelerate left, do nothing, or accelerate right. |
| `P` | Transition dynamics: the environment produces the next observation after an action. |
| `R` | Reward: MountainCar supplies `-1` for each step. |
| `gamma` | Discount factor that weights later rewards relative to immediate reward. |

The Markov assumption is that the current state contains the information needed
to reason about the next transition. This agent uses the supplied observation as
that state, after discretizing it.

## Discretization: a finite table for continuous observations

A Q-table needs hashable, finite state keys. For each observation dimension,
`QLearningAgent` creates `n_bins - 1` interior edges between the environment's
published lower and upper bounds. `np.digitize` maps an observation to an index
from `0` through `n_bins - 1`; the two indices form a tuple such as `(6, 11)`.

With the default `n_bins=20`, the table has at most `20 x 20 = 400` grid cells.
The exact lower and upper environment bounds map to valid first and last bin
indices. This compression makes tabular learning practical, but states in the
same cell become indistinguishable to the policy.

## Epsilon-greedy action selection

During training, epsilon-greedy selection balances two choices:

- **Explore** with probability `epsilon`: sample uniformly from all actions.
- **Exploit** otherwise: choose `argmax_a Q(s, a)`.

In code, a random sample strictly below `epsilon` explores. Setting
`deterministic=True` bypasses that sample and always exploits, even if epsilon
is high. Ties use NumPy's `argmax` tie-breaking, which selects the first largest
entry.

## Temporal-difference update

For a nonterminal transition, the one-step Q-Learning target is:

```text
target = reward + gamma * max_a' Q(next_state, a')
Q(state, action) = Q(state, action) + lr * (target - Q(state, action))
```

For example, let `Q(s, a) = 1`, `reward = -1`, `gamma = 0.9`,
`max_a' Q(s', a') = 5`, and `lr = 0.5`. Then:

```text
target = -1 + 0.9 * 5 = 3.5
new Q(s, a) = 1 + 0.5 * (3.5 - 1) = 2.25
```

This is a bootstrapped estimate: it uses the current table's best estimate for
the next state rather than waiting for a complete episode return.

## Termination is different from truncation

Gymnasium returns separate flags:

| Flag | Meaning | TD target in this agent |
|---|---|---|
| `terminated` | The environment reached a terminal state, such as MountainCar reaching its goal. | `reward`; do not bootstrap. |
| `truncated` | The episode stopped because of an external limit, such as the time limit. | Bootstrap from `next_state`. |

Both flags end the training loop. Only `terminated` is passed to `_update`, so
a truncated transition still uses the nonterminal target. That distinction
preserves the value estimate of the state reached at the time limit.

## Hyperparameters and evaluation

| Hyperparameter | Role |
|---|---|
| `n_bins` | Grid resolution. More bins preserve detail but require more visits. |
| `lr` | Learning rate: how far one TD target moves a table entry. |
| `gamma` | Discount factor for estimated future value. |
| `epsilon_start` | Initial probability of exploration. |
| `epsilon_end` | Lower bound for exploration after decay. |
| `epsilon_decay` | Per-episode multiplier applied to epsilon. |

Training calls `select_action` with its default `deterministic=False`, allowing
exploration and updating the table. Evaluation and rendering use deterministic
prediction: they choose the greedy table action and do not update values. An
evaluation score therefore measures the stored policy, not an exploratory
training episode.

## Limitations and pending empirical conclusions

- Discretization loses information within each grid cell and scales poorly as
  state dimensions or bin counts grow.
- Q-Learning's greedy bootstrap can overestimate values and may be sensitive to
  exploration and learning-rate schedules.
- This bounded slice verifies state bins, action-selection branches, numerical
  TD updates, and persistence only. It does not establish learning quality,
  solve rate, convergence, or a preferred hyperparameter setting.
- A reproducible empirical conclusion requires a separately recorded training
  budget, random seeds, evaluation protocol, and observed results. No full
  training was run for this slice.
