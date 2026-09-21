# DQN foundations

This exercise implements the neural-network prediction and Bellman-update parts of DQN. It also changes only the exploratory action-selection schedule for MountainCar: the neural-network prediction, Bellman update, reward handling, replay buffer, and target-network synchronization remain unchanged.

## Quick path

1. `QNetwork` maps a batch of observations to one unbounded Q-value per discrete action.
2. The online network learns from replayed transitions toward Bellman targets made by the target network.
3. A true terminal transition stops bootstrapping; a time-limit truncation does not.

## Network

`QNetwork` is a fully connected model:

```text
state_dim -> hidden -> hidden -> action_dim
```

A ReLU follows each hidden layer. The output layer has no activation because Q-values estimate expected return; they are neither probabilities nor bounded scores. For a batch shaped `(B, state_dim)`, the model returns `(B, action_dim)`.

## Bellman update

For a replayed transition `(s, a, r, s', terminated)`, the online network selects the value for the action actually taken:

```text
current_q = Q_online(s, a)
```

The training target is:

```text
target_q = r + gamma * max_a' Q_target(s', a') * (1 - terminated)
```

`terminated` is stored as `1` for a real terminal state and `0` otherwise. This masking prevents an invalid bootstrap after a true terminal state.

## Target network and replay buffer

| Component | Purpose |
| --- | --- |
| Online network | Produces `current_q` and receives gradients from the loss. |
| Target network | Produces `next_q` inside `torch.no_grad()`, so the target remains fixed during each gradient step. It is synchronized by the existing training loop. |
| Replay buffer | Keeps a fixed-size FIFO collection of transitions and samples mini-batches, which reduces correlation between consecutive experiences. |

## Termination is not truncation

An environment can end an episode in two different ways:

- **Termination** means the environment reached a real terminal state. Its next-state value must be masked out.
- **Truncation** means an external limit, such as a time limit, ended the episode. The transition still has a valid bootstrap value, so it is stored as non-terminal.

The provided training loop records `terminated`, not `terminated or truncated`, in the replay buffer.

## Correlated exploratory runs

MountainCar needs sustained pushes to build momentum. The diagnosis in
[`dqn-exploration-diagnosis.md`](dqn-exploration-diagnosis.md) measured that
independent uniformly random actions did not reach the flag in 500 sampled
episodes, and that the longest observed identical-action run was 11 steps.
That measurement motivates making sustained exploratory actions reachable; it
does not establish an optimal run length or demonstrate trained performance.

`DQNAgent` therefore retains epsilon-greedy exploration but adds
`explore_run_length`, whose default is 20. When an exploratory run begins, the
agent samples an action uniformly and samples a total run length uniformly
from 1 through `explore_run_length`. It reuses that action until the selected
length is consumed. The run state is reset at the beginning of every training
episode, so no exploratory action carries across an environment reset.

`deterministic=True` remains pure greedy evaluation even if an exploratory run
is active. This change is deliberately limited to action selection: it does
not alter rewards, the Bellman target, replay-buffer contents, target-network
updates, or the existing hyperparameter defaults. `explore_run_length` is
persisted with the other DQN hyperparameters.

The default 20 is a practical bound motivated by the need for sustained runs,
not a tuned claim. Correlated runs increase the chance of long pushes, but do
not guarantee that their direction, timing, or sequence will reach the goal;
training and evaluation evidence belong to a separate experiment.
