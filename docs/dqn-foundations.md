# DQN foundations

This exercise implements the neural-network prediction and Bellman-update parts of DQN. It deliberately leaves MountainCar exploration unchanged: the remaining exploration limitation belongs to exercise 3.

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

## Known exploration limitation

The current policy uses textbook per-step epsilon-greedy exploration. In MountainCar, independent random actions may fail to produce the sustained action sequences needed to build momentum. This is intentionally unresolved in this unit: exercise 3 will investigate and change action selection if the evidence supports it. No conclusion about MountainCar learning performance is claimed here.
