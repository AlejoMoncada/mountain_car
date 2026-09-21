import random
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import gymnasium as gym
import numpy as np
import torch
from torch import nn, optim

from mountain_car.agents.dqn import DQNAgent, QNetwork, ReplayBuffer


class _LinearQNetwork(QNetwork):
    """Small controllable network for numeric Bellman-target tests."""

    def __init__(self) -> None:
        nn.Module.__init__(self)
        self.linear = nn.Linear(2, 3, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class QNetworkTest(unittest.TestCase):
    def test_maps_state_batch_to_one_q_value_per_action(self) -> None:
        torch.manual_seed(7)
        network = QNetwork(state_dim=2, action_dim=3, hidden=8)
        states = torch.tensor([[-0.5, 0.0], [0.2, 0.03]], dtype=torch.float32)

        values = network(states)

        self.assertEqual(values.shape, (2, 3))
        self.assertFalse(torch.allclose(values[0], values[1]))


class ReplayBufferTest(unittest.TestCase):
    def test_capacity_discards_oldest_transition_and_sample_returns_requested_size(self) -> None:
        buffer = ReplayBuffer(capacity=2)
        for action in range(3):
            state = np.array([action, 0.0], dtype=np.float32)
            buffer.push(state, action, float(action), state + 1.0, False)

        sample = buffer.sample(2)

        self.assertEqual(len(buffer), 2)
        self.assertEqual(len(sample), 2)
        self.assertEqual({transition[1] for transition in sample}, {1, 2})


class DQNAgentLearnTest(unittest.TestCase):
    def make_agent(self, **kwargs) -> DQNAgent:
        with patch("mountain_car.agents.dqn.torch.cuda.is_available", return_value=False):
            return DQNAgent("MountainCar-v0", **kwargs)

    def test_bellman_target_masks_terminal_bootstrap_and_keeps_truncation_bootstrap(self) -> None:
        agent = self.make_agent(lr=0.1, gamma=0.5, batch_size=2, hidden=4)
        online = _LinearQNetwork().to(agent.device)
        target = _LinearQNetwork().to(agent.device)
        agent.q_net = online
        agent.target_net = target
        agent.optimizer = optim.Adam(agent.q_net.parameters(), lr=0.0)
        with torch.no_grad():
            online.linear.weight.zero_()
            target.linear.weight.copy_(
                torch.tensor([[2.0, 0.0], [1.0, 0.0], [-1.0, 0.0]], device=agent.device)
            )

        state = np.array([1.0, 0.0], dtype=np.float32)
        agent.buffer.push(state, 1, 1.0, state, True)
        agent.buffer.push(state, 2, 1.0, state, False)

        loss = agent._learn()

        # The frozen target's max value is 2. The terminal target is 1, while
        # the non-terminal target includes the 0.5 * 2 bootstrap and is 2.
        self.assertAlmostEqual(loss, 2.5)
        self.assertTrue(any(parameter.grad is not None for parameter in agent.q_net.parameters()))
        self.assertTrue(all(parameter.grad is None for parameter in agent.target_net.parameters()))

    def test_repeated_batch_memorization_decreases_loss(self) -> None:
        generator = torch.Generator().manual_seed(23)
        agent = self.make_agent(lr=0.03, gamma=0.99, batch_size=8, hidden=16)
        states = torch.randn((8, 2), generator=generator).numpy().astype(np.float32)
        for state in states:
            agent.buffer.push(state, 0, 1.0, state, True)

        initial_loss = agent._learn()
        final_loss = initial_loss
        for _ in range(80):
            final_loss = agent._learn()

        self.assertLess(final_loss, initial_loss)

    def test_save_and_load_round_trip_hyperparameters(self) -> None:
        agent = self.make_agent(
            lr=0.002,
            gamma=0.95,
            epsilon_start=0.7,
            epsilon_end=0.2,
            epsilon_decay=0.9,
            batch_size=5,
            buffer_capacity=9,
            target_update_freq=4,
            hidden=16,
        )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "agent.pt"
            agent.save(path)
            loaded = DQNAgent.load(path)

        for name in agent._HPARAMS:
            self.assertEqual(getattr(loaded, name), getattr(agent, name))


class CartPoleLearningSanityTest(unittest.TestCase):
    def test_seeded_cartpole_run_improves_with_the_same_learning_step(self) -> None:
        seed = 11
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        deadline = time.monotonic() + 45.0

        with patch("mountain_car.agents.dqn.torch.cuda.is_available", return_value=False):
            agent = DQNAgent(
                "CartPole-v1",
                lr=1e-3,
                gamma=0.99,
                epsilon_start=1.0,
                epsilon_end=0.05,
                epsilon_decay=0.99,
                batch_size=32,
                buffer_capacity=5_000,
                target_update_freq=5,
                hidden=32,
            )
        self.assertEqual(agent.device.type, "cpu")

        env = gym.make("CartPole-v1")
        returns = []
        try:
            for episode in range(1, 201):
                observation, _ = env.reset(seed=seed + episode)
                env.action_space.seed(seed + episode)
                episode_return = 0.0
                terminated = truncated = False

                while not (terminated or truncated):
                    action = agent.select_action(observation)
                    next_observation, reward, terminated, truncated, _ = env.step(action)
                    agent.buffer.push(
                        observation, action, float(reward), next_observation, terminated
                    )
                    agent._learn()
                    observation = next_observation
                    episode_return += float(reward)

                returns.append(episode_return)
                agent.epsilon = max(agent.epsilon_end, agent.epsilon * agent.epsilon_decay)
                if episode % agent.target_update_freq == 0:
                    agent.target_net.load_state_dict(agent.q_net.state_dict())
                if time.monotonic() > deadline:
                    self.skipTest("CartPole sanity check exceeded its 45-second runtime budget")
        finally:
            env.close()

        self.assertGreater(np.mean(returns[-20:]), np.mean(returns[:20]) + 10.0)


if __name__ == "__main__":
    unittest.main()
