"""Focused unit tests for tabular Q-Learning updates and action selection."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from mountain_car.agents.qlearning import QLearningAgent


class QLearningAgentTest(unittest.TestCase):
    def make_agent(self, **kwargs) -> QLearningAgent:
        return QLearningAgent("MountainCar-v0", n_bins=4, **kwargs)

    def test_discretize_maps_environment_bounds_to_valid_bin_indices(self) -> None:
        agent = self.make_agent()

        low = agent.discretize(np.array([-1.2, -0.07]))
        high = agent.discretize(np.array([0.6, 0.07]))

        self.assertEqual(low, (0, 0))
        self.assertEqual(high, (agent.n_bins - 1, agent.n_bins - 1))
        self.assertTrue(all(0 <= index < agent.n_bins for index in low + high))

    def test_deterministic_selection_exploits_without_sampling_exploration(self) -> None:
        agent = self.make_agent(epsilon_start=1.0)
        state = (1, 2)
        agent.q_table[state] = np.array([0.0, 4.0, 1.0])

        with patch("mountain_car.agents.qlearning.np.random.random") as random_value:
            self.assertEqual(agent.select_action(state, deterministic=True), 1)

        random_value.assert_not_called()

    def test_epsilon_greedy_explores_when_sample_is_below_epsilon(self) -> None:
        agent = self.make_agent(epsilon_start=0.5)
        state = (1, 2)
        agent.q_table[state] = np.array([0.0, 4.0, 1.0])

        with (
            patch("mountain_car.agents.qlearning.np.random.random", return_value=0.1),
            patch("mountain_car.agents.qlearning.np.random.randint", return_value=2) as randint,
        ):
            action = agent.select_action(state)

        self.assertEqual(action, 2)
        randint.assert_called_once_with(agent.n_actions)

    def test_epsilon_greedy_exploits_when_sample_is_not_below_epsilon(self) -> None:
        agent = self.make_agent(epsilon_start=0.5)
        state = (1, 2)
        agent.q_table[state] = np.array([0.0, 4.0, 1.0])

        with (
            patch("mountain_car.agents.qlearning.np.random.random", return_value=0.5),
            patch("mountain_car.agents.qlearning.np.random.randint") as randint,
        ):
            action = agent.select_action(state)

        self.assertEqual(action, 1)
        randint.assert_not_called()

    def test_terminal_update_uses_reward_without_bootstrapping(self) -> None:
        agent = self.make_agent(lr=0.25, gamma=0.9)
        state, next_state = (0, 0), (1, 1)
        agent.q_table[state] = np.array([2.0, 0.0, 0.0])
        agent.q_table[next_state] = np.array([100.0, 0.0, 0.0])

        agent._update(state, action=0, reward=3.0, next_state=next_state, terminated=True)

        self.assertAlmostEqual(agent.q_table[state][0], 2.25)

    def test_nonterminal_update_bootstraps_from_the_best_next_action(self) -> None:
        agent = self.make_agent(lr=0.5, gamma=0.9)
        state, next_state = (0, 0), (1, 1)
        agent.q_table[state] = np.array([0.0, 0.0, 1.0])
        agent.q_table[next_state] = np.array([3.0, 5.0, 4.0])

        agent._update(state, action=2, reward=-1.0, next_state=next_state, terminated=False)

        self.assertAlmostEqual(agent.q_table[state][2], 2.25)

    def test_save_and_load_preserve_learned_values_and_training_state(self) -> None:
        agent = self.make_agent(epsilon_start=0.3, lr=0.25, gamma=0.9)
        agent.q_table[(2, 3)] = np.array([1.0, -2.0, 4.5])
        agent.training_episodes = 7

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "agent.pkl"
            agent.save(path)
            loaded = QLearningAgent.load(path)

        self.assertEqual(loaded.training_episodes, 7)
        self.assertEqual(loaded.epsilon, 0.3)
        self.assertEqual(loaded.n_bins, 4)
        np.testing.assert_array_equal(loaded.q_table[(2, 3)], [1.0, -2.0, 4.5])


if __name__ == "__main__":
    unittest.main()
