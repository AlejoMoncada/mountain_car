"""End-to-end checks for the fresh Q-learning experiment runner."""
import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts/run_qlearning_experiment.py"


class QLearningExperimentTest(unittest.TestCase):
    def run_experiment(self, output_dir: Path, save_path: Path) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--episodes",
                "2",
                "--seed",
                "123",
                "--eval-seed",
                "456",
                "--eval-episodes",
                "2",
                "--output-dir",
                str(output_dir),
                "--save-path",
                str(save_path),
            ],
            cwd=PROJECT_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )

    def test_csv_and_summary_describe_the_same_tiny_run(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output_dir, save_path = root / "output", root / "agent.pkl"
            result = self.run_experiment(output_dir, save_path)
            self.assertEqual(result.returncode, 0, result.stderr)

            with (output_dir / "training_episodes.csv").open() as file:
                training = list(csv.DictReader(file))
            with (output_dir / "evaluation_episodes.csv").open() as file:
                evaluation = list(csv.DictReader(file))
            summary = json.loads((output_dir / "summary.json").read_text())

            self.assertEqual([row["seed"] for row in training], ["123", "124"])
            self.assertEqual([row["seed"] for row in evaluation], ["456", "457"])
            self.assertEqual(summary["statistics"]["training"]["episodes"], len(training))
            self.assertEqual(summary["statistics"]["evaluation"]["episodes"], len(evaluation))
            self.assertEqual(summary["best_return"]["training_trajectory"], max(float(row["return"]) for row in training))
            self.assertEqual(summary["best_return"]["greedy_evaluation"], max(float(row["return"]) for row in evaluation))
            self.assertTrue((output_dir / "training_curve.png").is_file())
            self.assertTrue(save_path.is_file())

    def test_nonempty_output_directory_is_rejected_before_training(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output_dir = root / "output"
            output_dir.mkdir()
            (output_dir / "previous.txt").write_text("do not overwrite")
            result = self.run_experiment(output_dir, root / "agent.pkl")

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("output directory is not empty", result.stderr)
            self.assertFalse((root / "agent.pkl").exists())


if __name__ == "__main__":
    unittest.main()
