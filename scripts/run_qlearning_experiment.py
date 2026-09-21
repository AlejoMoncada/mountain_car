"""Run one fresh, reproducible tabular Q-learning experiment."""
import argparse
import csv
import json
import platform
import time
from importlib.metadata import version
from pathlib import Path
from typing import Any

import gymnasium as gym
import matplotlib  # pyright: ignore[reportMissingImports] -- installed through uv
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # pyright: ignore[reportMissingImports] -- installed through uv

from mountain_car.agents.qlearning import QLearningAgent

ENV_ID = "MountainCar-v0"
TRAIN_FIELDS = ("episode", "seed", "return", "steps", "terminated", "truncated", "epsilon", "time")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20_250_308)
    parser.add_argument("--eval-seed", type=int, default=30_250_308)
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/qlearning/seed-20250308"))
    parser.add_argument("--save-path", type=Path)
    return parser.parse_args()


def seed_range_overlaps(seed: int, count: int, other_seed: int, other_count: int) -> bool:
    return max(seed, other_seed) <= min(seed + count - 1, other_seed + other_count - 1)


def validate(args: argparse.Namespace) -> Path:
    if args.episodes < 1 or args.eval_episodes < 1:
        raise ValueError("--episodes and --eval-episodes must be positive")
    if args.seed < 0 or args.eval_seed < 0:
        raise ValueError("seeds must be non-negative")
    if seed_range_overlaps(args.seed, args.episodes, args.eval_seed, args.eval_episodes):
        raise ValueError("training and evaluation seed ranges must not overlap")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {args.output_dir}")
    save_path = args.save_path or Path(f"saves/qlearning_seed-{args.seed}.pkl")
    if save_path.exists():
        raise FileExistsError(f"checkpoint already exists: {save_path}")
    return save_path


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(TRAIN_FIELDS))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def evaluate(agent: QLearningAgent, episodes: int, seed: int) -> list[dict[str, Any]]:
    env = gym.make(ENV_ID)
    rows = []
    try:
        for index in range(episodes):
            episode_seed = seed + index
            started = time.perf_counter()
            obs, _ = env.reset(seed=episode_seed)
            total_return, steps, terminated, truncated = 0.0, 0, False, False
            while not (terminated or truncated):
                action = agent.select_action(agent.discretize(obs), deterministic=True)
                obs, reward, terminated, truncated, _ = env.step(action)
                total_return += float(reward)
                steps += 1
            rows.append({
                "episode": index + 1,
                "seed": episode_seed,
                "return": total_return,
                "steps": steps,
                "terminated": terminated,
                "truncated": truncated,
                "epsilon": agent.epsilon,
                "time": time.perf_counter() - started,
            })
    finally:
        env.close()
    return rows


def statistics(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    returns = np.array([row["return"] for row in rows], dtype=float)
    steps = np.array([row["steps"] for row in rows], dtype=float)
    return {
        "episodes": len(rows),
        "mean_return": float(returns.mean()),
        "std_return": float(returns.std()),
        "min_return": float(returns.min()),
        "max_return": float(returns.max()),
        "mean_steps": float(steps.mean()),
        "successes": sum(bool(row["terminated"]) for row in rows),
    }


def plot_training(rows: list[dict[str, Any]], path: Path) -> None:
    returns = np.array([row["return"] for row in rows], dtype=float)
    episodes = np.arange(1, len(returns) + 1)
    plt.figure(figsize=(9, 5))
    plt.plot(episodes, returns, alpha=0.35, label="Episode return")
    if len(returns) >= 100:
        rolling = np.convolve(returns, np.ones(100) / 100, mode="valid")
        plt.plot(np.arange(100, len(returns) + 1), rolling, label="Rolling mean (100)")
    plt.xlabel("Training episode")
    plt.ylabel("Return (less negative is better)")
    plt.title("Q-learning training returns")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def main() -> None:
    args = parse_args()
    try:
        save_path = validate(args)
    except (ValueError, FileExistsError) as error:
        raise SystemExit(f"error: {error}") from error

    args.output_dir.mkdir(parents=True, exist_ok=True)
    agent = QLearningAgent(ENV_ID)
    training_rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    agent.train(args.episodes, seed=args.seed, episode_callback=training_rows.append)
    training_seconds = time.perf_counter() - started

    started = time.perf_counter()
    evaluation_rows = evaluate(agent, args.eval_episodes, args.eval_seed)
    evaluation_seconds = time.perf_counter() - started
    agent.save(save_path)

    write_csv(args.output_dir / "training_episodes.csv", training_rows)
    write_csv(args.output_dir / "evaluation_episodes.csv", evaluation_rows)
    plot_training(training_rows, args.output_dir / "training_curve.png")
    summary = {
        "config": {
            "environment": ENV_ID,
            "episodes": args.episodes,
            "seed": args.seed,
            "eval_seed": args.eval_seed,
            "eval_episodes": args.eval_episodes,
            "fresh_agent": True,
            "checkpoint": str(save_path),
        },
        "versions": {name: version(name) for name in ("gymnasium", "matplotlib", "numpy", "mountain_car")}
        | {"python": platform.python_version()},
        "durations_seconds": {"training": training_seconds, "evaluation": evaluation_seconds},
        "statistics": {"training": statistics(training_rows), "evaluation": statistics(evaluation_rows)},
        "best_return": {
            "training_trajectory": max(row["return"] for row in training_rows),
            "greedy_evaluation": max(row["return"] for row in evaluation_rows),
        },
    }
    with (args.output_dir / "summary.json").open("w") as file:
        json.dump(summary, file, indent=2)
    print(f"Wrote experiment artifacts to {args.output_dir}")


if __name__ == "__main__":
    main()
