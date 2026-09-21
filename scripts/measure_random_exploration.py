"""Measure uniformly random exploration on MountainCar-v0 reproducibly."""
import argparse
import csv
import json
import platform
from collections import Counter
from importlib.metadata import version
from pathlib import Path
from statistics import fmean
from typing import Any

import gymnasium as gym

ENV_ID = "MountainCar-v0"
ACTION_COUNT_FIELDS = ("action_0_count", "action_1_count", "action_2_count")
CSV_FIELDS = [
    "episode_index",
    "seed",
    "return",
    "steps",
    "terminated",
    "truncated",
    "final_position",
    "max_position",
    "longest_identical_action_run",
    "longest_run_action",
    "longest_run_start_step",
    *ACTION_COUNT_FIELDS,
]


def as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"expected a numeric environment value, got {value!r}") from error


def as_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"expected an integer action, got {value!r}") from error


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20_250_310)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/dqn/random-exploration/seed-20250310"),
    )
    return parser.parse_args()


def validate(args: argparse.Namespace) -> None:
    if args.episodes < 1:
        raise ValueError("--episodes must be positive")
    if args.seed < 0:
        raise ValueError("--seed must be non-negative")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {args.output_dir}")


def play_episode(
    env: Any, episode_index: int, base_seed: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    episode_seed = base_seed + episode_index
    # Resetting and seeding the sampler make both initial state and actions repeatable.
    observation, _ = env.reset(seed=episode_seed)
    env.action_space.seed(episode_seed)

    action_counts: Counter[int] = Counter()
    total_return = 0.0
    steps = 0
    terminated = truncated = False
    max_position = as_float(observation[0])
    current_action: int | None = None
    current_run = 0
    longest_run = {"length": 0, "action": None, "start_step": None}

    while not (terminated or truncated):
        action = as_int(env.action_space.sample())
        action_counts[action] += 1
        steps += 1

        if action == current_action:
            current_run += 1
        else:
            current_action = action
            current_run = 1
        if current_run > longest_run["length"]:
            longest_run = {
                "length": current_run,
                "action": action,
                "start_step": steps - current_run + 1,
            }

        observation, reward, terminated, truncated, _ = env.step(action)
        total_return += as_float(reward)
        position = as_float(observation[0])
        max_position = max(max_position, position)

    row: dict[str, Any] = {
        "episode_index": episode_index,
        "seed": episode_seed,
        "return": total_return,
        "steps": steps,
        "terminated": terminated,
        "truncated": truncated,
        "final_position": as_float(observation[0]),
        "max_position": max_position,
        "longest_identical_action_run": longest_run["length"],
        "longest_run_action": longest_run["action"],
        "longest_run_start_step": longest_run["start_step"],
    }
    for action in range(env.action_space.n):
        row[f"action_{action}_count"] = action_counts[action]

    return row, longest_run


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def summarize(rows: list[dict[str, Any]], longest_run: dict[str, Any]) -> dict[str, Any]:
    final_positions = [as_float(row["final_position"]) for row in rows]
    max_positions = [as_float(row["max_position"]) for row in rows]
    returns = [as_float(row["return"]) for row in rows]
    terminated_count = sum(bool(row["terminated"]) for row in rows)
    truncated_count = sum(bool(row["truncated"]) for row in rows)
    return {
        "episodes": len(rows),
        "terminated_count": terminated_count,
        "truncated_count": truncated_count,
        "success_rate": terminated_count / len(rows),
        "mean_return": fmean(returns),
        "mean_final_position": fmean(final_positions),
        "max_final_position": max(final_positions),
        "mean_max_position": fmean(max_positions),
        "max_max_position": max(max_positions),
        "longest_identical_action_run": longest_run["length"],
        "longest_run_context": longest_run,
    }


def print_summary(output_dir: Path, summary: dict[str, Any]) -> None:
    context = summary["longest_run_context"]
    print("Random MountainCar exploration measurement")
    print(f"Output directory: {output_dir}")
    print(f"Episodes: {summary['episodes']}")
    print(f"Terminated (flag reached): {summary['terminated_count']}")
    print(f"Truncated (time limit): {summary['truncated_count']}")
    print(f"Success rate: {summary['success_rate']:.4%}")
    print(f"Mean return: {summary['mean_return']:.6f}")
    print(f"Mean final position: {summary['mean_final_position']:.6f}")
    print(f"Max final position: {summary['max_final_position']:.6f}")
    print(f"Mean max position: {summary['mean_max_position']:.6f}")
    print(f"Max max position: {summary['max_max_position']:.6f}")
    print(
        "Longest identical consecutive action run: "
        f"{summary['longest_identical_action_run']} "
        f"(action {context['action']}, episode {context['episode_index']}, "
        f"starting at step {context['start_step']})"
    )


def main() -> None:
    args = parse_args()
    try:
        validate(args)
    except (ValueError, FileExistsError) as error:
        raise SystemExit(f"error: {error}") from error

    args.output_dir.mkdir(parents=True, exist_ok=True)
    env = gym.make(ENV_ID)
    rows: list[dict[str, Any]] = []
    longest_run = {"length": 0, "action": None, "episode_index": None, "start_step": None}
    try:
        for episode_index in range(args.episodes):
            row, episode_longest_run = play_episode(env, episode_index, args.seed)
            rows.append(row)
            if episode_longest_run["length"] > longest_run["length"]:
                longest_run = episode_longest_run | {"episode_index": episode_index}
    finally:
        env.close()

    summary = summarize(rows, longest_run)
    summary |= {
        "environment": ENV_ID,
        "environment_max_episode_steps": env.spec.max_episode_steps if env.spec else None,
        "base_seed": args.seed,
        "action_sampler": "env.action_space.sample()",
        "action_space_seeded_per_episode": True,
        "versions": {"gymnasium": version("gymnasium"), "python": platform.python_version()},
        "episode_csv": "episodes.csv",
    }
    write_csv(args.output_dir / "episodes.csv", rows)
    with (args.output_dir / "summary.json").open("w") as file:
        json.dump(summary, file, indent=2)
    print_summary(args.output_dir, summary)


if __name__ == "__main__":
    main()
