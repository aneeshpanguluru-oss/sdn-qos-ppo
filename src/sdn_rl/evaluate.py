"""Evaluate a trained PPO router in the simulated SDN environment."""

from __future__ import annotations

import argparse
from statistics import mean
from typing import Callable

from stable_baselines3 import PPO

from .config import CANDIDATE_PATHS, LINKS
from .traffic_env import QosRoutingEnv


def shortest_path_action() -> int:
    """Return the candidate with the lowest fixed end-to-end link delay."""
    path_delays = [
        sum(LINKS[frozenset((left, right))][1] for left, right in zip(path, path[1:]))
        for path in CANDIDATE_PATHS
    ]
    return int(min(range(len(path_delays)), key=path_delays.__getitem__))


def evaluate(selector: Callable, episodes: int, label: str) -> None:
    results = []
    for episode in range(episodes):
        env = QosRoutingEnv(seed=10_000 + episode)
        observation, _ = env.reset()
        done = False
        episode_info = []
        while not done:
            action = selector(observation)
            observation, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_info.append(info)
        results.extend(episode_info)
    print(f"{label} evaluation summary")
    print(f"mean throughput Mbps: {mean(x['throughput_mbps'] for x in results):.2f}")
    print(f"mean maximum utilization: {mean(x['max_utilization'] for x in results):.3f}")
    print(f"mean delay ms: {mean(x['delay_ms'] for x in results):.2f}")
    print(f"feasible decisions: {mean(x['feasible'] for x in results) * 100:.1f}%")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/ppo_qos_router.zip")
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--compare-shortest", action="store_true", help="Evaluate PPO and a fixed shortest-path baseline on identical scenarios.")
    args = parser.parse_args()
    model = PPO.load(args.model)
    evaluate(lambda observation: int(model.predict(observation, deterministic=True)[0]), args.episodes, "PPO")
    if args.compare_shortest:
        evaluate(lambda observation: shortest_path_action(), args.episodes, "Shortest-path baseline")


if __name__ == "__main__":
    main()
