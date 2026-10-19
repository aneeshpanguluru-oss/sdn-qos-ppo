"""Evaluate routing policies under normal traffic and a controlled link failure."""

from __future__ import annotations

import argparse
import time
from statistics import mean
from typing import Callable

from stable_baselines3 import PPO

from .config import CANDIDATE_PATHS, LINKS, MODEL_PATH, PREDICTIVE_MODEL_PATH
from .traffic_env import QosRoutingEnv

ActionSelector = Callable[[object], int]


def shortest_path_action(_: object) -> int:
    delays = [sum(LINKS[frozenset((a, b))][1] for a, b in zip(path, path[1:])) for path in CANDIDATE_PATHS]
    return int(min(range(len(delays)), key=delays.__getitem__))


def min_hop_action(_: object) -> int:
    return int(min(range(len(CANDIDATE_PATHS)), key=lambda index: len(CANDIDATE_PATHS[index])))


def evaluate_policy(label: str, selector: ActionSelector, episodes: int, predictive: bool, failure_scenario: bool) -> dict[str, float | str]:
    records: list[dict] = []
    decision_times_us: list[float] = []
    churn_per_episode: list[int] = []
    for episode in range(episodes):
        env = QosRoutingEnv(seed=10_000 + episode, predictive=predictive, failure_scenario=failure_scenario)
        observation, _ = env.reset()
        previous_action, churn, done = None, 0, False
        while not done:
            started = time.perf_counter_ns()
            action = selector(observation)
            decision_times_us.append((time.perf_counter_ns() - started) / 1_000)
            if previous_action is not None and action != previous_action:
                churn += 1
            previous_action = action
            observation, _, terminated, truncated, info = env.step(action)
            records.append(info)
            done = terminated or truncated
        churn_per_episode.append(churn)
    return {
        "policy": label,
        "throughput": mean(record["throughput_mbps"] for record in records),
        "max_util": mean(record["max_utilization"] for record in records),
        "delay": mean(record["delay_ms"] for record in records),
        "loss": mean(record["loss_rate"] for record in records) * 100,
        "feasibility": mean(record["feasible"] for record in records) * 100,
        "decision_us": mean(decision_times_us),
        "route_churn": mean(churn_per_episode),
    }


def print_results(title: str, rows: list[dict[str, float | str]]) -> None:
    print(f"\n{title}")
    print(f"{'Policy':<22} {'Throughput':>12} {'Max util':>10} {'Delay':>10} {'Loss':>10} {'Feasible':>10} {'Decision':>13} {'Churn/ep':>10}")
    for row in rows:
        print(f"{row['policy']:<22} {row['throughput']:>10.2f}M {row['max_util']:>10.3f} {row['delay']:>8.2f}ms {row['loss']:>9.2f}% {row['feasibility']:>9.1f}% {row['decision_us']:>10.1f}us {row['route_churn']:>10.1f}")


def run_comparison(standard: PPO, predictive_model: PPO, episodes: int, failure_scenario: bool) -> None:
    rows = [
        evaluate_policy("Shortest path (delay)", shortest_path_action, episodes, False, failure_scenario),
        evaluate_policy("Min hop", min_hop_action, episodes, False, failure_scenario),
        evaluate_policy("PPO standard QoS", lambda obs: int(standard.predict(obs, deterministic=True)[0]), episodes, False, failure_scenario),
        evaluate_policy("PPO predictive QoS", lambda obs: int(predictive_model.predict(obs, deterministic=True)[0]), episodes, True, failure_scenario),
    ]
    title = "Fault tolerance: s2-s4 fails at step 60" if failure_scenario else "Normal traffic scenario"
    print_results(title, rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=f"{MODEL_PATH}.zip")
    parser.add_argument("--predictive-model", default=f"{PREDICTIVE_MODEL_PATH}.zip")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--compare-all", action="store_true")
    parser.add_argument("--failure-scenario", action="store_true")
    args = parser.parse_args()
    standard = PPO.load(args.model)
    if args.compare_all:
        predictive_model = PPO.load(args.predictive_model)
        run_comparison(standard, predictive_model, args.episodes, failure_scenario=False)
        run_comparison(standard, predictive_model, args.episodes, failure_scenario=True)
        return
    row = evaluate_policy("PPO standard QoS", lambda obs: int(standard.predict(obs, deterministic=True)[0]), args.episodes, False, args.failure_scenario)
    print_results("PPO evaluation", [row])


if __name__ == "__main__":
    main()
