"""A small, reproducible SDN traffic-engineering environment for PPO."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .config import CANDIDATE_PATHS, EPISODE_STEPS, LINKS, MAX_DEMAND_MBPS


class QosRoutingEnv(gym.Env):
    """Route one bandwidth-demanding flow per step through candidate paths.

    Observation has four normalized QoS values per candidate path:
    maximum utilization, delay, loss estimate, and residual capacity.  The
    final value is the normalized demand of the incoming flow.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, episode_steps: int = EPISODE_STEPS, seed: int | None = None, predictive: bool = False, failure_scenario: bool = False):
        super().__init__()
        self.episode_steps = episode_steps
        self.predictive = predictive
        self.failure_scenario = failure_scenario
        self.rng = np.random.default_rng(seed)
        self.link_load: dict[frozenset[str], float] = {}
        self.background_load: dict[frozenset[str], float] = {}
        self.active_flows: list[tuple[list[frozenset[str]], float, int]] = []
        self.previous_action: int | None = None
        self.demand_mbps = 0.0
        self.step_count = 0
        self.previous_path_utilization = [0.0] * len(CANDIDATE_PATHS)
        feature_count = len(CANDIDATE_PATHS) * (5 if predictive else 4) + 1
        self.observation_space = spaces.Box(0.0, 1.0, shape=(feature_count,), dtype=np.float32)
        self.action_space = spaces.Discrete(len(CANDIDATE_PATHS))

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.background_load = {
            link: float(self.rng.uniform(0.05, 0.35) * capacity)
            for link, (capacity, _, _) in LINKS.items()
        }
        self.active_flows = []
        self._recalculate_link_load()
        self.previous_action = None
        self.step_count = 0
        self._sample_demand()
        initial_observation = self._observation()
        self._remember_path_utilizations()
        return initial_observation, {}

    def step(self, action: int):
        path = CANDIDATE_PATHS[int(action)]
        path_links = self._links_for_path(path)
        feasible = all(
            self._link_available(link) and self.link_load[link] + self.demand_mbps <= LINKS[link][0]
            for link in path_links
        )

        if feasible:
            # A traffic request remains active for a few control intervals.
            # This gives route choices a realistic, but not permanent, effect
            # on subsequent decisions.
            self.active_flows.append((path_links, self.demand_mbps, int(self.rng.integers(3, 8))))
            self._recalculate_link_load()
            throughput = self.demand_mbps
        else:
            throughput = 0.0

        # Score the state *after* placing the flow.  This gives PPO a direct
        # signal that a path becoming congested is worse than a feasible path
        # with spare capacity.
        max_util, delay_ms, loss, residual = self._path_metrics(path)
        route_change = self.previous_action is not None and int(action) != self.previous_action
        if feasible:
            reward = (
                3.0 * (throughput / MAX_DEMAND_MBPS)
                + 1.2 * min(residual / MAX_DEMAND_MBPS, 1.0)
                - 1.0 * max_util
                - 0.01 * delay_ms
                - 3.0 * loss
                - (0.04 if route_change else 0.0)
            )
        else:
            # Rejecting an over-capacity route must always be less attractive
            # than any viable path.
            reward = -2.5

        self._advance_background_traffic()
        self.previous_action = int(action)
        self.step_count += 1
        terminated = self.step_count >= self.episode_steps
        self._sample_demand()
        info = {
            "chosen_path": path,
            "throughput_mbps": throughput,
            "max_utilization": max_util,
            "delay_ms": delay_ms,
            "loss_rate": loss,
            "feasible": feasible,
        }
        observation = self._observation()
        self._remember_path_utilizations()
        return observation, float(reward), terminated, False, info

    def _sample_demand(self) -> None:
        self.demand_mbps = float(self.rng.uniform(2.0, MAX_DEMAND_MBPS))

    @staticmethod
    def _links_for_path(path: tuple[str, ...]) -> list[frozenset[str]]:
        return [frozenset((left, right)) for left, right in zip(path, path[1:])]

    def _path_metrics(self, path: tuple[str, ...]) -> tuple[float, float, float, float]:
        values = []
        for link in self._links_for_path(path):
            capacity, delay, base_loss = LINKS[link]
            if not self._link_available(link):
                values.append((1.0, delay, 1.0, 0.0))
                continue
            utilization = min(self.link_load[link] / capacity, 1.5)
            dynamic_loss = base_loss + max(utilization - 0.75, 0.0) * 0.12
            values.append((utilization, delay, dynamic_loss, max(capacity - self.link_load[link], 0.0)))
        return max(x[0] for x in values), sum(x[1] for x in values), sum(x[2] for x in values), min(x[3] for x in values)

    def _observation(self) -> np.ndarray:
        features: list[float] = []
        for index, path in enumerate(CANDIDATE_PATHS):
            max_util, delay, loss, residual = self._path_metrics(path)
            features.extend((min(max_util, 1.0), min(delay / 30.0, 1.0), min(loss / 0.15, 1.0), min(residual / MAX_DEMAND_MBPS, 1.0)))
            if self.predictive:
                # One-step momentum forecast: current load plus its latest change.
                previous = self.previous_path_utilization[index]
                features.append(float(np.clip(max_util + (max_util - previous), 0.0, 1.0)))
        features.append(self.demand_mbps / MAX_DEMAND_MBPS)
        return np.asarray(features, dtype=np.float32)

    def _remember_path_utilizations(self) -> None:
        self.previous_path_utilization = [self._path_metrics(path)[0] for path in CANDIDATE_PATHS]

    def _link_available(self, link: frozenset[str]) -> bool:
        # Controlled fault injection halfway through each episode.
        return not (self.failure_scenario and self.step_count >= self.episode_steps // 2 and link == frozenset(("s2", "s4")))

    def _advance_background_traffic(self) -> None:
        remaining_flows = []
        for path_links, demand, lifetime in self.active_flows:
            if lifetime > 1:
                remaining_flows.append((path_links, demand, lifetime - 1))
        self.active_flows = remaining_flows
        for link, (capacity, _, _) in LINKS.items():
            background = self.rng.normal(0.0, capacity * 0.04)
            self.background_load[link] = float(np.clip(self.background_load[link] * 0.93 + background, 0.0, capacity * 0.70))
        self._recalculate_link_load()

    def _recalculate_link_load(self) -> None:
        self.link_load = dict(self.background_load)
        for path_links, demand, _ in self.active_flows:
            for link in path_links:
                self.link_load[link] += demand

    def render(self):
        print({"step": self.step_count, "demand_mbps": round(self.demand_mbps, 2), "loads": self.link_load})
