from sdn_rl.config import get_port_capacity_bps
from sdn_rl.evaluate import min_hop_action, shortest_path_action
from sdn_rl.traffic_env import QosRoutingEnv


def test_environment_transition_has_valid_observation():
    env = QosRoutingEnv(seed=5)
    observation, _ = env.reset()
    next_observation, reward, _, _, info = env.step(0)
    assert env.observation_space.contains(observation)
    assert env.observation_space.contains(next_observation)
    assert isinstance(reward, float)
    assert "feasible" in info


def test_predictive_observation_has_forecast_features():
    env = QosRoutingEnv(seed=5, predictive=True)
    observation, _ = env.reset()
    assert observation.shape == (16,)
    assert env.observation_space.contains(observation)


def test_failure_makes_s2_s4_path_infeasible_after_midpoint():
    env = QosRoutingEnv(seed=5, episode_steps=4, failure_scenario=True)
    env.reset()
    env.step(0)
    env.step(0)
    _, _, _, _, info = env.step(0)
    assert not info["feasible"]
    assert info["loss_rate"] >= 1.0


def test_port_capacity_uses_actual_bottleneck_rate():
    assert get_port_capacity_bps(2, 4) == 15_000_000
    assert get_port_capacity_bps(1, 2) == 30_000_000


def test_baseline_selectors_are_deterministic():
    assert shortest_path_action(None) == 2
    assert min_hop_action(None) == 0
