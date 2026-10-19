from sdn_rl.traffic_env import QosRoutingEnv


def test_environment_produces_valid_step():
    env = QosRoutingEnv(seed=7)
    observation, _ = env.reset()
    assert env.observation_space.contains(observation)
    next_observation, reward, terminated, truncated, info = env.step(0)
    assert env.observation_space.contains(next_observation)
    assert isinstance(reward, float)
    assert not terminated
    assert not truncated
    assert "throughput_mbps" in info
