from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "ppo_qos_router"

# The policy selects one of these candidate paths from s1 to s4.  Keeping a
# fixed action space makes it suitable for Stable-Baselines3's PPO algorithm.
CANDIDATE_PATHS = (
    ("s1", "s2", "s4"),
    ("s1", "s3", "s4"),
    ("s1", "s2", "s3", "s4"),
)

# Undirected links: capacity Mbps, one-way delay ms, baseline loss fraction.
LINKS = {
    frozenset(("s1", "s2")): (30.0, 4.0, 0.002),
    frozenset(("s2", "s4")): (25.0, 6.0, 0.003),
    frozenset(("s1", "s3")): (20.0, 8.0, 0.004),
    frozenset(("s3", "s4")): (30.0, 3.0, 0.002),
    frozenset(("s2", "s3")): (15.0, 2.0, 0.005),
}

MAX_DEMAND_MBPS = 20.0
EPISODE_STEPS = 120
