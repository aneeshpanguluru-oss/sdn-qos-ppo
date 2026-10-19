"""Train the QoS-aware PPO policy."""

from __future__ import annotations

import argparse

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor

from .config import MODEL_PATH, PREDICTIVE_MODEL_PATH
from .traffic_env import QosRoutingEnv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--predictive", action="store_true", help="Train with predicted next-interval utilization features.")
    parser.add_argument("--failure-scenario", action="store_true", help="Inject an s2-s4 failure halfway through each training episode.")
    args = parser.parse_args()
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model_path = PREDICTIVE_MODEL_PATH if args.predictive else MODEL_PATH
    environment = make_vec_env(lambda: Monitor(QosRoutingEnv(seed=args.seed, predictive=args.predictive, failure_scenario=args.failure_scenario)), n_envs=4, seed=args.seed)
    model = PPO(
        "MlpPolicy",
        environment,
        learning_rate=3e-4,
        n_steps=512,
        batch_size=128,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.01,
        verbose=1,
        seed=args.seed,
    )
    model.learn(total_timesteps=args.timesteps)
    model.save(str(model_path))
    print(f"Saved {'predictive ' if args.predictive else ''}PPO model to {model_path}.zip")


if __name__ == "__main__":
    main()
