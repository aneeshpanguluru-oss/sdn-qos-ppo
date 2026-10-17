"""Train the QoS-aware PPO policy."""

from __future__ import annotations

import argparse

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor

from .config import MODEL_PATH
from .traffic_env import QosRoutingEnv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    environment = make_vec_env(lambda: Monitor(QosRoutingEnv(seed=args.seed)), n_envs=4, seed=args.seed)
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
    model.save(str(MODEL_PATH))
    print(f"Saved PPO model to {MODEL_PATH}.zip")


if __name__ == "__main__":
    main()
