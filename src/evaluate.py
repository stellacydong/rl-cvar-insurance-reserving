import argparse
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from src.env import ReservingEnv
from src.utils import load_and_preprocess, compute_metrics

def run_eval(model_path, dataset, fixed_shocks=None):
    df = load_and_preprocess(f"data/{dataset}_pos.csv")
    curriculum = {0:(1.0,0.1),1:(1.2,0.2),2:(1.5,0.3),3:(1.8,0.4)}
    model = PPO.load(model_path)

    # stochastic test
    env = ReservingEnv(df, curriculum)
    rollouts = []
    obs, _ = env.reset()
    for _ in range(len(df)-1):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _, _ = env.step(action)
        rollouts.append({
            "R": env.R,
            "L": df.loc[env.t-1, "IncurredLosses_norm"],
            "violation": int(env.R < 0.4 + 0.2 * df.loc[env.t-1, "LocalVolatility"])
        })
        if done: break
    print("Stochastic metrics:", compute_metrics(rollouts))

    # fixed‐shock tests
    if fixed_shocks:
        for shock in fixed_shocks:
            curriculum_fs = {0:(shock,0.0)}
            env_fs = ReservingEnv(df, curriculum_fs)
            roll = []
            obs, _ = env_fs.reset()
            while True:
                a, _ = model.predict(obs, deterministic=True)
                obs, _, done, _, _ = env_fs.step(a)
                roll.append({
                    "R": env_fs.R,
                    "L": df.loc[env_fs.t-1, "IncurredLosses_norm"],
                    "violation": int(env_fs.R < 0.4 + 0.2 * df.loc[env_fs.t-1, "LocalVolatility"])
                })
                if done: break
            print(f"Fixed‐shock={shock}:", compute_metrics(roll))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", choices=["wkcomp", "othliab"], required=True)
    parser.add_argument("--shocks", nargs="+", type=float, default=[0.8,1.0,1.5,2.0])
    args = parser.parse_args()
    run_eval(args.model, args.dataset, args.shocks)
