import argparse
from stable_baselines3 import PPO
from src.env import ReservingEnv
from src.utils import load_and_preprocess

def main(dataset):
    df = load_and_preprocess(f"data/{dataset}_pos.csv")
    curriculum = {
        0: (1.0, 0.1),
        1: (1.2, 0.2),
        2: (1.5, 0.3),
        3: (1.8, 0.4),
    }
    env = ReservingEnv(df, curriculum)
    model = PPO(
        "MlpPolicy", env,
        batch_size=2048,
        n_epochs=10,
        learning_rate=3e-4,
        verbose=1
    )
    model.learn(total_timesteps=int(1e6))
    model.save("models/ppo_cvar_reserving")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["wkcomp", "othliab"], required=True)
    args = parser.parse_args()
    main(args.dataset)
