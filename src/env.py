# src/env.py

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

class ReservingEnv(gym.Env):
    """
    Gymnasium environment for CVaR-constrained insurance reserving.
    State vector: [R_t, L_t, V_t, K_t, M_t, nu_t, level]
    Actions: 7 discrete reserve-adjustment ratios between -0.10 and +0.10
    """
    metadata = {"render.modes": []}

    def __init__(self, data: pd.DataFrame, curriculum: dict, buffer_size: int = 1024):
        super().__init__()
        self.df = data.reset_index(drop=True)
        self.curriculum = curriculum
        self.buffer_size = buffer_size

        # Discrete action space: 7 possible adjustments
        self.action_space = spaces.Discrete(7)
        # 7-dimensional continuous observation
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32
        )

        self.reset()

    def reset(self):
        self.t = 0
        self.R = 1.0          # normalized reserve
        self.nu = 0.0         # violation memory trace
        self.shortfalls = []  # buffer for CVaR estimation
        self.level = 0        # curriculum level
        return self._get_obs(), {}

    def _get_obs(self):
        row = self.df.iloc[self.t]
        Lt = row["IncurredLosses_norm"]
        Vt = row["LocalVolatility"]
        Kt = 1 - abs(self.R - Lt)
        mu, sigma = self.curriculum[self.level]
        Mt = np.random.normal(mu, sigma)
        obs = np.array(
            [self.R, Lt, Vt, Kt, Mt, self.nu, float(self.level)],
            dtype=np.float32
        )
        return obs

    def step(self, action):
        # Map discrete action to proportional delta
        deltas = np.linspace(-0.10, 0.10, 7)
        delta = deltas[action]
        self.R = max(0.0, self.R * (1 + delta))

        row = self.df.iloc[self.t]
        Lt = row["IncurredLosses_norm"]
        Vt = row["LocalVolatility"]
        shortfall = max(0.0, Lt - self.R)
        inefficiency = abs(self.R - Lt)
        Rreg = 0.4 + 0.2 * Vt
        violation = int(self.R < Rreg)

        # Update memory trace
        self.nu = 0.95 * self.nu + 0.05 * violation
        # Update shortfall buffer
        self.shortfalls.append(shortfall)
        if len(self.shortfalls) > self.buffer_size:
            self.shortfalls.pop(0)

        # CVaR estimation
        alpha = 0.90 + 0.05 * min(1.0, Vt)
        VaR = np.quantile(self.shortfalls, alpha)
        tail = [s for s in self.shortfalls if s >= VaR]
        cvar = float(np.mean(tail)) if tail else 0.0

        # Reward: composite negative cost
        reward = -(shortfall + cvar + inefficiency + violation)

        self.t += 1
        done = self.t >= len(self.df) - 1
        obs = (
            self._get_obs()
            if not done
            else np.zeros(self.observation_space.shape, dtype=np.float32)
        )

        return obs, reward, done, False, {}
