import gym
import numpy as np
from gym import spaces

class TradingEnv(gym.Env):
    def __init__(self, data):
        super(TradingEnv, self).__init__()
        self.data = data.reset_index(drop=True)
        self.current_step = 0
        self.balance = 1000
        self.position = 0

        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=0, high=1, shape=(6,), dtype=np.float32)

    def reset(self):
        self.current_step = 0
        self.balance = 1000
        self.position = 0
        return self._next_observation()

    def _next_observation(self):
        row = self.data.iloc[self.current_step]
        return np.array([
            row["open"], row["high"], row["low"], row["close"],
            row["volume"], self.position
        ])

    def step(self, action):
        prev_price = self.data.iloc[self.current_step]["close"]
        self.current_step += 1
        done = self.current_step >= len(self.data) - 1
        price = self.data.iloc[self.current_step]["close"]

        reward = 0
        if action == 0:  # buy
            self.position = price
        elif action == 1 and self.position > 0:  # sell
            profit = price - self.position
            reward = profit
            self.balance += profit
            self.position = 0

        info = {"balance": self.balance}
        return self._next_observation(), reward, done, info