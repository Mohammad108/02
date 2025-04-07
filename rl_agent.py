from stable_baselines3 import DQN
import os

class RLAgent:
    def __init__(self):
        self.model = None

    def train(self, env):
        self.model = DQN("MlpPolicy", env, verbose=1)
        self.model.learn(total_timesteps=10000)
        self.model.save("models/rl_model")

    def load(self, path):
        if os.path.exists(path):
            self.model = DQN.load(path)