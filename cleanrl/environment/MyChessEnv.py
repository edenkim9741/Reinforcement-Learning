import gymnasium as gym
import numpy as np

class MyChessEnv(gym.Env):
    def __init__(self):
        super().__init__()

        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(8, 8, 12), dtype=np.float32)

        self.action_space = gym.spaces.Discrete(4672)

    def reset(self, seed=None, options=None):
        return observation, info

    def step(self, action):
        return observation, reward, terminated, truncated, info

