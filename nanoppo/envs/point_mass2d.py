import numpy as np
import gym
from gym import spaces

class PointMass2DEnv(gym.Env):
    """
    A point mass on a 2D plane with continuous action space.
    The goal is to move the point mass to the target location (0, 0).
    """
    def __init__(self, max_episode_steps=200):
        super(PointMass2DEnv, self).__init__()
        self.max_episode_steps = max_episode_steps
        self.current_step = 0

        self.observation_space = spaces.Box(low=-10, high=10, shape=(2,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)

        self.state = None

    def reset(self):
        self.state = np.array([np.random.uniform(-10, 10), np.random.uniform(-10, 10)])
        self.current_step = 0
        return self.state, {}

    def step(self, action):
        self.current_step += 1
        self.state = self.state + action
        self.state = np.clip(self.state, -10, 10)

        reward = -np.linalg.norm(self.state)
        done = self.current_step >= self.max_episode_steps
        truncated = False
        return self.state, reward, done, truncated, {}

    def render(self, mode='human'):
        pass

    def close(self):
        pass
