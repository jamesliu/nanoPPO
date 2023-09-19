import numpy as np
import gym
from gym import spaces
"""
class PointMass2DEnv(gym.Env):
#    A point mass on a 2D plane with continuous action space.
#    The goal is to move the point mass to the target location (0, 0).

    def __init__(self, range=10, max_episode_steps=200):
        super(PointMass2DEnv, self).__init__()
        self.range = range
        self.max_episode_steps = max_episode_steps
        self.current_step = 0
        self.observation_space = spaces.Box(low=-self.range, high=self.range, shape=(2,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)

        self.state = None

    def reset(self):
        self.state = np.array([np.random.uniform(-self.range, self.range), np.random.uniform(-self.range, self.range)], dtype=np.float32)
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
"""

class PointMass2DEnv(gym.Env):
    """
    A revised version of the PointMass2DEnv.
    """
    def __init__(self, range=10, max_episode_steps=200, boundary_penalty=5.0):
        super(PointMass2DEnv, self).__init__()
        self.range = range
        self.max_episode_steps = max_episode_steps
        self.boundary_penalty = boundary_penalty
        self.current_step = 0
        self.observation_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        self.state = None

    def reset(self):
        self.state = np.array([np.random.uniform(-self.range, self.range), np.random.uniform(-self.range, self.range)], dtype=np.float32)
        self.current_step = 0
        return self._normalize_state(self.state), {}

    def step(self, action):
        self.current_step += 1
        
        # Apply action
        self.state = self.state + action * self.range * 0.1
        reward = -np.linalg.norm(self.state) / self.range
        
        # Check boundaries and apply penalty if needed
        if np.any(self.state < -self.range) or np.any(self.state > self.range):
            reward -= self.boundary_penalty
            self.state = np.clip(self.state, -self.range, self.range)

        done = self.current_step >= self.max_episode_steps
        return self._normalize_state(self.state), reward, done, False, {}

    def _normalize_state(self, state):
        return state / self.range

    def render(self, mode='human'):
        pass

    def close(self):
        pass

