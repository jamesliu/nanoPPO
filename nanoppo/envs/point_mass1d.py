import gym
from gym import spaces
import numpy as np

class PointMass1DEnv(gym.Env):
    def __init__(self, action_range=1.0, position_limit=10.0, damping_factor=0.9, max_episode_steps=200):
        super(PointMass1DEnv, self).__init__()
        self.action_range = action_range
        # Action space: Force [-1.0, 1.0]
        self.action_space = spaces.Box(low=-action_range, high=action_range, shape=(1,), dtype=float)
        
        # State space: Position and Velocity
        self.observation_space = spaces.Box(low=-float(position_limit), high=float(position_limit), shape=(2,), dtype=float)
        
        # Parameters
        self.max_steps = max_episode_steps
        self.current_step = 0
        self.damping_factor = damping_factor

    def reset(self):
        self.state = np.array([0.5 * (2 * np.random.rand() - 1), 0.0], dtype=np.float32)  # random initial position, zero velocity
        self.current_step = 0
        return self.state, {}

    def step(self, action):
        position, velocity = self.state

        # Update velocity and position based on the force (action)
        # Update velocity and position based on the force (action)
        velocity = self.damping_factor * (velocity + action[0])  # Introducing damping
        position += velocity
        
        # Clamp position to bounds
        position = np.clip(position, self.observation_space.low[0], self.observation_space.high[0])

        self.state = np.array([position, velocity], dtype=np.float32)

        # Calculate reward
        reward = 0.5 -abs(position)  # reward is negative absolute distance to origin
        
        self.current_step += 1
        done = self.current_step >= self.max_steps
        truncated = False
        return self.state, reward, done, truncated, {}

    def render(self, mode='console'):
        if mode == 'console':
            print(f"Position: {self.state[0]}, Velocity: {self.state[1]}")

    def close(self):
        pass