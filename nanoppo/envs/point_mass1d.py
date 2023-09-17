import gym
from gym import spaces
import numpy as np

class PointMass1DEnv(gym.Env):
    def __init__(self):
        super(PointMass1DEnv, self).__init__()
        
        # Action space: Force [-1.0, 1.0]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=float)
        
        # State space: Position and Velocity
        self.observation_space = spaces.Box(low=-float('inf'), high=float('inf'), shape=(2,), dtype=float)
        
        # Parameters
        self.max_steps = 200
        self.current_step = 0

    def reset(self):
        self.state = [0.5 * (2 * np.random.rand() - 1), 0.0]  # random initial position, zero velocity
        self.current_step = 0
        return self.state, {}

    def step(self, action):
        position, velocity = self.state

        # Update velocity and position based on the force (action)
        velocity += action[0]
        position += velocity

        self.state = [position, velocity]

        # Calculate reward
        reward = -abs(position)  # reward is negative absolute distance to origin
        
        self.current_step += 1
        done = self.current_step >= self.max_steps
        truncated = False
        return self.state, reward, done, truncated, {}

    def render(self, mode='console'):
        if mode == 'console':
            print(f"Position: {self.state[0]}, Velocity: {self.state[1]}")

    def close(self):
        pass
