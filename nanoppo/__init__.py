import gym
from .envs.point_mass1d import PointMass1DEnv
from .envs.point_mass2d import PointMass2DEnv

__version__ = "0.15.0"

gym.register("PointMass1D-v0", entry_point=PointMass1DEnv, max_episode_steps=200)
gym.register("PointMass2D-v0", entry_point=PointMass2DEnv, max_episode_steps=200)


