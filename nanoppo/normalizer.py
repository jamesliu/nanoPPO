import numpy as np


class Normalizer:
    def __init__(self, dim):
        # Mean, standard deviation, and count for each dimension
        self.n = np.zeros(dim, dtype=np.float32)
        self.mean = np.zeros(dim, dtype=np.float32)
        self.mean_diff = np.zeros(dim, dtype=np.float32)
        self.variance = np.zeros(dim, dtype=np.float32)

    def observe(self, x):
        """Update statistics"""
        self.n += 1.0
        last_mean = self.mean.copy()
        self.mean += (x - self.mean) / self.n
        self.mean_diff += (x - last_mean) * (x - self.mean)
        self.variance = (self.mean_diff / self.n).clip(min=1e-2)

    def normalize(self, inputs):
        """Normalize input using running mean and variance"""
        obs_std = np.sqrt(self.variance)
        v = (inputs - self.mean) / obs_std
        return v.astype(np.float32)

    def get_state(self):
        return {
            "n": self.n,
            "mean": self.mean,
            "mean_diff": self.mean_diff,
            "variance": self.variance,
        }

    def set_state(self, state):
        self.n = state["n"]
        self.mean = state["mean"]
        self.mean_diff = state["mean_diff"]
        self.variance = state["variance"]
