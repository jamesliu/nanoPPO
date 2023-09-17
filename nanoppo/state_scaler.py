from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
    QuantileTransformer,
)
import numpy as np


class StateScaler:
    def __init__(self, env, scale_type, sample_size=10000):
        self.env = env
        self.scale_type = scale_type

        if self.scale_type in ["standard", "minmax", "robust", "quantile"]:
            self.scaler = self._init_scaler(env, sample_size, scale_type)

    def _init_scaler(self, env, sample_size, scale_type):
        samples = [env.observation_space.sample() for _ in range(sample_size)]
        state_space_samples = np.array(
            [
                sample["obs"] if isinstance(sample, dict) else sample
                for sample in samples
            ]
        )

        if scale_type == "standard":
            scaler = StandardScaler()
        elif scale_type == "minmax":
            scaler = MinMaxScaler()
        elif scale_type == "robust":
            scaler = RobustScaler()
        elif scale_type == "quantile":
            scaler = QuantileTransformer()
        else:
            raise ValueError(f"Unknown scale type: {scale_type}")
        scaler.fit(state_space_samples)
        return scaler

    def scale_state(self, state):
        if self.scale_type in ["standard", "minmax", "robust", "quantile"]:
            scaled = self.scaler.transform([state])
            r = scaled[0]  # Return a 1D array instead of 2D
        elif self.scale_type == "env":
            # -1 to 1 scaling
            r = (
                2
                * (state - self.env.observation_space.low)
                / (self.env.observation_space.high - self.env.observation_space.low)
                - 1
            )
        else:
            raise ValueError(f"Unknown scale type: {self.scale_type}")
        return r.astype(np.float32)
