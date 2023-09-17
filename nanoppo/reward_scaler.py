import numpy as np


class RewardScaler:
    """
    This class is designed to normalize the rewards using Welford's online algorithm
    for computing running mean and variance. Normalizing rewards can be beneficial
    for training as it ensures that the scale of the rewards is roughly consistent,
    which can improve the stability and convergence speed of the learning algorithm.

    The formula for Welford's algorithm is defined as:

    M_{k} = M_{k-1} + (x_k - M_{k-1})/k
    S_{k} = S_{k-1} + (x_k - M_{k-1})(x_k - M_{k})

    Where:
    M_k: Running mean after observing k samples.
    S_k: Running sum of squares of differences from the current mean after observing k samples.
    x_k: k-th sample.

    The variance is then computed as:
    variance = S_k / (k-1)
    (for k > 1).
    """

    def __init__(self):
        self.running_mean = 0  # Running mean of rewards.
        self.running_sum_of_square_diffs = (
            0  # Running sum of squares of differences from the current mean.
        )
        self.count = 0  # Count of observed samples.

    def update(self, rewards):
        """
        Update the running mean and variance using Welford's online algorithm.

        Parameters:
        - rewards (list or array): Batch of rewards from recent episodes.
        """
        batch_mean = np.mean(rewards)
        batch_count = len(rewards)
        new_count = self.count + batch_count

        delta = batch_mean - self.running_mean
        new_mean = self.running_mean + delta * batch_count / new_count

        for reward in rewards:
            delta = reward - new_mean
            delta2 = reward - self.running_mean
            self.running_sum_of_square_diffs += delta * delta2

        self.running_mean = new_mean
        self.count = new_count

    def scale_rewards(self, rewards):
        """
        Scale rewards using the running statistics and update the statistics.

        Parameters:
        - rewards (list or array): Batch of rewards from recent episodes.

        Returns:
        - Normalized rewards.
        """
        self.update(rewards)
        variance = (
            self.running_sum_of_square_diffs / (self.count - 1)
            if self.count > 1
            else 1.0
        )
        std = np.sqrt(variance)

        return [(reward - self.running_mean) / (std + 1e-8) for reward in rewards]
