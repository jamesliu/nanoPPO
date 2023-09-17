class RewardShaper:
    """
    Base class for reward shaping. It provides a structure and can be subclassed
    for specific environments.
    """

    def __init__(self):
        pass

    def reshape(self, rewards, observations, next_observations=None):
        """
        This method should be overridden by subclasses to provide specific reward shaping logic.

        Parameters:
        - rewards (list or array): Original rewards from the environment.
        - observations (list or array): Corresponding observations for the rewards.

        Returns:
        - Reshaped rewards.
        """
        # By default, just return the original rewards.
        return rewards


class MountainCarRewardShaper(RewardShaper):
    """
    Reward shaping specifically designed for the MountainCar environment.
    """

    def __init__(self, position_weight=1.0, velocity_weight=1.0):
        """
        Initialize the reward shaper with given weights.

        Parameters:
        - position_weight (float): Weight for position-based reward shaping.
        - velocity_weight (float): Weight for velocity-based reward shaping.
        """
        super().__init__()
        self.position_weight = position_weight
        self.velocity_weight = velocity_weight

    def reshape(self, rewards, observations, next_observations=None):
        """
        Reward shaping for MountainCar based on position and velocity.

        Parameters:
        - rewards (list or array): Original rewards from the environment.
        - observations (list or array): Corresponding observations for the rewards.

        Returns:
        - Reshaped rewards.
        """
        # Extract position and velocity from observations.
        positions = [obs[0] for obs in observations]
        velocities = [obs[1] for obs in observations]

        # Calculate position-based and velocity-based rewards.
        position_rewards = [pos * self.position_weight for pos in positions]
        velocity_rewards = [abs(vel) * self.velocity_weight for vel in velocities]

        # Combine the original rewards with the shaped rewards.
        reshaped_rewards = [
            orig_reward + pos_reward + vel_reward
            for orig_reward, pos_reward, vel_reward in zip(
                rewards, position_rewards, velocity_rewards
            )
        ]

        return reshaped_rewards


import numpy as np


class MountainCarHeightRewardShaper(RewardShaper):
    """
    Reward shaping specifically designed for the MountainCar environment based on the car's height.
    """

    def __init__(self, height_weight=1.0):
        """
        Initialize the reward shaper with given weight.

        Parameters:
        - height_weight (float): Weight for height-based reward shaping.
        """
        super().__init__()
        self.height_weight = height_weight

    def height(self, position):
        """
        Calculate the height of the car based on its position.

        Parameters:
        - position (float): Current position of the car.

        Returns:
        - Height of the car.
        """
        # This formula is derived from the MountainCar environment's terrain shape.
        return np.sin(3 * position) * 0.45 + 0.55

    def reshape(self, rewards, observations, next_observations=None):
        """
        Reward shaping for MountainCar based on the car's height.

        Parameters:
        - rewards (list or array): Original rewards from the environment.
        - observations (list or array): Corresponding observations for the rewards.

        Returns:
        - Reshaped rewards.
        """
        # Calculate height-based rewards.
        height_rewards = [
            self.height(obs[0]) * self.height_weight for obs in observations
        ]

        # Combine the original rewards with the shaped rewards.
        reshaped_rewards = [
            orig_reward + height_reward
            for orig_reward, height_reward in zip(rewards, height_rewards)
        ]

        return reshaped_rewards


class MountainCarAdvancedRewardShaper(RewardShaper):
    """
    Advanced reward shaping for the MountainCar environment, combining multiple techniques.
    """

    def __init__(
        self,
        height_weight=1.0,
        velocity_weight=1.0,
        still_penalty=-0.5,
        still_threshold=0.01,
    ):
        """
        Initialize the reward shaper with given weights and parameters.

        Parameters:
        - height_weight (float): Weight for height-based reward shaping.
        - velocity_weight (float): Weight for velocity-based reward shaping.
        - still_penalty (float): Penalty for the car when it's nearly still.
        - still_threshold (float): Velocity threshold to consider the car as being 'still'.
        """
        super().__init__()
        self.height_weight = height_weight
        self.velocity_weight = velocity_weight
        self.still_penalty = still_penalty
        self.still_threshold = still_threshold

    def height(self, position):
        """
        Calculate the height of the car based on its position.

        Parameters:
        - position (float): Current position of the car.

        Returns:
        - Height of the car.
        """
        return np.sin(3 * position) * 0.45 + 0.55

    def reshape(self, rewards, observations, next_observations=None):
        """
        Advanced reward shaping for MountainCar based on multiple techniques.

        Parameters:
        - rewards (list or array): Original rewards from the environment.
        - observations (list or array): Corresponding observations for the rewards.

        Returns:
        - Reshaped rewards.
        """
        reshaped_rewards = []
        for reward, obs in zip(rewards, observations):
            position, velocity = obs

            # Height-based reward
            height_reward = self.height(position) * self.height_weight

            # Velocity-based reward
            velocity_reward = abs(velocity) * self.velocity_weight

            # Penalty for staying still
            penalty = self.still_penalty if abs(velocity) < self.still_threshold else 0

            # Combine the rewards
            total_reward = reward + height_reward + velocity_reward + penalty
            reshaped_rewards.append(total_reward)

        return reshaped_rewards


class MountainCarDirectionalRewardShaper(RewardShaper):
    """
    Reward shaping for MountainCar based on height, velocity direction, and distance from the goal.
    """

    def __init__(
        self, height_weight=1.0, velocity_direction_weight=1.0, distance_weight=1.0
    ):
        super().__init__()
        self.height_weight = height_weight
        self.velocity_direction_weight = velocity_direction_weight
        self.distance_weight = distance_weight

    def height(self, position):
        return np.sin(3 * position) * 0.45 + 0.55

    def reshape(self, rewards, observations, next_observations=None):
        reshaped_rewards = []

        for reward, obs in zip(rewards, observations):
            position, velocity = obs

            # Height-based reward
            height_reward = self.height(position) * self.height_weight

            # Velocity direction reward
            if (position < 0 and velocity > 0) or (position > 0 and velocity < 0):
                direction_reward = abs(velocity) * self.velocity_direction_weight
            else:
                direction_reward = 0

            # Distance from the goal reward
            goal_position = 0.5
            distance_reward = -abs(goal_position - position) * self.distance_weight

            # Combine the rewards
            total_reward = reward + height_reward + direction_reward + distance_reward
            reshaped_rewards.append(total_reward)

        return reshaped_rewards


import torch


class TDRewardShaper(RewardShaper):
    """
    Reward shaping for MountainCar based on Temporal Difference (TD) error.
    """

    def __init__(self, model, device, gamma=0.99):
        super().__init__()
        self.model = model  # The neural network model used by PPO to estimate values
        self.device = device
        self.gamma = gamma

    def td_error(self, reward, current_state, next_state):
        current_state = torch.from_numpy(current_state).float().to(self.device)
        current_value = self.model(current_state)
        next_state = torch.from_numpy(next_state).float().to(self.device)
        next_value = self.model(next_state)

        return reward + self.gamma * next_value - current_value

    def reshape(self, rewards, observations, next_observations):
        reshaped_rewards = []
        with torch.no_grad():
            for t in range(len(rewards)):
                reward = rewards[t]
                current_state = observations[t]
                next_state = next_observations[t]
                td = self.td_error(reward, current_state, next_state)
                r = (reward + td).cpu().numpy().item()
                reshaped_rewards.append(r)
        return reshaped_rewards
