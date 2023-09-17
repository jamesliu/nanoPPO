# random_utils.py

import random
import numpy as np
import torch
import time


def set_seed(seed=None, use_torch=True):
    """
    Set the seed for all random number generators.
    If no seed is provided, sets a random seed based on current time.

    Parameters:
    - seed (int): The seed to set. If None, will set a random seed.
    - use_torch (bool): If True, will also set the seed for PyTorch.
    """
    if seed is None:
        seed = int((time.time() * 1e6) % 1e6)
    random.seed(seed)
    np.random.seed(seed)
    if use_torch:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_seed():
    """
    Retrieve the current seed value for the Python random module.
    Note: This won't fetch seeds for numpy or torch.
    """
    return random.getstate()


def sample_action(prob_distribution):
    """
    Sample an action from a given probability distribution.

    Parameters:
    - prob_distribution (list or numpy array): The probability distribution over actions.

    Returns:
    - The sampled action index.
    """
    return np.random.choice(len(prob_distribution), p=prob_distribution)


def epsilon_greedy_action(values, epsilon=0.1):
    """
    Choose an action using epsilon-greedy strategy.

    Parameters:
    - values (list or numpy array): Estimated values or Q-values for each action.
    - epsilon (float): The probability of choosing a random action.

    Returns:
    - The chosen action index.
    """
    if random.random() < epsilon:
        return random.randint(0, len(values) - 1)
    else:
        return np.argmax(values)
