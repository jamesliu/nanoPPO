import torch
from typing import List


def compute_gae(
    next_value: float,
    rewards: List[torch.Tensor],  # List of float32 tensors
    masks: List[torch.Tensor],  # List of float32 tensors
    values: List[torch.Tensor],  # List of float32 tensors
    gamma: float,
    tau: float,
):
    values = values + [next_value]
    gae = 0
    returns = []
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
        gae = delta + gamma * tau * masks[step] * gae
        returns.insert(0, gae + values[step])
    return returns


def compute_returns_and_advantages_without_gae(
    rewards, states, next_states, dones, value, gamma=0.99
):
    """
    Compute returns and advantages without using Generalized Advantage Estimation (GAE).

    Parameters:
    - rewards: list of rewards for each timestep
    - states: list of states for each timestep
    - next_states: list of next states for each timestep
    - dones: list of done flags for each timestep
    - value: function to compute the value of a state
    - gamma: discount factor

    Returns:
    - returns: list of computed returns for each timestep
    - advs: list of computed advantages for each timestep
    """
    returns = []
    advs = []

    g = 0
    with torch.no_grad():
        for r, state, next_state, done in zip(
            reversed(rewards),
            reversed(states),
            reversed(next_states),
            reversed(dones),
        ):
            mask = 1 - done.item()
            next_value = value(next_state).item()
            next_value = next_value * mask
            g = r + gamma * next_value
            returns.insert(0, g)

            value_curr_state = value(state).item()
            delta = g - value_curr_state
            advs.insert(0, delta)

    return returns, advs

def get_grad_norm(parameters):
    """
    Compute the 2-norm of gradients for the provided parameters.

    Args:
    - parameters (Iterable[torch.Tensor]): Network parameters.

    Returns:
    - float: Gradient norm.
    """
    total_norm = 0.0
    for param in parameters:
        if param.grad is not None:
            param_norm = param.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm
