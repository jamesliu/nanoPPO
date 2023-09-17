import pytest
import torch
from nanoppo.ppo_utils import compute_returns_and_advantages_without_gae

# Mock value function
def mock_value(state):
    # This can be any simple function for testing purposes.
    return torch.tensor([[state[0] * 2]])

# Test case
def test_compute_returns_and_advantages_without_gae():
    batch_rewards = [torch.tensor([1.0]), torch.tensor([2.0]), torch.tensor([3.0])]
    batch_states = [torch.tensor([1.0]), torch.tensor([2.0]), torch.tensor([3.0])]
    batch_next_states = [torch.tensor([2.0]), torch.tensor([3.0]), torch.tensor([4.0])]
    batch_dones = [torch.tensor([0]), torch.tensor([0]), torch.tensor([1])]
    gamma = 0.9

    returns, advs = compute_returns_and_advantages_without_gae(
        batch_rewards, batch_states, batch_next_states, batch_dones, mock_value, gamma
    )

    # Assert the outputs based on the mock value function and inputs
    # The exact values will depend on the mock_value function and inputs used.
    # For simplicity, we are checking if returns and advantages are computed without errors and have the expected length.
    assert len(returns) == 3
    assert len(advs) == 3

    # If you want more specific assertions, you can compute the expected returns and advantages manually
    # and then assert that they match the outputs. For now, we just check lengths.