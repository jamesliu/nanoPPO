import numpy as np
from nanoppo.ppo_utils import compute_gae

def test_compute_gae():
    # Define some dummy data
    next_value = 5
    rewards = [1, 2, 3]
    masks = [1, 1, 0]  # Last entry is episode termination
    values = [4, 5, 6]
    gamma = 0.99
    tau = 0.95

    # Compute GAE
    computed_returns = compute_gae(next_value, rewards, masks, values, gamma, tau)

    # Expected returns computed manually
    # Note: This is a simple example, and the expected values are calculated based on the GAE formula.
    delta_1 = 1 + gamma * 5 - 4
    delta_2 = 2 + gamma * 6 - 5
    delta_3 = 3 + gamma * next_value * 0 - 6
    gae_3 = delta_3
    gae_2 = delta_2 + gamma * tau * gae_3
    gae_1 = delta_1 + gamma * tau * gae_2
    expected_returns = [gae_1 + 4, gae_2 + 5, gae_3 + 6]

    # Assert that the computed GAE returns are close to the expected returns
    assert np.allclose(
        computed_returns, expected_returns
    ), f"Expected {expected_returns}, but got {computed_returns}"
