import numpy as np
import pytest
from nanoppo.reward_scaler import RewardScaler  # Change 'your_module' to the name of your module.


def test_initialization():
    scaler = RewardScaler()
    assert scaler.running_mean == 0
    assert scaler.running_sum_of_square_diffs == 0
    assert scaler.count == 0


def test_update_single_reward():
    scaler = RewardScaler()
    scaler.update([1.0])
    assert scaler.running_mean == 1.0
    assert scaler.running_sum_of_square_diffs == 0
    assert scaler.count == 1


def test_update_multiple_rewards():
    rewards = [1.0, 2.0, 3.0]
    scaler = RewardScaler()
    scaler.update(rewards)
    
    assert scaler.running_mean == 2.0
    assert scaler.count == 3


def test_scale_rewards():
    rewards = [1.0, 2.0, 3.0]
    scaler = RewardScaler()
    scaled_rewards = scaler.scale_rewards(rewards)
    
    assert len(scaled_rewards) == 3
    assert np.isclose(scaled_rewards[1], 0.0, atol=1e-8)  # The middle value (2.0) should be close to 0 after scaling


@pytest.mark.parametrize("rewards, expected_mean", [
    ([1.0, 2.0, 3.0], 2.0),
    ([0.0, 0.0, 0.0], 0.0),
    ([-1.0, 0.0, 1.0], 0.0),
])
def test_running_mean(rewards, expected_mean):
    scaler = RewardScaler()
    scaler.update(rewards)
    assert scaler.running_mean == expected_mean
