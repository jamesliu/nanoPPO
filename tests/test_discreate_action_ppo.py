import numpy as np
import torch
from nanoppo.discrete_action_ppo import PPO


def test_compute_advantage():
    state_dim = 4
    action_dim = 2
    ppo = PPO(state_dim, action_dim)

    rewards = torch.tensor([1, 2, 3], dtype=torch.float32)
    values = torch.tensor([1, 1, 1], dtype=torch.float32)
    next_value = 0
    done = torch.tensor([0, 0, 1], dtype=torch.float32)

    advantages = ppo.compute_advantage(rewards, values, next_value, done)
    expected_advantages = torch.tensor(
        [4.9203, 3.9700, 2.000]
    ) 
    assert torch.isclose(advantages, expected_advantages, atol=1e-4).all()