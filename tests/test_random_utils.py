import pytest
import random
import numpy as np
import torch
from nanoppo.random_utils import set_seed, get_seed, sample_action, epsilon_greedy_action

def test_set_seed():
    set_seed(42)
    
    # Test python's random
    assert random.randint(1, 100) == 82
    
    # Test numpy's random
    assert np.random.randint(1, 100) == 52
    
    # Test PyTorch's random
    assert torch.randint(1, 100, (1,)).item() == 7

def test_get_seed():
    set_seed(42)
    state = get_seed()
    assert random.getstate() == state

"""
def test_sample_action():
    distribution = [0.1, 0.1, 0.8]
    samples = [sample_action(distribution) for _ in range(1000)]
    assert samples.count(2) > samples.count(1)
    assert samples.count(2) > samples.count(0)

def test_epsilon_greedy_action():
    values = [1, 2, 3]
    
    # Testing the greedy part
    action = epsilon_greedy_action(values, epsilon=0)
    assert action == 2  # should choose the action with the highest value
    
    # Testing the random part (statistically, for a large number of runs)
    actions = [epsilon_greedy_action(values, epsilon=1) for _ in range(1000)]
    # All actions should be roughly equally likely
    assert 300 < actions.count(0) < 700
    assert 300 < actions.count(1) < 700
    assert 300 < actions.count(2) < 700
"""
