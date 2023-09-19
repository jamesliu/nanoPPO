import random
from collections import deque
import torch
import numpy as np


class RolloutBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, log_prob, reward, next_state, done):
        self.buffer.append((state, action, log_prob, reward, next_state, done))

    def sample(self, batch_size, device, randomize=True):
        if randomize:
            states, actions, log_probs, rewards, next_states, dones = zip(
                *random.sample(self.buffer, batch_size)
            )
        else:
            states, actions, log_probs, rewards, next_states, dones = zip(
                *list(self.buffer)[-batch_size:]
            )
        states_t = torch.tensor(np.array(states, dtype=np.float32), device=device)
        actions_t = torch.tensor(np.array(actions, dtype=np.float32), device=device)
        log_probs_t = torch.tensor(np.array(log_probs, dtype=np.float32), device=device)
        rewards_t = torch.tensor(np.array(rewards, dtype=np.float32), device=device)
        next_states_t = torch.tensor(
            np.array(next_states, dtype=np.float32), device=device
        )
        dones_t = torch.tensor(np.array(dones, dtype=np.float32), device=device)
        return states_t, actions_t, log_probs_t, rewards_t, next_states_t, dones_t

    def __len__(self):
        return len(self.buffer)

    def clear(self):
        self.buffer.clear()
