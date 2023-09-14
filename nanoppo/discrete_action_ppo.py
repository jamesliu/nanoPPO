import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_actions):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, n_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x), dim=-1)


class ValueNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class PPO:
    def __init__(
        self, state_dim, action_dim, hidden_dim=64, lr=0.001, gamma=0.99, epsilon=0.2
    ):
        self.policy = PolicyNetwork(state_dim, hidden_dim, action_dim)
        self.value = ValueNetwork(state_dim, hidden_dim)
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.value_optimizer = optim.Adam(self.value.parameters(), lr=lr)
        self.gamma = gamma
        self.epsilon = epsilon

    def compute_advantage(self, rewards, values, next_value, done):
        advantages = torch.zeros_like(rewards)
        returns = next_value
        for t in reversed(range(len(rewards))):
            returns = rewards[t] + self.gamma * returns * (1 - done[t])
            advantages[t] = returns - values[t]
        return advantages

    def update(self, states, actions, rewards, next_states, done, log_probs):
        states = torch.tensor(np.array(states), dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32)
        done = torch.tensor(done, dtype=torch.float32)
        old_log_probs = torch.tensor(log_probs, dtype=torch.float32)

        # Compute advantage
        values = self.value(states).squeeze()
        if done[-1]:
            next_value = 0
        else:
            next_value = self.value(next_states[-1]).item()
        advantages = self.compute_advantage(
            rewards, values, next_value, done
        ).detach()  # Detach advantages

        # Compute value loss and update value network
        targets = rewards + self.gamma * self.value(next_states).squeeze() * (1 - done)
        value_loss = F.mse_loss(values, targets.detach())
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

        # Compute new log probabilities
        new_log_probs = (
            self.policy(states).gather(1, actions.unsqueeze(1)).log().squeeze()
        )

        # Compute probability ratio
        ratio = torch.exp(new_log_probs - old_log_probs)

        # Compute surrogate loss
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        # Update policy network
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
