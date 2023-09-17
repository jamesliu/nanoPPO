import torch
import torch.nn as nn
import torch.nn.functional as F


# Define the weight initialization function
def init_weights(m, init_type="he"):
    if isinstance(m, nn.Linear):
        if init_type == "he":
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
        elif init_type == "xavier":
            nn.init.xavier_normal_(m.weight)
        else:
            raise ValueError(f"Initialization type {init_type} not recognized.")
        nn.init.zeros_(m.bias)


class PolicyNetwork(nn.Module):
    def __init__(
        self, state_size, action_size, init_type, min_log_std=-10, max_log_std=0.5
    ):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)
        # self.log_std = nn.Parameter(torch.zeros(action_size))
        self.log_std = nn.Parameter(torch.full((action_size,), -0.511))
        self.min_log_std = min_log_std
        self.max_log_std = max_log_std

        # Apply the initialization
        self.apply(lambda m: init_weights(m, init_type=init_type))

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = torch.tanh(self.fc3(x))

        # Clamp log_std to ensure it's within a specific range
        clamped_log_std = self.log_std.clamp(self.min_log_std, self.max_log_std)

        # Compute std using the exponential of log_std
        std = torch.exp(clamped_log_std)
        dist = torch.distributions.normal.Normal(mean, std)
        return dist


class ValueNetwork(nn.Module):
    def __init__(self, state_size, hidden_size, init_type):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)

        # Apply the initialization
        self.apply(lambda m: init_weights(m, init_type=init_type))

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
