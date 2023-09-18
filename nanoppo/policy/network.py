import torch
import torch.nn as nn
import torch.nn.functional as F


# Define the weight initialization function
def init_weights(m, init_type="default"):
    if isinstance(m, nn.Linear):
        if init_type == "he":
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            nn.init.zeros_(m.bias)
        elif init_type == "xavier":
            nn.init.xavier_normal_(m.weight)
            nn.init.zeros_(m.bias)
        elif init_type == "default":
            # The default initialization in PyTorch for Linear layers
            # is almost like Xavier uniform. We'll simply skip any
            # custom initialization to rely on the default.
            pass
        else:
            raise ValueError(f"Initialization type {init_type} not recognized.")

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, n_latent_var, init_type="he"):
        super(PolicyNetwork, self).__init__()

        # Actor: outputs mean and log standard deviation
        self.action_mu = nn.Sequential(
            nn.Linear(state_dim, n_latent_var),
            nn.Tanh(),
            nn.Linear(n_latent_var, n_latent_var),
            nn.Tanh(),
            nn.Linear(n_latent_var, action_dim)
        )

        self.action_log_std = nn.Sequential(
            nn.Linear(state_dim, n_latent_var),
            nn.Tanh(),
            nn.Linear(n_latent_var, n_latent_var),
            nn.Tanh(),
            nn.Linear(n_latent_var, action_dim)
        )

        # Apply the initialization
        self.apply(lambda m: init_weights(m, init_type=init_type))

    def forward(self, x):
        mean = self.action_mu(x)
        log_std = self.action_log_std(x)
        
        # Compute std using the exponential of log_std
        std = torch.exp(log_std)
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
