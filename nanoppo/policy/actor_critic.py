import torch
import torch.nn as nn

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, n_latent_var, action_low, action_high):
        super(ActorCritic, self).__init__()
        self.action_low = action_low
        self.action_high = action_high

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

        # Critic
        self.value_layer = nn.Sequential(
            nn.Linear(state_dim, n_latent_var),
            nn.Tanh(),
            nn.Linear(n_latent_var, n_latent_var),
            nn.Tanh(),
            nn.Linear(n_latent_var, 1)
        )

    def forward(self, state):
        mu = self.action_mu(state)
        log_std = self.action_log_std(state)
        std = log_std.exp()
        dist = torch.distributions.Normal(mu, std)
        return dist

    def act(self, state, action=None, compute_logprobs=True):
        mu = self.action_mu(state)
        log_std = self.action_log_std(state)
        std = log_std.exp()
        dist = torch.distributions.Normal(mu, std)
        if action is None:
            action = dist.sample()
            # Assuming env.action_space.low and env.action_space.high are numpy arrays
            action_low_tensor = torch.tensor(self.action_low, dtype=torch.float32)
            action_high_tensor = torch.tensor(self.action_high, dtype=torch.float32)
            #action = torch.clamp(action, action_low_tensor, action_high_tensor)

        if compute_logprobs:
            logprobs = dist.log_prob(action).sum(axis=-1)
            return action, logprobs
        
        return action

    def evaluate(self, state, action):
        mu = self.action_mu(state)
        log_std = self.action_log_std(state)
        std = log_std.exp()
        
        dist = torch.distributions.Normal(mu, std)
        logprobs = dist.log_prob(action).sum(axis=-1)
        state_value = self.value_layer(state)
        
        return logprobs, torch.squeeze(state_value)

    def get_value(self, state):
        return self.value_layer(state)
