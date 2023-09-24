import torch
import torch.nn as nn

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, n_latent_var, action_low_tensor, action_high_tensor, rescale=False):
        super(ActorCritic, self).__init__()
        self.action_low_tensor = action_low_tensor
        self.action_high_tensor = action_high_tensor
        self.rescale = rescale
        self.epsilon = 1e-5

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
            # Rescale action to be in the desired range
            if self.rescale:
                action = self.rescale_action(action)

        if compute_logprobs:
            logprobs = dist.log_prob(action).sum(axis=-1)
            return action, logprobs
        
        return action
    
    def rescale_action(self, action):
        """Rescale action from [-1, 1] to [action_low, action_high]"""
        # It introduces training instability because of tanh's gradient
        # Add a small epsilon to the output of tanh to ensure that you don't get values exactly equal to -1 or 1
        # It helps to delays the inf policy loss or nan problem, but not completely solve it
        # TODO: Disable rescale for now
        action = torch.clamp(torch.tanh(action), -1 + self.epsilon, 1 - self.epsilon)

        # Adjust the range for the epsilon-clamped tanh
        adjusted_low = (-1 + self.epsilon)  # This becomes the new "minimum" of your tanh output
        adjusted_high = (1 - self.epsilon)  # This becomes the new "maximum" of your tanh output
    
        # This ensures that the output of tanh is linearly mapped from [adjusted_low, adjusted_high] to [action_low, action_high]
        action_range = self.action_high_tensor - self.action_low_tensor
        action = self.action_low_tensor + (action - adjusted_low) / (adjusted_high - adjusted_low) * action_range
        return action

    def evaluate(self, state, action):
        if self.rescale:
            assert (action >= self.action_low_tensor).all() and (action <= self.action_high_tensor).all(), "Actions are not rescaled!"

        mu = self.action_mu(state)
        log_std = self.action_log_std(state)
        std = log_std.exp()
        
        dist = torch.distributions.Normal(mu, std)
        logprobs = dist.log_prob(action).sum(axis=-1)
        state_value = self.value_layer(state)
        
        return logprobs, torch.squeeze(state_value)

    def get_value(self, state):
        return self.value_layer(state)
