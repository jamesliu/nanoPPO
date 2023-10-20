import torch
import torch.nn as nn

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, n_latent_var, action_low_tensor, action_high_tensor, rescale=False, debug=False):
        super(ActorCritic, self).__init__()
        self.action_low_tensor = action_low_tensor
        self.action_high_tensor = action_high_tensor
        self.rescale = rescale
        self.debug = False
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

        if self.debug:
            # Check for NaN values immediately after computation
            if torch.isnan(mu).any() or torch.isnan(log_std).any():
                print("NaN detected in 'mu' or 'log_std' during forward pass.")
                breakpoint()
            
        # Avoid policy loss NAN or entropy Loss INF
        # std = log_std.exp() May cause NaN values in self.policy and Inf values in entropy_loss due to action_pro approaching zero.
        # The entropy of a policy in certain contexts is calculated using the probabilities of the actions, which the policy might take. 
        # If these probabilities approach zero, the log of near-zero probabilities can become negative infinity, 
        # and hence the entropy becomes positive infinity (since entropy is calculated as the expected value of -log(prob)). This is usually the cause of entropy_loss becoming inf.
        # Clamping the standard deviation to avoid extreme values
        std = torch.clamp(log_std.exp(), min=self.epsilon, max=1e2)  # you can adjust the max value based on your context
        
        dist = torch.distributions.Normal(mu, std)
        return dist

    def act(self, state, action=None, compute_logprobs=True):
        if self.debug:
            # Check if the state contains NaN values
            if torch.isnan(state).any():
                print("NaN detected in the state input of the 'act' method.")
                breakpoint()

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
            # policy_loss:NAN, entropy_loss:INF
            # Ensure that action values are within a valid range before computing log_prob
            # action = torch.clamp(action, self.action_low_tensor + self.epsilon, self.action_high_tensor - self.epsilon)
            logprobs = dist.log_prob(action).sum(axis=-1)

            if self.debug and (torch.isnan(logprobs).any() or torch.isinf(logprobs).any()):
                print("Anomaly detected in 'logprobs'")
                breakpoint()  # Stopping here if debugging is enabled
 
            # Sanitize logprobs to replace -inf with large negative numbers
            clean_logprobs = torch.where(
                torch.isinf(logprobs),
                torch.full_like(logprobs, -1e7),  # Replace -inf with -1e7
                logprobs
            )
            return action, clean_logprobs
        
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

    # Define the debug function
    def debug_this(self, model):
        print("Debugging Information:")
        # Check and print the weights and biases
        print("\nWeights and Biases in each layer:")
        for name, param in model.named_parameters():
            print(name, param.data)
        
        # Pass the input through each layer individually and print the outputs
        x = state
        for i, layer in enumerate(model):
            x = layer(x)
            print(f"\nOutput after layer {i} ({layer.__class__.__name__}):")
            print(x)
            # If you find NaN at this stage, you can break the loop for efficiency
            if torch.isnan(x).any():
                print(f"Stopping early due to NaN at layer {i}")
                break

    def check_for_nan_gradients(self):
        if self.debug:
            for name, param in self.named_parameters():
                if param.grad is not None and torch.isnan(param.grad).any():
                    print(f"NaN detected in the gradients. Parameter: {name}")
                    # Here you can either call your debug function or raise an error
                    breakpoint()

    def evaluate(self, state, action):
        if self.rescale:
            assert (action >= self.action_low_tensor).all() and (action <= self.action_high_tensor).all(), "Actions are not rescaled!"

        mu = self.action_mu(state)
        if self.debug:
            # Check if 'mu' contains any NaN values and call the debug function if it does
            if torch.isnan(mu).any():
                print("NaN detected in output. Starting debug sequence...\n")
                self.debug_this(self.action_mu)
                breakpoint()
        log_std = self.action_log_std(state)
        std = log_std.exp()
        
        dist = torch.distributions.Normal(mu, std)
        logprobs = dist.log_prob(action).sum(axis=-1)

        if self.debug:
            # Before calculating the ratio, add these checks
            if (torch.isnan(logprobs).any() or torch.isinf(logprobs).any()):
                print("Anomaly detected in 'logprobs'")
                breakpoint()  # Insert your handling here

        # Sanitize logprobs to replace -inf with large negative numbers
        clean_logprobs = torch.where(
            torch.isinf(logprobs),
            torch.full_like(logprobs, -1e7),  # Replace -inf with -1e7
            logprobs
        )
        state_value = self.value_layer(state)
        return clean_logprobs, torch.squeeze(state_value)

    def get_value(self, state):
        return self.value_layer(state)
