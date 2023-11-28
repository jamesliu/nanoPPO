import torch
import torch.nn as nn
from torch.nn import MultiheadAttention
from nanoppo.sinusoidal_positional_encoding import SinusoidalPositionalEncoding

class ActorCriticCausalAttention(nn.Module):
    def __init__(self, state_dim, action_dim, nhead, action_low_tensor, action_high_tensor, device, rescale=False, debug=False):
        super(ActorCriticCausalAttention, self).__init__()
        self.action_low_tensor = action_low_tensor
        self.action_high_tensor = action_high_tensor
        self.rescale = rescale
        self.debug = debug
        self.epsilon = 1e-5
        self.positional_encoding = SinusoidalPositionalEncoding(d_model = state_dim, device=device)
        
        # Actor (Mu)
        self.action_mu = nn.Sequential(
            MultiheadAttention(state_dim, nhead, batch_first=True),
            nn.Linear(state_dim, nhead),
            #nn.Tanh(),
            nn.ReLU(),
            nn.Linear(nhead, action_dim)
        )
        
        # Actor (Log Std)
        self.action_log_std = nn.Sequential(
            MultiheadAttention(state_dim, nhead, batch_first=True),
            nn.Linear(state_dim, nhead),
            #nn.Tanh(),
            nn.ReLU(),
            nn.Linear(nhead, action_dim)
        )
        
        # Critic
        self.value_layer = nn.Sequential(
            MultiheadAttention(state_dim, nhead, batch_first=True),
            nn.Linear(state_dim, nhead),
            #nn.Tanh(),
            nn.ReLU(),
            nn.Linear(nhead, 1)
        )

    def forward(self, state): 
        # Validate state inputs at the very beginning of the method
        if self.debug:
            if torch.any(torch.isnan(state)):
                print("NaN detected in state input during forward pass.")
                breakpoint()  # Starts a pdb session so you can investigate.

        # Ensure state has at least 3 dimensions
        if len(state.shape) == 1:
            state = state.unsqueeze(0).unsqueeze(0)
        if len(state.shape) == 2:
            state = state.unsqueeze(0)
        # Create causal mask
        length = state.size(1)
        mask = torch.triu(torch.ones(length, length), diagonal=1).bool().to(state.device)
        
        state = self.positional_encoding(state)
        # Actor (Mu)
        attn_output_mu, _ = self.action_mu[0](state, state, state, attn_mask=mask)
        mu = self.action_mu[1:](attn_output_mu)

        # Validate outputs from the attention mechanism
        if self.debug:
            if torch.any(torch.isnan(attn_output_mu)):
                print("NaN detected in attn_output_mu during forward pass.")
                breakpoint()

        # Actor (Log Std)
        attn_output_std, _ = self.action_log_std[0](state, state, state, attn_mask=mask)

        # Validate outputs from the attention mechanism
        if self.debug:
            if torch.any(torch.isnan(attn_output_std)):
                print("NaN detected in attn_output_std during forward pass.")
                breakpoint()

        log_std = self.action_log_std[1:](attn_output_std)

        if self.debug:
            # Check for NaN values immediately after computation
            if torch.isnan(mu).any() or torch.isnan(log_std).any():
                print("NaN detected in 'mu' or 'log_std' during forward pass.")
                breakpoint() 

        std = torch.clamp(log_std.exp(), min=self.epsilon, max=1e2)

        # Validate the 'std' tensor after computation
        if self.debug:
            if torch.any(torch.isnan(std)):
                print("NaN detected in 'std' tensor during forward pass.")
                breakpoint()

        dist = torch.distributions.Normal(mu, std)
        return dist

    def act(self, state, action=None, compute_logprobs=True):
        # Ensure state has at least 3 dimensions
        if len(state.shape) == 1:
            state = state.unsqueeze(0).unsqueeze(0)
        if len(state.shape) == 2:
            state = state.unsqueeze(0)
        if self.debug:
            # Check if the state contains NaN values
            if torch.isnan(state).any():
                print("NaN detected in the state input of the 'act' method.")
                breakpoint()
        # Create causal mask for attention
        length = state.size(1)
        mask = torch.triu(torch.ones(length, length), diagonal=1).bool().to(state.device)

        state = self.positional_encoding(state)
        # Actor (Mu)
        attn_output_mu, _ = self.action_mu[0](state, state, state, attn_mask=mask)
        mu = self.action_mu[1:](attn_output_mu)
        mu = mu[:, -1, :]  # Take the last sequence element

        # Validate 'mu' after its computation
        if self.debug:
            if torch.any(torch.isnan(mu)):
                print("NaN detected in 'mu' tensor during act method.")
                breakpoint()
        
        # Actor (Log Std)
        attn_output_std, _ = self.action_log_std[0](state, state, state, attn_mask=mask)
        log_std = self.action_log_std[1:](attn_output_std)
        log_std = log_std[:, -1, :]  # Take the last sequence element

        # Validate 'log_std' after its computation
        if self.debug:
            if torch.any(torch.isnan(log_std)):
                print("NaN detected in 'log_std' tensor during act method.")
                breakpoint()

        std = torch.clamp(log_std.exp(), min=self.epsilon, max=1e2)
        if action is None:
            dist = torch.distributions.Normal(mu, std)
            action = dist.sample()
            action = torch.clamp(action, min=-10, max=10)
            # Rescale action to be in the desired range
            if self.rescale:
                action = self.rescale_action(action)

        if compute_logprobs:
            #dist = torch.distributions.Normal(mu, std)
            #logprobs = dist.log_prob(action).sum(axis=-1)
            # --- Start of modified code for calculating log probabilities ---
            variance = std.pow(2)
            # Probability density function calculation for normal distribution
            probs = torch.exp(-((action - mu).pow(2)) / (2 * variance)) / torch.sqrt(2 * torch.tensor(torch.pi).to(action.device) * variance)
        
            # Add a small epsilon to the probabilities to prevent taking log of zero
            safe_probs = probs + self.epsilon  # where self.epsilon is a small constant, e.g., 1e-10
        
            # Now, calculate log probabilities
            logprobs = torch.log(safe_probs).sum(axis=-1)  # You may not need to sum, depending on your specific use case

            if self.debug and (torch.isnan(logprobs).any() or torch.isinf(logprobs).any()):
                print("Anomaly detected in 'logprobs'")
                breakpoint()  # Stopping here if debugging is enabled

            # Sanitize logprobs to replace -inf with large negative numbers
            clean_logprobs = torch.where(
                torch.isinf(logprobs),
                torch.full_like(logprobs, -1e7),
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
    def debug_this(self, model, state):
        print("Debugging Information:")
        # Check and print the weights and biases
        print("\nWeights and Biases in each layer:")
        for name, param in model.named_parameters():
            print(name, param.data)
        
        state = self.positional_encoding(state)
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

    def get_value(self, state):
        # Ensure state has at least 3 dimensions
        if len(state.shape) == 1:
            state = state.unsqueeze(0).unsqueeze(0)
        if len(state.shape) == 2:
            state = state.unsqueeze(0)
        # Create causal mask for attention
        length = state.size(1)
        mask = torch.triu(torch.ones(length, length), diagonal=1).bool().to(state.device)
        
        state = self.positional_encoding(state)
        # Value Layer
        attn_output_value, _ = self.value_layer[0](state, state, state, attn_mask=mask)
        return self.value_layer[1:](attn_output_value)[:, -1, :]

    def evaluate(self, state, action):
        # Ensure state has at least 3 dimensions
        if len(state.shape) == 1:
            state = state.unsqueeze(0).unsqueeze(0)
        if len(state.shape) == 2:
            state = state.unsqueeze(0)
        # Create causal mask for attention
        length = state.size(1)
        mask = torch.triu(torch.ones(length, length), diagonal=1).bool().to(state.device)

        state = self.positional_encoding(state)
        # Actor (Mu)
        attn_output_mu, _ = self.action_mu[0](state, state, state, attn_mask=mask)
        mu = self.action_mu[1:](attn_output_mu)
        mu = mu[:, -1, :]  # Take the last sequence element
        if self.debug:
            # Check if 'mu' contains any NaN values and call the debug function if it does
            if torch.isnan(mu).any():
                print("NaN detected in output. Starting debug sequence...\n")
                self.debug_this(self.action_mu, state)
                breakpoint()

        # Actor (Log Std)
        attn_output_std, _ = self.action_log_std[0](state, state, state, attn_mask=mask)
        log_std = self.action_log_std[1:](attn_output_std)
        log_std = log_std[:, -1, :]  # Take the last sequence element

        std = log_std.exp()
        #dist = torch.distributions.Normal(mu, std)
        #logprobs = dist.log_prob(action).sum(axis=-1)
        variance = std.pow(2)
        # Probability density function calculation for normal distribution
        probs = torch.exp(-((action - mu).pow(2)) / (2 * variance)) / torch.sqrt(2 * torch.tensor(torch.pi).to(action.device) * variance)
    
        # Add a small epsilon to the probabilities to prevent taking log of zero
        safe_probs = probs + self.epsilon  # where self.epsilon is a small constant, e.g., 1e-10
    
        # Now, calculate log probabilities
        logprobs = torch.log(safe_probs).sum(axis=-1)  # You may not need to sum, depending on your specific use case

        if self.debug:
            # Before calculating the ratio, add these checks
            if (torch.isnan(logprobs).any() or torch.isinf(logprobs).any()):
                print("Anomaly detected in 'logprobs'")
                breakpoint()  # Insert your handling here
                
        # Sanitize logprobs to replace -inf with large negative numbers
        clean_logprobs = torch.where(
            torch.isinf(logprobs),
            torch.full_like(logprobs, -1e7),
            logprobs
        )

        # Value Layer
        attn_output_value, _ = self.value_layer[0](state, state, state, attn_mask=mask)
        state_value = self.value_layer[1:](attn_output_value)[:,-1,:]

        return clean_logprobs, torch.squeeze(state_value)

