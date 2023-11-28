import torch
import torch.nn as nn
import torch.optim as optim

# Neural Network for Actor-Critic
import torch.nn as nn
import torch.nn.functional as F
from nanoppo.policy.actor_critic import ActorCritic
import wandb


# PPO Agent
class PPOAgent:
    def __init__(
        self,
        state_dim,
        action_dim,
        n_latent_var,
        policy_class,
        policy_lr,
        value_lr,
        betas,
        gamma,
        K_epochs,
        eps_clip,
        state_normalizer,
        action_low,
        action_high,
        vl_coef=0.5,
        el_coef=0.001,
        lr_scheduler=None,  # Add lr_scheduler as an optional argument
        device="cpu",
        wandb_log=False,
        debug = False
    ):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.state_normalizer = state_normalizer
        self.action_low = action_low
        self.action_high = action_high
        self.vl_coef = vl_coef
        self.el_coef = el_coef
        self.device = device
        self.debug = debug

        # Initialize optimizer with a placeholder
        self.optimizer = None

        action_low_tensor = torch.tensor(action_low, dtype=torch.float32).to(device)
        action_high_tensor = torch.tensor(action_high, dtype=torch.float32).to(device)
        
        if policy_class:
            self.policy = policy_class(
                state_dim, action_dim, n_latent_var, action_low_tensor, action_high_tensor, device=device, debug=debug
            ).float().to(device)
            self.policy_old = policy_class(
                state_dim, action_dim, n_latent_var, action_low_tensor, action_high_tensor, device=device, debug=debug
            ).float().to(device)
            self.policy_old.load_state_dict(self.policy.state_dict())
        else:
            self.policy = ActorCritic(
                state_dim, action_dim, n_latent_var, action_low_tensor, action_high_tensor, debug=debug
            ).float().to(device)
            self.policy_old = ActorCritic(
                state_dim, action_dim, n_latent_var, action_low_tensor, action_high_tensor, debug=debug
            ).float().to(device)
            self.policy_old.load_state_dict(self.policy.state_dict())

        # Separate the parameters of the actor and critic networks
        actor_params = list(self.policy.action_mu.parameters()) + list(
            self.policy.action_log_std.parameters()
        )
        critic_params = list(self.policy.value_layer.parameters())

        # Use the learning rates from the lr_scheduler if provided, or use the original learning rates
        if lr_scheduler is not None:
            actor_lr = lr_scheduler.get_lr_actor(it=0)
            critic_lr = lr_scheduler.get_lr_critic(it=0)
        else:
            actor_lr = policy_lr
            critic_lr = value_lr

        # Create an optimizer with different learning rates for the actor and critic
        self.optimizer = torch.optim.Adam(
            [
                {"params": actor_params, "lr": actor_lr},
                {"params": critic_params, "lr": critic_lr},
            ],
            betas=betas,
        )

        self.mse_loss = nn.MSELoss()
        self.wandb_log = wandb_log

        # Store the lr_scheduler if provided
        self.lr_scheduler = lr_scheduler

        self.iterations = 0

    def check_weights(self, model_part):
        if self.debug:
            for name, param in model_part.named_parameters():
                if torch.isnan(param.data).any():
                    print(f"NaN detected in weights: {name}")
                    # Here you can call your debugging method or do other appropriate handling
                    self.debug_this(model_part)
                    breakpoint()  # or raise an exception, or log the issue, etc.
    
    def check_gradients(self, model_part):
        if self.debug:
            for name, param in model_part.named_parameters():
                if param.grad is not None and torch.isnan(param.grad).any():
                    print(f"NaN detected in gradients: {name}")
                    # Here you can call your debugging method or do other appropriate handling
                    self.debug_this(model_part)
                    breakpoint()  # or raise an exception, or log the issue, etc.
        
    # Define the debug function
    def debug_this(self, model, state = None):
        print("Debugging Information:")
        # Check and print the weights and biases
        print("\nWeights and Biases in each layer:")
        for name, param in model.named_parameters():
            print(name, param.data)
        
        # Pass the input through each layer individually and print the outputs
        if state:
            x = state
            for i, layer in enumerate(model):
                x = layer(x)
                print(f"\nOutput after layer {i} ({layer.__class__.__name__}):")
                print(x)
                # If you find NaN at this stage, you can break the loop for efficiency
                if torch.isnan(x).any():
                    print(f"Stopping early due to NaN at layer {i}")
                    break

    def update(self, states, actions, returns, next_states, dones):
        # Before backpropagation: Check for NaN
        self.check_weights(self.policy.action_mu)

        for _ in range(self.K_epochs):
            if self.debug:
                # Check the actions for invalid values
                if torch.isnan(actions).any() or torch.isinf(actions).any():
                    print("Invalid values detected in 'actions'")
                    breakpoint()  # Insert your handling here
     
            # Getting predicted values and log probs for given states and actions
            logprobs, state_values = self.policy.evaluate(states, actions)

            # NEW: Check for NaNs in state_values returned from the policy
            if self.debug:
                if torch.isnan(state_values).any() or torch.isinf(state_values).any():
                    print("Invalid values detected in 'state_values'")
                    breakpoint()  # Insert your handling here

            # Calculate the advantages
            advantages = returns - state_values.detach()
            # Normalize the advantages (optional, but can help in training stability)
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)

            # NEW: Check for NaNs after normalizing advantages
            if self.debug:
                if torch.isnan(advantages).any() or torch.isinf(advantages).any():
                    print("Invalid values detected after normalizing 'advantages'")
                    breakpoint()  # Insert your handling here

            # Compute ratio for PPO
            old_logprobs, _ = self.policy_old.evaluate(states, actions)
            
            if self.debug:
                # Before calculating the ratio, add these checks
                if torch.isnan(logprobs).any() or torch.isinf(logprobs).any():
                    print("Anomaly detected in 'logprobs'")
                    breakpoint()  # Insert your handling here
                
                if torch.isnan(old_logprobs).any() or torch.isinf(old_logprobs).any():
                    print("Anomaly detected in 'old_logprobs'")
                    breakpoint()  # Insert your handling here

            log_diff = logprobs - old_logprobs.detach()
            clamp_value = 50
            log_diff_clamped = torch.clamp(log_diff, -clamp_value, clamp_value)  # clamp_value could be a large number like 50 or 100
            ratio = torch.exp(log_diff_clamped)

            if self.debug:
                # Checking critical tensors for NaNs
                if torch.isnan(ratio).any():
                    print("NaN detected in 'ratio'")
                    breakpoint()  # Insert your handling here
                
                if torch.isinf(ratio).any():
                    print("Inf detected in 'ratio'")
                    breakpoint()  # Insert your handling here
                
                if torch.isnan(advantages).any():
                    print("NaN detected in 'advantages'")
                    breakpoint()  # Insert your handling here

                if torch.isinf(advantages).any():
                    print("Inf detected in 'advantages'")
                    breakpoint()  # Insert your handling here

            # Surrogate loss
            surr1 = ratio * advantages
            surr2 = (
                torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            )
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss
            value_loss = self.vl_coef * self.mse_loss(state_values, returns)

            # Entropy (for exploration)
            entropy_loss = -self.el_coef * logprobs.mean()

            if self.debug:
                if torch.isnan(policy_loss) or torch.isinf(policy_loss):
                    print("Anomaly in policy loss detected.")
                    # Additional checks can be placed here
                    self.check_gradients(self.policy)
                
                    breakpoint()  # Insert your handling here
                
                if torch.isnan(entropy_loss) or torch.isinf(entropy_loss):
                    print("Anomaly in entropy loss detected.")
                    breakpoint()  # Insert your handling here
    
            # Total loss
            loss = policy_loss + value_loss + entropy_loss
            
            if self.debug:
                if torch.isnan(loss):
                    print("NaN detected in loss")
                    # handle the issue as appropriate: stop training, log, raise exception, etc.
                    breakpoint()

            # Optimize policy network
            self.optimizer.zero_grad()
            loss.backward()

            # After backpropagation/Before optimizer step: Check if gradients are NaN
            self.check_gradients(self.policy.action_mu)

            self.policy.check_for_nan_gradients()

            # Clip gradients to prevent exploding gradients
            # Large gradients can cause the weights to update too aggressively. 
            # Actions can have extream values e.g.(100, -0.10, ...)
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.7)
        
            self.optimizer.step()
            # After optimizer step: Check for NaN
            self.check_weights(self.policy.action_mu)

            if self.wandb_log:
                wandb.log(
                    {
                        "policy_loss": policy_loss.item(),
                        "value_loss": value_loss.item(),
                        "entropy_loss": entropy_loss.item(),
                        "total_loss": loss.item(),
                    }
                )
            
            if self.debug:
                # Check for NaNs in the model's parameters
                for name, param in self.policy.named_parameters():
                    if torch.isnan(param.data).any():
                        print(f"NaN detected in the model parameters: {name}")
                        breakpoint()

        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        # Update learning rates using the lr_scheduler if provided
        self.iterations += 1
        if self.lr_scheduler is not None:
            self.lr_scheduler.step(self.optimizer, self.iterations)

        # Log the learning rates for each param_group if wandb_log is enabled
        if self.wandb_log:
            for i, param_group in enumerate(self.optimizer.param_groups):
                learning_rate = param_group['lr']
                wandb.log({f"learning_rate_group_{i}": learning_rate})

    def save(self, path):
        # Saving
        torch.save(
            {
                "model_state_dict": self.policy.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "state_normalizer_state": self.state_normalizer.get_state(),
            },
            path,
        )

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint["model_state_dict"])
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.state_normalizer.set_state(checkpoint["state_normalizer_state"])
