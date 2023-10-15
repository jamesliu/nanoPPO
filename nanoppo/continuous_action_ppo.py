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
        # Initialize optimizer with a placeholder
        self.optimizer = None

        action_low_tensor = torch.tensor(action_low, dtype=torch.float32).to(device)
        action_high_tensor = torch.tensor(action_high, dtype=torch.float32).to(device)

        self.policy = ActorCritic(
            state_dim, action_dim, n_latent_var, action_low_tensor, action_high_tensor
        ).float().to(device)
        self.policy_old = ActorCritic(
            state_dim, action_dim, n_latent_var, action_low_tensor, action_high_tensor
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

    def update(self, states, actions, returns, next_states, dones):
        for _ in range(self.K_epochs):
            # Getting predicted values and log probs for given states and actions
            logprobs, state_values = self.policy.evaluate(states, actions)
            # Calculate the advantages
            advantages = returns - state_values.detach()
            # Normalize the advantages (optional, but can help in training stability)
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)

            # Compute ratio for PPO
            old_logprobs, _ = self.policy_old.evaluate(states, actions)
            ratio = torch.exp(logprobs - old_logprobs.detach())

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

            # Total loss
            loss = policy_loss + value_loss + entropy_loss

            # Optimize policy network
            self.optimizer.zero_grad()
            loss.backward()

            # Clip gradients to prevent exploding gradients
            # Large gradients can cause the weights to update too aggressively. 
            # Actions can have extream values e.g.(100, -0.10, ...)
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
        
            self.optimizer.step()
            if self.wandb_log:
                wandb.log(
                    {
                        "policy_loss": policy_loss.item(),
                        "value_loss": value_loss.item(),
                        "entropy_loss": entropy_loss.item(),
                        "total_loss": loss.item(),
                    }
                )

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
