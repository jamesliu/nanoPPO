import torch
from torch import optim
import gym
from nanoppo.policy.network import PolicyNetwork, ValueNetwork
from nanoppo.policy.actor_critic import ActorCritic
from torch.optim.lr_scheduler import ExponentialLR, CosineAnnealingLR


class NetworkManager:
    def __init__(self, env, optimizer_config, hidden_size, init_type, device, network_type="actor_critic"):
        self.network_type = network_type
        self.env = env
        self.optimizer_config = optimizer_config
        self.hidden_size = hidden_size
        self.init_type = init_type
        self.device = device

    def setup_networks(self):
        if isinstance(self.env.observation_space, gym.spaces.Dict):
            observation_space = self.env.observation_space.spaces["obs"]
        else:
            observation_space = self.env.observation_space
        if isinstance(self.env.action_space, gym.spaces.Box):
            action_dim = self.env.action_space.shape[0]
        else:
            action_dim = self.env.action_space.n

        if self.network_type == "actor_critic":
            policy = ActorCritic(
                state_dim=observation_space.shape[0],
                action_dim=action_dim,
                n_latent_var=self.hidden_size,
                action_low=self.env.action_space.low,
                action_high=self.env.action_space.high
            ).to(self.device)
            value = policy.value_layer
            optimizer = optim.Adam(
                policy.parameters(),
                lr=self.optimizer_config["policy_lr"],
                betas=(self.optimizer_config["beta1"], self.optimizer_config["beta2"]),
                eps=self.optimizer_config["epsilon"],
                weight_decay=self.optimizer_config["weight_decay"],
            )
        else:
            policy = PolicyNetwork(
                observation_space.shape[0], action_dim, init_type=self.init_type
            ).to(self.device)
            value = ValueNetwork(
                observation_space.shape[0],
                hidden_size=self.hidden_size,
                init_type=self.init_type,
            ).to(self.device)
    
            policy_lr = self.optimizer_config["policy_lr"]
            value_lr = self.optimizer_config["value_lr"]
    
            optimizer = optim.Adam(
                [
                    {"params": policy.parameters(), "lr": policy_lr},
                    {"params": value.parameters(), "lr": value_lr},
                ],
                betas=(self.optimizer_config["beta1"], self.optimizer_config["beta2"]),
                eps=self.optimizer_config["epsilon"],
                weight_decay=self.optimizer_config["weight_decay"],
            )
    
        if self.optimizer_config["scheduler"] is None:
            scheduler = None
        elif self.optimizer_config["scheduler"] == "exponential":
            scheduler = ExponentialLR(
                optimizer, gamma=self.optimizer_config["exponential_gamma"]
            )
        elif self.optimizer_config["scheduler"] == "cosine":
            scheduler = CosineAnnealingLR(
                optimizer, T_max=self.optimizer_config["cosine_T_max"]
            )
        else:
            raise ValueError(
                f"Scheduler {self.optimizer_config['scheduler']} not recognized."
            )

        return policy, value, optimizer, scheduler
