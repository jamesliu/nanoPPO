import torch
from torch import optim
import torch.nn as nn
from torch.nn import functional as F
from time import time
from torch.optim.lr_scheduler import ExponentialLR, CosineAnnealingLR
import numpy as np
from tqdm import tqdm
import gym
import os
from nanoppo.environment_manager import EnvironmentManager
from nanoppo.network_manager import NetworkManager
from nanoppo.checkpoint_manager import CheckpointManager
from nanoppo.wandb_logger import WandBLogger
from nanoppo.policy.network import PolicyNetwork, ValueNetwork
from nanoppo.rollout_buffer import RolloutBuffer
from nanoppo.reward_scaler import RewardScaler
from nanoppo.reward_shaper import RewardShaper, TDRewardShaper
from nanoppo.normalizer import Normalizer
from nanoppo.state_scaler import StateScaler
from nanoppo.metrics_recorder import MetricsRecorder
from nanoppo.ppo_utils import (
    compute_gae,
    compute_returns_and_advantages_without_gae,
    get_grad_norm,
)

import warnings

# Suppress the specific warning
warnings.filterwarnings(
    "ignore",
    message="Could not parse CUBLAS_WORKSPACE_CONFIG, using default workspace size of 8519680 bytes.",
)


class PPOAgent:
    def __init__(self, config, optimizer_config, force_cpu=False):
        self.config = config
        self.optimizer_config = optimizer_config
        if force_cpu:
            self.device = torch.device("cpu")
        else:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.project = config["project"]
        self.env_name = config["env_name"]
        self.env_manager = EnvironmentManager(self.env_name, config["env_config"])
        self.env = self.env_manager.setup_env()

        self.network_manager = NetworkManager(
            self.env,
            optimizer_config,
            config["hidden_size"],
            config["init_type"],
            self.device,
        )
        (
            self.policy,
            self.value,
            self.optimizer,
            self.scheduler,
            self.policy_old,
            self.value_old,
        ) = self.network_manager.setup_networks()

        self.rollout_buffer = RolloutBuffer(self.config["batch_size"])
        if self.config["rescaling_rewards"]:
            self.reward_scaler = RewardScaler()
        else:
            self.reward_scaler = None

        self.normalizer = None
        self.state_scaler = None
        if self.config["scale_states"] == "default":
            self.normalizer = Normalizer(self.env.observation_space.shape[0])
        elif self.config["scale_states"] in [
            "env",
            "standard",
            "minmax",
            "robust",
            "quantile",
        ]:
            self.state_scaler = StateScaler(
                self.env, sample_size=10000, scale_type=self.config["scale_states"]
            )
        else:
            raise ValueError(f"Unknown scale type: {self.config['scale_states']}")

        self.metrics_log = self.config["metrics_log"]
        if self.metrics_log:
            self.metrics_recorder = MetricsRecorder()
        else:
            self.metrics_recorder = None

        self.project = self.config["project"]
        self.wandb_log = self.config["wandb_log"]
        # Initialize WandB
        if self.wandb_log:
            config = locals().copy()
            keys_to_delete = [
                "checkpoint_interval",
                "checkpoint_path",
                "resume_training",
                "resume_epoch",
                "report_func",
            ]
            [config.pop(key, None) for key in keys_to_delete]
            WandBLogger.init(project=self.project, name=self.env_name, config=config)
        self.checkpoint_interval = self.config["checkpoint_interval"]
        self.checkpoint_dir = os.path.join(
            self.config["checkpoint_dir"], self.project, self.env_name
        )
        self.log_interval = self.config["log_interval"]

    @staticmethod
    def get_epoch_iterator(last_epoch, epochs, verbose: int):
        if verbose > 0:
            progress_iterator = tqdm(range(last_epoch, last_epoch + epochs))
        else:
            progress_iterator = range(last_epoch, last_epoch + epochs)
        return progress_iterator

    @staticmethod
    def load_checkpoint(self, checkpoint_path, epoch=None):
        last_epoch = CheckpointManager.load_checkpoint(
            self.policy,
            self.value,
            self.optimizer,
            self.normalizer,
            checkpoint_path,
            epoch,
        )
        return last_epoch

    @staticmethod
    def select_action(policy: PolicyNetwork, state, device, action_min, action_max):
        state = torch.FloatTensor(state).to(device)
        dist = policy(state)
        action = dist.sample()
        # If action space is continuous, compute the log_prob of the action, sum(-1) to sum over all dimensions
        log_prob = dist.log_prob(action).sum(-1)

        # Extract mean and std of the action distribution
        action_mean = dist.mean
        action_std = dist.stddev

        # action = torch.tanh(action)  # Pass the sampled action through the tanh activation function
        # action = action + torch.normal(mean=torch.zeros_like(action), std=action_std)  # Add noise to the action
        # action = action.clamp(
        #    action_min, action_max
        # )  # Clip the action to the valid range of the action space
        return (
            action.cpu().detach().numpy(),
            log_prob.cpu().detach().numpy(),
            action_mean.cpu().detach().numpy(),
            action_std.cpu().detach().numpy(),
        )

    @staticmethod
    def surrogate(policy, old_probs, states, actions, advs, clip_param, entropy_coef):
        # Policy loss
        dist = policy(states)
        new_probs = dist.log_prob(actions).sum(-1)
        ratio = torch.exp(new_probs - old_probs)  # Importance sampling ratio
        surr1 = ratio * advs
        surr2 = (
            torch.clamp(ratio, 1 - clip_param, 1 + clip_param) * advs
        )  # Trust region clipping
        # entropy = dist.entropy().mean()  # Compute the mean entropy of the distribution
        # entropy_loss = - entropy_coef * entropy

        # Entropy (for exploration)
        approximate_entropy_loss = -entropy_coef * new_probs.mean()

        return (
            -torch.min(surr1, surr2).mean(),
            approximate_entropy_loss,
        )  # Add the entropy term to the policy loss

    @staticmethod
    def compute_value_loss(state_values, returns):
        # Compute value loss
        v_pred = state_values
        v_target = returns.squeeze()
        value_loss = F.mse_loss(v_pred, v_target)
        return value_loss

    @staticmethod
    def minibatch_update(
        iter_num,
        policy,
        value,
        policy_old,
        value_old,
        optimizer,
        scheduler,
        rollout_buffer: RolloutBuffer,
        device,
        batch_size,
        sgd_iters,
        gamma,
        clip_param,
        vf_coef,
        entropy_coef,
        max_grad_norm,
        use_gae,
        tau,
        wandb_log,
        metrics_recorder: MetricsRecorder,
    ):
        (
            batch_states,
            batch_actions,
            batch_log_probs,
            batch_rewards,
            batch_next_states,
            batch_dones,
        ) = rollout_buffer.sample(batch_size, device=device, randomize=False)

        # Compute returns once from rollout buffer in order
        if use_gae:
            # Compute Advantage using GAE and Returns
            values = (
                value(batch_states).detach().squeeze().tolist()
            )  # For values + [next_value]
            next_value = value(batch_next_states[-1]).item()
            masks = [1 - done.item() for done in batch_dones]
            # Only compute returns once
            returns = compute_gae(next_value, batch_rewards, masks, values, gamma, tau)
        else:
            returns, advs = compute_returns_and_advantages_without_gae(
                batch_rewards,
                batch_states,
                batch_next_states,
                batch_dones,
                value,
                gamma,
            )

        returns = torch.tensor(returns, dtype=torch.float32).to(device)
        for sgd_iter in range(sgd_iters):
            # Compute advantages separately for each SGD iteration
            state_values = value(batch_states).squeeze()
            advantages = returns - state_values.detach()
            # Normalize the advantages (optional, but can help in training stability)
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)

            policy_loss, entropy_loss = PPOAgent.surrogate(
                policy,
                old_probs=batch_log_probs,
                states=batch_states,
                actions=batch_actions,
                advs=advantages,
                clip_param=clip_param,
                entropy_coef=entropy_coef,
            )

            value_loss = PPOAgent.compute_value_loss(state_values, returns)

            # Compute total loss and update parameters
            total_loss = policy_loss + entropy_loss + vf_coef * value_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            # if scheduler is not None:
            #    scheduler.step()

            """
            # Clip the gradients to avoid exploding gradients
            policy_grad_norm = nn.utils.clip_grad_norm_(
                policy.parameters(), max_grad_norm
            )
            value_grad_norm = nn.utils.clip_grad_norm_(
                value.parameters(), max_grad_norm
            )
            """
            action_mu_grad_norm = get_grad_norm(policy.action_mu.parameters())
            action_log_std_grad_norm = get_grad_norm(policy.action_log_std.parameters())
            value_grad_norm = get_grad_norm(value.parameters())

            # compute activation norm
            # remove the forward hooks
            """
            for hook in hooks:
                hook.remove()
            # compute the mean activation norm for the value network
            activation_norm = sum(activation_norms) / len(activation_norms)
            """

            if wandb_log:
                # Log the losses and gradients to WandB
                WandBLogger.log(
                    {
                        "iteration": iter_num,
                        "Loss/Total": total_loss.item(),
                        "Loss/Policy": policy_loss.item(),
                        "Loss/Entropy": entropy_loss.item(),
                        "Loss/Value": value_loss.item(),
                        "Loss/Coef_Value": vf_coef * value_loss.item(),
                    }
                )
                # wandb.log({"Gradients/PolicyNet": wandb.Histogram(policy.fc1.weight.grad.detach().cpu().numpy())})
                WandBLogger.log(
                    {
                        "Policy/Mu_Gradient_Norm": action_mu_grad_norm,
                        "Policy/Log_Std_Gradient_Norm": action_log_std_grad_norm,
                        "Value/Gradient_Norm": value_grad_norm,
                        # "Value/Activation_Norm": activation_norm
                    }
                )
                # wandb.log({"Gradients/ValueNet": wandb.Histogram(value.fc1.weight.grad.detach().cpu().numpy())})
                log_std_value = (
                    policy.action_log_std(batch_states).detach().cpu().numpy()
                )
                WandBLogger.log({"Policy/Log_Std": log_std_value})
            # log the learning rate to wandb
            lrs = {}
            for i, param_group in enumerate(optimizer.param_groups):
                lr = param_group["lr"]
                lrs["learning_rate_{}".format(i)] = lr
                if wandb_log:
                    WandBLogger.log({"LR/LearningRate_{}".format(i): lr})

            if metrics_recorder:
                metrics_recorder.record_losses(
                    total_loss.item(),
                    policy_loss.item(),
                    entropy_loss.item(),
                    value_loss.item(),
                )
                metrics_recorder.record_learning(lrs)

            iter_num += 1

        rollout_buffer.clear()  # clear the rollout buffer, all data is from the current policy
        assert len(rollout_buffer) == 0
        # Copy new weights into old policy
        policy_old.load_state_dict(policy.state_dict())
        return (
            policy,
            value,
            iter_num,
        )

    @staticmethod
    def train_with_epoch(
        project: str,
        env: gym.Env,
        env_name: str,
        env_config: dict,
        rollout_buffer: RolloutBuffer,
        state_scaler: StateScaler,
        shape_reward: RewardShaper,
        normalizer: Normalizer,
        reward_scaler: RewardScaler,
        policy: PolicyNetwork,
        value: ValueNetwork,
        policy_old,
        value_old,
        optimizer: optim.Optimizer,
        scheduler: optim.lr_scheduler._LRScheduler,
        epochs: int,
        max_timesteps: int,
        batch_size: int,
        sgd_iters: int,
        gamma: float,
        clip_param: float,
        vf_coef: float,
        entropy_coef: float,
        max_grad_norm: float,
        use_gae: bool,
        tau: float,
        verbose: int,
        checkpoint_interval: int,
        checkpoint_dir: str,
        log_interval: int,
        resume_training: bool,
        resume_epoch: int,
        report_func: callable = None,
        device: str = "cpu",
        wandb_log: bool = True,
        metrics_recorder: MetricsRecorder = None,
    ):
        checkpoint_path = os.path.join(checkpoint_dir, project, env_name)
        if resume_training:
            if verbose > 0:
                print(
                    "Resuming training from checkpoint...",
                    checkpoint_path,
                    "from latest"
                    if resume_epoch <= 0
                    else f"from epoch {resume_epoch}",
                )
            last_epoch = CheckpointManager.load_checkpoint(
                policy,
                value,
                optimizer,
                checkpoint_path,
                None if resume_epoch <= 0 else resume_epoch - 1,
            )
            last_epoch += 1
        else:
            last_epoch = 0

        if shape_reward is None:
            reward_shaper = None
        elif shape_reward == TDRewardShaper:
            reward_shaper = TDRewardShaper(model=value, device=device)
        else:
            reward_shaper = shape_reward()

        # Set up training loop
        total_iters = epochs * sgd_iters
        time_steps = 0
        episode_steps = []
        episode_rewards = []  # Initialize cumulative reward
        train_iters = 0
        average_reward = -np.inf
        best_reward = -np.inf
        start = time()
        for epoch in PPOAgent.get_epoch_iterator(last_epoch, epochs, verbose):
            state, info = env.reset()
            if isinstance(state, dict):
                state = state["obs"]
            if normalizer:
                normalizer.observe(state)
                scaled_state = normalizer.normalize(state)
            elif state_scaler:
                scaled_state = state_scaler.scale_state(state)
            else:
                raise ValueError("No state scaler or normalizer is provided")
            done = False
            total_reward = 0

            for step in range(max_timesteps):
                action, log_prob, action_mean, action_std = PPOAgent.select_action(
                    policy,
                    scaled_state,
                    device,
                    env.action_space.low[0],
                    env.action_space.high[0],
                )
                if wandb_log:
                    WandBLogger.log_action_distribution_parameters(
                        action_mean, action_std
                    )

                next_state, reward, done, truncated, info = env.step(action)

                if isinstance(next_state, dict):
                    next_state = next_state["obs"]
                if normalizer:
                    normalizer.observe(next_state)
                    scaled_next_state = normalizer.normalize(next_state)
                elif state_scaler:
                    scaled_next_state = state_scaler.scale_state(next_state)
                else:
                    raise ValueError("No state scaler or normalizer is provided")

                scaled_state = scaled_next_state
                total_reward += reward
                # Reshape rewards
                if reward_shaper is None:
                    reshaped_reward = reward
                else:
                    raise NotImplementedError
                    reshaped_reward = reward_shaper.reshape(
                        [reward], [state], [next_state]
                    )[0]
                # Scale rewards
                if reward_scaler is None:
                    scaled_reward = reshaped_reward
                    # click.secho("Warning: Reward scaling is not applied.", fg="yellow", err=True)
                else:
                    raise NotImplementedError
                    scaled_reward = reward_scaler.scale_rewards([reshaped_reward])[0]
                rollout_buffer.push(
                    state=scaled_state,
                    action=action.squeeze(),
                    log_prob=log_prob,
                    reward=scaled_reward,
                    next_state=scaled_next_state,
                    done=done,
                )
                time_steps += 1
                if time_steps % batch_size == 0:
                    _, _, train_iters = PPOAgent.minibatch_update(
                        train_iters,
                        policy,
                        value,
                        policy_old,
                        value_old,
                        optimizer,
                        scheduler,
                        rollout_buffer,
                        device,
                        batch_size,
                        sgd_iters,
                        gamma,
                        clip_param,
                        vf_coef,
                        entropy_coef,
                        max_grad_norm=max_grad_norm,
                        use_gae=use_gae,
                        tau=tau,
                        wandb_log=wandb_log,
                        metrics_recorder=metrics_recorder,
                    )

                if done or truncated:
                    break

            episode_steps.append(step + 1)
            episode_rewards.append(total_reward)

            # Average of last 10 episodes or all episodes if less than 10 episodes are available
            if len(episode_rewards) >= 20 and (epoch + 1) % log_interval == 0:
                average_reward = sum(episode_rewards[-20:]) / len(episode_rewards[-20:])
                if wandb_log:
                    WandBLogger.log_rewards(episode_rewards[-20:])
                if metrics_recorder:
                    metrics_recorder.record_rewards(episode_rewards[-20:])
                if report_func:
                    report_func(mean_reward=average_reward)  # Reporting the reward

            if verbose > 0 and (epoch + 1) % log_interval == 0:
                print(
                    "env",
                    env_name,
                    "epoch",
                    epoch + 1,
                    "reward episodes",
                    len(episode_rewards),
                    "everage steps",
                    np.mean(episode_steps),
                    "average reward",
                    round(average_reward, 2),
                    "train epochs",
                    epoch - last_epoch + 1,
                    "train iters",
                    train_iters,
                    "rollout episodes",
                    len(episode_rewards),
                    "rollout time steps",
                    time_steps,
                )

                action_mu_grad_norm = get_grad_norm(policy.action_mu.parameters())
                action_log_std_grad_norm = get_grad_norm(
                    policy.action_log_std.parameters()
                )
                value_grad_norm = get_grad_norm(value.parameters())
                print(
                    "action_mu_grad_norm",
                    round(action_mu_grad_norm, 2),
                    "action_log_std_grad_norm",
                    round(action_log_std_grad_norm, 2),
                    "value_grad_norm",
                    round(value_grad_norm, 2),
                )

            if checkpoint_interval > 0:
                if average_reward > best_reward:
                    print("Saving checkpoint...", checkpoint_path)
                    print("avg_reward", average_reward, "> best_reward", best_reward)
                    best_reward = average_reward
                    CheckpointManager.save_checkpoint(
                        policy, value, optimizer, normalizer, epoch, checkpoint_path
                    )

        end = time()
        print("Training time: ", round((end - start) / 60, 2), "minutes")
        if metrics_recorder:
            metrics_recorder.to_csv()
        if wandb_log:
            WandBLogger.finish()
        print(
            "Training complete",
            "average reward",
            average_reward,
            "total iters",
            total_iters,
        )
        return policy, value, average_reward, train_iters

    def train(self, epochs):
        PPOAgent.train_with_epoch(
            project=self.project,
            env=self.env,
            env_name=self.env_name,
            env_config=self.config["env_config"],
            rollout_buffer=self.rollout_buffer,
            state_scaler=self.state_scaler,
            shape_reward=self.config["shape_reward"],
            reward_scaler=self.reward_scaler,
            normalizer=self.normalizer,
            policy=self.policy,
            value=self.value,
            policy_old=self.policy_old,
            value_old=self.value_old,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            epochs=epochs,
            max_timesteps=self.config["max_timesteps"],
            batch_size=self.config["batch_size"],
            sgd_iters=self.config["sgd_iters"],
            gamma=self.config["gamma"],
            # optimizer_config=self.optimizer_config,
            # hidden_size=self.config["hidden_size"],
            # init_type=self.config["init_type"],
            clip_param=self.config.get("clip_param", 0.2),
            vf_coef=self.config["vf_coef"],
            entropy_coef=self.config["entropy_coef"],
            max_grad_norm=self.config["max_grad_norm"],
            use_gae=self.config["use_gae"],
            tau=self.config["tau"],
            verbose=self.config["verbose"],
            checkpoint_interval=self.config["checkpoint_interval"],
            checkpoint_dir=self.config["checkpoint_dir"],
            log_interval=self.config["log_interval"],
            resume_training=self.config["resume_training"],
            resume_epoch=self.config["resume_epoch"],
            report_func=self.config["report_func"],
            # project=self.config["project"],
            device=self.device,
            wandb_log=self.config["wandb_log"],
            metrics_recorder=self.metrics_recorder,
            # seed=self.config["seed"],
        )
