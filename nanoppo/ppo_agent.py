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
from nanoppo.network import PolicyNetwork, ValueNetwork
from nanoppo.rollout_buffer import RolloutBuffer
from nanoppo.reward_scaler import RewardScaler
from nanoppo.reward_shaper import RewardShaper, TDRewardShaper
from nanoppo.state_scaler import StateScaler
from nanoppo.metrics_recorder import MetricsRecorder
from nanoppo.ppo_utils import compute_gae, compute_returns_and_advantages_without_gae

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
        ) = self.network_manager.setup_networks()

        self.rollout_buffer = RolloutBuffer(self.config["rollout_buffer_size"])
        if self.config["rescaling_rewards"]:
            self.reward_scaler = RewardScaler()
        else:
            self.reward_scaler = None

        if self.config["scale_states"] is None:
            self.state_scaler = None
        else:
            self.state_scaler = StateScaler(
                self.env, sample_size=10000, scale_type=self.config["scale_states"]
            )

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
        self.checkpoint_path = os.path.join(
            self.config["checkpoint_dir"], self.project, self.env_name
        )

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
            self.policy, self.value, self.optimizer, checkpoint_path, epoch
        )
        return last_epoch

    @staticmethod
    def select_action(policy: PolicyNetwork, state, device, action_min, action_max):
        state = torch.from_numpy(state).float().to(device)
        dist = policy(state)
        action = dist.sample()
        # If action space is continuous, compute the log_prob of the action, sum(-1) to sum over all dimensions
        log_prob = dist.log_prob(action).sum(-1)

        # Extract mean and std of the action distribution
        action_mean = dist.mean
        action_std = dist.stddev

        # action = torch.tanh(action)  # Pass the sampled action through the tanh activation function
        # action = action + torch.normal(mean=torch.zeros_like(action), std=action_std)  # Add noise to the action
        action = action.clamp(
            action_min, action_max
        )  # Clip the action to the valid range of the action space
        return (
            action.cpu().detach(),
            log_prob.cpu().detach(),
            action_mean.cpu().detach(),
            action_std.cpu().detach(),
        )

    @staticmethod
    def rollout_with_step(
        policy,
        value,
        env,
        device,
        rollout_buffer: RolloutBuffer,
        state_scaler: StateScaler,
        reward_shaper: RewardShaper,
        reward_scaler: RewardScaler,
        wandb_log: bool,
        debug: bool,
    ):
        total_steps = 0
        while True:
            steps = 0
            state, info = env.reset()
            if isinstance(state, dict):
                state = state["obs"]
            if state_scaler:
                scaled_state = state_scaler.scale_state(state)
            else:
                scaled_state = state
            done = False
            truncated = False
            accumulated_rewards = 0

            while (not done) and (not truncated):
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

                action = action.numpy()
                log_prob = log_prob.numpy()

                next_state, reward, done, truncated, info = env.step(action)

                if isinstance(next_state, dict):
                    next_state = next_state["obs"]
                if state_scaler is None:
                    scaled_next_state = next_state
                else:
                    scaled_next_state = state_scaler.scale_state(next_state)

                scaled_state = scaled_next_state
                accumulated_rewards += reward
                # Reshape rewards
                if reward_shaper is None:
                    reshaped_reward = reward
                else:
                    reshaped_reward = reward_shaper.reshape(
                        [reward], [state], [next_state]
                    )
                # Scale rewards
                if reward_scaler is None:
                    scaled_reward = reshaped_reward
                    # click.secho("Warning: Reward scaling is not applied.", fg="yellow", err=True)
                else:
                    scaled_reward = reward_scaler.scale_rewards([reshaped_reward])[0]
                rollout_buffer.push(
                    scaled_state,
                    action.squeeze(),
                    log_prob,
                    scaled_reward,
                    scaled_next_state,
                    done,
                )
                total_steps += 1
                steps += 1
                if done or truncated:
                    total_rewards = accumulated_rewards
                    accumulated_rewards = 0
                    if debug:
                        print(
                            "total steps",
                            total_steps,
                            "steps",
                            steps,
                            "total_rewards(done or truncated)",
                            total_rewards,
                            "done",
                            done,
                            "truncated",
                            truncated,
                            "reward",
                            reward,
                            "reshaped_reward",
                            reshaped_reward,
                            "scaled_reward",
                            scaled_reward,
                        )
                    yield total_rewards, steps, total_steps
                else:
                    yield None, steps, total_steps

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
        entropy = dist.entropy().mean()  # Compute the mean entropy of the distribution
        return (
            -torch.min(surr1, surr2).mean(),
            -entropy_coef * entropy,
        )  # Add the entropy term to the policy loss

    @staticmethod
    def compute_value_loss(value, states, returns, l1_loss):
        # Compute value loss
        v_pred = value(states).squeeze()
        v_target = returns.squeeze()
        value_loss = (
            F.smooth_l1_loss(v_pred, v_target)
            if l1_loss
            else F.mse_loss(v_pred, v_target)
        )
        return value_loss

    @staticmethod
    def train_network(
        iter_num,
        policy,
        value,
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
        max_grad_norm,
        use_gae,
        tau,
        l1_loss,
        wandb_log,
        metrics_recorder: MetricsRecorder,
    ):
        assert (
            len(rollout_buffer) >= batch_size
        ), f"Rollout buffer length {len(rollout_buffer)} is less than batch size {batch_size}"
        for sgd_iter in range(sgd_iters):
            # Sample from the rollout buffer
            (
                batch_states,
                batch_actions,
                batch_probs,
                batch_rewards,
                batch_next_states,
                batch_dones,
            ) = rollout_buffer.sample(batch_size, device=device)

            # Compute Advantage and Returns
            returns = []
            advs = []
            g = 0
            # Compute returns and advantages from rollout buffer samples out of order
            with torch.no_grad():
                if use_gae:
                    # Compute Advantage using GAE and Returns
                    values = value(batch_states).squeeze().tolist()
                    next_value = value(batch_next_states[-1]).item()
                    masks = [1 - done.item() for done in batch_dones]
                    returns = compute_gae(
                        next_value, batch_rewards, masks, values, gamma, tau
                    )
                    advs = [ret - val for ret, val in zip(returns, values)]
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
            advs = torch.tensor(advs, dtype=torch.float32).to(device)
            advs = (advs - advs.mean()) / (advs.std() + 1e-10)

            # Create mini-batches
            num_samples = len(batch_rewards)
            assert num_samples == batch_size
            num_batches = num_samples // batch_size
            assert num_batches == 1

            policy_loss, entropy_loss = PPOAgent.surrogate(
                policy,
                old_probs=batch_probs,
                states=batch_states,
                actions=batch_actions,
                advs=advs,
                clip_param=clip_param,
                entropy_coef=entropy_coef,
            )
            """
            # clear the list of activation norms for each epoch
            activation_norms = []
            def hook(module, input, output):
                if isinstance(output, Tuple):
                    for o in output:
                        activation_norms.append(o.norm().item())
                else:
                    activation_norms.append(output.norm().item())
            # register activation norm hook
            # add forward hook to each layer of the value network
            hooks = []
            for module in value.modules():
                hooks.append(module.register_forward_hook(hook))
            """

            value_loss = PPOAgent.compute_value_loss(
                value, batch_states, returns, l1_loss
            )

            # Compute total loss and update parameters
            total_loss = policy_loss + entropy_loss + vf_coef * value_loss

            optimizer.zero_grad()
            total_loss.backward()

            # Clip the gradients to avoid exploding gradients
            policy_grad_norm = nn.utils.clip_grad_norm_(
                policy.parameters(), max_grad_norm
            )
            value_grad_norm = nn.utils.clip_grad_norm_(
                value.parameters(), max_grad_norm
            )

            # compute activation norm
            # remove the forward hooks
            """
            for hook in hooks:
                hook.remove()
            # compute the mean activation norm for the value network
            activation_norm = sum(activation_norms) / len(activation_norms)
            """

            optimizer.step()
            if scheduler is not None:
                scheduler.step()

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
                        "Policy/Gradient_Norm": policy_grad_norm,
                        "Value/Gradient_Norm": value_grad_norm,
                        # "Value/Activation_Norm": activation_norm
                    }
                )
                # wandb.log({"Gradients/ValueNet": wandb.Histogram(value.fc1.weight.grad.detach().cpu().numpy())})
                log_std_value = policy.log_std.detach().cpu().numpy()
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
        return (
            policy,
            value,
            iter_num,
        )

    @staticmethod
    def train_with_epoch(
        env: gym.Env,
        env_name: str,
        env_config: dict,
        rollout_buffer: RolloutBuffer,
        state_scaler: StateScaler,
        shape_reward: RewardShaper,
        reward_scaler: RewardScaler,
        policy: PolicyNetwork,
        value: ValueNetwork,
        optimizer: optim.Optimizer,
        scheduler: optim.lr_scheduler._LRScheduler,
        epochs: int,
        batch_size: int,
        sgd_iters: int,
        gamma: float,
        # optimizer_config:dict,
        # hidden_size:int,
        # init_type:str,
        clip_param: float,
        vf_coef: float,
        entropy_coef: float,
        max_grad_norm: float,
        use_gae: bool,
        tau: float,
        l1_loss: bool,
        verbose: int,
        rollout_buffer_size: int,
        checkpoint_interval: int = -1,
        checkpoint_path: str = None,
        resume_training: bool = False,
        resume_epoch: bool = None,
        report_func: callable = None,
        # project:str="continuous-action-ppo",
        device: str = "cpu",
        wandb_log: bool = True,
        metrics_recorder: MetricsRecorder = None,
        # seed:int=None,
        **kwargs,
    ):
        if resume_training:
            last_epoch = CheckpointManager.load_checkpoint(
                policy, value, optimizer, checkpoint_path, epoch
            )
        else:
            last_epoch = 0

        if shape_reward is None:
            reward_shaper = None
        elif shape_reward == TDRewardShaper:
            reward_shaper = TDRewardShaper(model=value, device=device)
        else:
            reward_shaper = shape_reward()

        # Set up rollout generator
        rollout = PPOAgent.rollout_with_step(
            policy=policy,
            value=value,
            env=env,
            device=device,
            rollout_buffer=rollout_buffer,
            state_scaler=state_scaler,
            reward_shaper=reward_shaper,
            reward_scaler=reward_scaler,
            wandb_log=wandb_log,
            debug=True if verbose > 1 else False,
        )
        # Set up training loop
        episode_rewards = []
        episode_steps = []
        total_iters = epochs * sgd_iters
        train_iters = 0
        rollout_steps = 0
        average_reward = -np.inf
        start = time()
        for epoch in PPOAgent.get_epoch_iterator(last_epoch, epochs, verbose):
            policy.eval()
            value.eval()
            rollout_buffer.clear()  # clear the rollout buffer, all data is from the current policy
            for r in range(rollout_buffer_size):
                total_rewards, steps, rollout_steps = next(rollout)
                if total_rewards is not None:
                    episode_steps.append(steps)
                    episode_rewards.append(total_rewards)
            policy.train()
            value.train()
            _, _, train_iters = PPOAgent.train_network(
                train_iters,
                policy,
                value,
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
                max_grad_norm,
                use_gae=use_gae,
                tau=tau,
                l1_loss=l1_loss,
                wandb_log=wandb_log,
                metrics_recorder=metrics_recorder,
            )
            if checkpoint_interval > 0 and ((epoch + 1) % checkpoint_interval) == 0:
                CheckpointManager.save_checkpoint(
                    self.policy, self.value, self.optimizer, epoch, checkpoint_path
                )

            # Average of last 10 episodes or all episodes if less than 10 episodes are available
            average_reward = sum(episode_rewards[-20:]) / len(episode_rewards[-20:])
            if len(episode_rewards) >= 20:
                if wandb_log:
                    WandBLogger.log_rewards(episode_rewards[-20:])
                if metrics_recorder:
                    self.metrics_recorder.record_rewards(episode_rewards[-20:])
                if report_func:
                    report_func(mean_reward=average_reward)  # Reporting the reward

            if verbose > 0:
                print(
                    "epoch",
                    epoch + 1,
                    "average reward",
                    average_reward,
                    "train epochs",
                    epoch - last_epoch + 1,
                    "train iters",
                    train_iters,
                    "rollout episodes",
                    len(episode_rewards),
                    "rollout steps",
                    rollout_steps,
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
            env=self.env,
            env_name=self.env_name,
            env_config=self.config["env_config"],
            rollout_buffer=self.rollout_buffer,
            state_scaler=self.state_scaler,
            shape_reward=self.config["shape_reward"],
            reward_scaler=self.reward_scaler,
            policy=self.policy,
            value=self.value,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            epochs=epochs,
            batch_size=self.config["batch_size"],
            sgd_iters=self.config["sgd_iters"],
            gamma=self.config["gamma"],
            # optimizer_config=self.optimizer_config,
            # hidden_size=self.config["hidden_size"],
            # init_type=self.config["init_type"],
            clip_param=self.config["clip_param"],
            vf_coef=self.config["vf_coef"],
            entropy_coef=self.config["entropy_coef"],
            max_grad_norm=self.config["max_grad_norm"],
            use_gae=self.config["use_gae"],
            tau=self.config["tau"],
            l1_loss=self.config["l1_loss"],
            verbose=self.config["verbose"],
            rollout_buffer_size=self.config["rollout_buffer_size"],
            checkpoint_interval=self.config["checkpoint_interval"],
            checkpoint_path=self.checkpoint_path,
            resume_training=self.config["resume_training"],
            resume_epoch=self.config["resume_epoch"],
            report_func=self.config["report_func"],
            # project=self.config["project"],
            device=self.device,
            wandb_log=self.config["wandb_log"],
            metrics_recorder=self.metrics_recorder,
            # seed=self.config["seed"],
        )
