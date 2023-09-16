import torch
from torch import optim
import torch.nn as nn
from torch.nn import functional as F
from time import time
from torch.optim.lr_scheduler import ExponentialLR, CosineAnnealingLR
import numpy as np
from tqdm import tqdm
from nanoppo.environment_manager import EnvironmentManager
from nanoppo.network_manager import NetworkManager
from nanoppo.checkpoint_manager import CheckpointManager
from nanoppo.wandb_logger import WandBLogger
from nanoppo.network import PolicyNetwork, ValueNetwork
from nanoppo.rollout_buffer import RolloutBuffer
from nanoppo.reward_scaler import RewardScaler
from nanoppo.reward_shaper import RewardShaper
from nanoppo.state_scaler import StateScaler
from nanoppo.metrics_recorder import MetricsRecorder

class PPOAgent:
    def __init__(self, config, optimizer_config, force_cpu=False):
        self.config = config
        self.optimizer_config = optimizer_config
        if force_cpu:
            self.device = torch.device("cpu")
        else:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.env_manager = EnvironmentManager(config["env_name"], config["env_config"])
        self.env = self.env_manager.setup_env()

        self.network_manager = NetworkManager(self.env, optimizer_config, config["hidden_size"], config["init_type"], self.device)
        self.policy, self.value, self.optimizer, self.scheduler = self.network_manager.setup_networks()

        self.rollout_buffer = RolloutBuffer(self.config["rollout_buffer_size"])
        if self.config["rescaling_rewards"]:
            self.reward_scaler = RewardScaler()
        else:
            self.reward_scaler = None

        if self.config["scale_states"] is None:
            self.state_scaler = None
        else:
            self.state_scaler = StateScaler(self.env, sample_size=10000, scale_type=self.config["scale_states"])
        
        self.metrics_recorder = MetricsRecorder()

    def get_epoch_iterator(self, last_epoch, epochs, verbose):
        if verbose:
            progress_iterator = tqdm(range(last_epoch, last_epoch + epochs))
        else:
            progress_iterator = range(last_epoch, last_epoch + epochs)
        return progress_iterator

    def load_checkpoint(self, checkpoint_path, epoch=None):
        last_epoch = CheckpointManager.load_checkpoint(self.policy, self.value, self.optimizer, checkpoint_path, epoch)
        return last_epoch

    def rollout_with_step(self,
        policy,
        value,
        env,
        device,
        rollout_buffer: RolloutBuffer,
        state_scaler: StateScaler,
        reward_shaper: RewardShaper,
        reward_scaler: RewardScaler,
        wandb_log:bool,
        debug: bool
    ):
        total_steps =0
        while True:
            steps = 0
            state, info = env.reset()
            if isinstance(state, dict):
                state = state["obs"]
            if state_scaler is None:
                scaled_state = state
            else:
                scaled_state = state_scaler.scale_state(state)
            done = False
            truncated = False
            accumulated_rewards = 0
    
            while (not done) and (not truncated):
                action, log_prob, action_mean, action_std = select_action(
                        policy,
                        scaled_state,
                        device,
                        env.action_space.low[0],
                        env.action_space.high[0],
                    )
                if wandb_log:
                    wandb.log( {f"Policy/Action{i}_Mean":action_mean  for i, action_mean in enumerate(action_mean.numpy().tolist())})
                    wandb.log( {f"Policy/Action{i}_Std":action_std  for i, action_std in enumerate(action_std.numpy().tolist())})
    
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
                    reshaped_reward = reward_shaper.reshape([reward], [state], [next_state])    
                # Scale rewards
                if reward_scaler is None:
                    scaled_reward = reshaped_reward
                    #click.secho("Warning: Reward scaling is not applied.", fg="yellow", err=True)
                else:
                    scaled_reward = reward_scaler.scale_rewards([reshaped_reward])[0]
                rollout_buffer.push(
                    scaled_state, action.squeeze(), log_prob, scaled_reward, scaled_next_state, done
                )
                total_steps += 1
                steps += 1
                if done or truncated:
                    total_rewards = accumulated_rewards
                    if debug:
                         print('total steps', total_steps, 'steps', steps, 'accumulated_rewards', accumulated_rewards,  'done', done, 'truncated', truncated, 'reward', reward, 'reshaped_reward', reshaped_reward, 'scaled_reward', scaled_reward)
                    yield total_rewards, steps, total_steps
                else:
                    yield None, steps, total_steps

    def compute_gae(next_value, rewards, masks, values, gamma, tau):
        values = values + [next_value]
        gae = 0
        returns = []
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
            gae = delta + gamma * tau * masks[step] * gae
            returns.insert(0, gae + values[step])
        return returns
    
    
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
    
    
    def compute_value_loss(value, states, returns, l1_loss):
        # Compute value loss
        v_pred = value(states).squeeze()
        v_target = returns.squeeze()
        value_loss = (
            F.smooth_l1_loss(v_pred, v_target) if l1_loss else F.mse_loss(v_pred, v_target)
        )
        return value_loss
    
    
    def select_action(policy:PolicyNetwork, state, device, action_min, action_max):
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

    def train_networks(self,
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
        assert len(rollout_buffer) >= batch_size, f"Rollout buffer length {len(rollout_buffer)} is less than batch size {batch_size}"
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
                    for r, state, next_state, done in zip(
                        reversed(batch_rewards),
                        reversed(batch_states),
                        reversed(batch_next_states),
                        reversed(batch_dones),
                    ):
                        mask = 1 - done.item()
                        next_value = value(next_state).item()
                        next_value = next_value * mask
                        returns.insert(0, g)
                        value_curr_state = value(state).item()
                        delta = r + gamma * next_value - value_curr_state
                        advs.insert(0, delta)
                        g = r + gamma * next_value * mask
    
            returns = torch.tensor(returns, dtype=torch.float32).to(device)
            advs = torch.tensor(advs, dtype=torch.float32).to(device)
            advs = (advs - advs.mean()) / (advs.std() + 1e-10)
    
            # Create mini-batches
            num_samples = len(batch_rewards)
            assert num_samples == batch_size
            num_batches = num_samples // batch_size
            assert num_batches == 1
    
            optimizer.zero_grad()
            policy_loss, entropy_loss = surrogate(
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
    
            value_loss = compute_value_loss(value, batch_states, returns, l1_loss)
    
            # Dynamically adjust vf_coef based on observed training dynamics
            loss_ratio = policy_loss.item() / (
                value_loss.item() + 1e-10
            )  # Adding a small epsilon to avoid division by zero
            # If policy loss is significantly larger, increase vf_coef
            if loss_ratio > 10:
                vf_coef *= 1.1
            # If value loss is significantly larger, decrease vf_coef
            if loss_ratio < 0.1:
                vf_coef *= 0.9
            # Limit vf_coef to a reasonable range to prevent it from becoming too large or too small
            vf_coef = min(max(vf_coef, 0.1), 10)
    
            # Compute total loss and update parameters
            total_loss = policy_loss + entropy_loss + vf_coef * value_loss
    
            total_loss.backward()
    
            # Clip the gradients to avoid exploding gradients
            policy_grad_norm = nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
            value_grad_norm = nn.utils.clip_grad_norm_(value.parameters(), max_grad_norm)
    
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
                wandb.log(
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
                wandb.log(
                    {
                        "Policy/Gradient_Norm": policy_grad_norm,
                        "Value/Gradient_Norm": value_grad_norm,
                        # "Value/Activation_Norm": activation_norm
                    }
                )
                # wandb.log({"Gradients/ValueNet": wandb.Histogram(value.fc1.weight.grad.detach().cpu().numpy())})
                log_std_value = policy.log_std.detach().cpu().numpy()
                wandb.log({"Policy/Log_Std": log_std_value})
            # log the learning rate to wandb
            lrs = {}
            for i, param_group in enumerate(optimizer.param_groups):
                lr = param_group["lr"]
                lrs["learning_rate_{}".format(i)] = lr
                if wandb_log:
                    wandb.log({"LR/LearningRate_{}".format(i): lr})
    
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

    def train(self,
        env_name:str,
        env_config:dict,
        shape_reward:RewardShaper,
        rescaling_rewards: bool,
        scale_states: str,
        epochs:int,
        batch_size:int,
        sgd_iters:int,
        gamma:float,
        optimizer_config:dict,
        hidden_size:int,
        init_type:str,
        clip_param:float,
        vf_coef:float,
        entropy_coef:float,
        max_grad_norm:float,
        use_gae:bool,
        tau:float,
        l1_loss:bool,
        wandb_log:bool,
        verbose:int,
        rollout_buffer_size:int,
        checkpoint_interval:int=-1,
        checkpoint_path:str=None,
        resume_training:bool=False,
        resume_epoch:bool=None,
        report_func:callable=None,
        project:str="continuous-action-ppo",
        seed:int=None
    ):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # Initialize WandB
        if wandb_log:
            config = locals().copy()
            keys_to_delete = [
                "checkpoint_interval",
                "checkpoint_path",
                "resume_training",
                "resume_epoch",
                "report_func",
            ]
            [config.pop(key, None) for key in keys_to_delete]
            wandb.init(project=project, name=env_name, config=config)
        metrics_recorder = MetricsRecorder()
    
        # Set up environment and neural networks
        env = self.setup_env()
        policy, value, optimizer, scheduler = self.setup_networks(
            env,
            optimizer_config,
            hidden_size=hidden_size,
            init_type=init_type,
            device=device
        )
        rollout_buffer = RolloutBuffer(rollout_buffer_size)
    
        if rescaling_rewards:
            reward_scaler = RewardScaler()
        else:
            reward_scaler = None
            click.secho("No reward scaling", fg="red", err=True)
    
        if scale_states is None:
            state_scaler = None
            click.secho("No state scaling", fg="red", err=True)
        else:
            state_scaler = StateScaler(env, sample_size=10000, scale_type=scale_states)
    
        if resume_training:
            last_epoch = CheckpointManager.load_checkpoint(self.policy, self.value, self.optimizer, checkpoint_path, epoch)
        else:
            last_epoch = 0
        
        if shape_reward is None:
            reward_shaper = None
        elif shape_reward == TDRewardShaper:
            reward_shaper = TDRewardShaper(model=value, device=device)
        else:
            reward_shaper = shape_reward()
    
        # Set up rollout generator
        rollout = rollout_with_step(
                    policy,
                    value,
                    env,
                    device,
                    rollout_buffer,
                    state_scaler,
                    reward_shaper,
                    reward_scaler,
                    wandb_log,
                    debug = True if verbose > 1 else False
                )
        # Set up training loop
        episode_rewards = []
        episode_steps = []
        total_iters = epochs * sgd_iters
        train_iters = 0
        rollout_steps = 0
        average_reward = -np.inf
        start = time()
        for epoch in get_progress_iterator(last_epoch, epochs, verbose):
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
            _, _, train_iters = train_networks(
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
                CheckpointManager.save_checkpoint(self.policy, self.value, self.optimizer, epoch, checkpoint_path)
    
            # Average of last 10 episodes or all episodes if less than 10 episodes are available
            average_reward = sum(episode_rewards[-20:]) / len(episode_rewards[-20:])
            if len(episode_rewards) >= 20:
                if wandb_log:
                    WandBLogger.log_rewards(episode_rewards[-20:]) 
                if metrics_recorder:
                    metrics_recorder.record_rewards(episode_rewards[-20:])
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
        print('Training time: ', round((end - start)/60, 2), 'minutes')
    
        metrics_recorder.to_csv()
        if wandb_log:
            wandb.finish()
        print(
            "Training complete",
            "average reward",
            average_reward,
            "total iters",
            total_iters,
        )
        return policy, value, average_reward, train_iters
    