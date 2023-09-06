import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.optim as optim
import gym
from scipy import stats
import numpy as np
import wandb
from tqdm import tqdm
import random
import os
import glob
from typing import Tuple
from collections import deque, defaultdict
from pathlib import Path
import ray.tune as tune
import sklearn
import click

# Define the weight initialization function
def init_weights(m, init_type='he'):
    if isinstance(m, nn.Linear):
        if init_type == 'he':
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        elif init_type == 'xavier':
            nn.init.xavier_normal_(m.weight)
        else:
            raise ValueError(f"Initialization type {init_type} not recognized.")
        nn.init.zeros_(m.bias)

class PolicyNetwork(nn.Module):
    def __init__(self, state_size, action_size, init_type):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)
        self.log_std = nn.Parameter(torch.zeros(action_size))

        # Apply the initialization
        self.apply(lambda m: init_weights(m, init_type=init_type))

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        mean = self.fc3(x)
        # Compute std using the exponential of log_std
        std = torch.exp(self.log_std)
        dist = torch.distributions.normal.Normal(mean, std)
        return dist

class ValueNetwork(nn.Module):
    def __init__(self, state_size, hidden_size, init_type):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)

        # Apply the initialization
        self.apply(lambda m: init_weights(m, init_type=init_type))

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def compute_gae(next_value, rewards, masks, values, gamma, tau):
    values = values + [next_value]
    gae = 0
    returns = []
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
        gae = delta + gamma * tau * masks[step] * gae
        returns.insert(0, gae + values[step])
    return returns
 
def surrogate(policy, old_probs, states, actions, advs, clip_param=0.2, entropy_coef=0.01):
    # Policy loss
    dist = policy(states)
    new_probs = dist.log_prob(actions).sum(-1) 
    ratio = torch.exp(new_probs - old_probs) # Importance sampling ratio
    surr1 = ratio * advs
    surr2 = torch.clamp(ratio, 1-clip_param, 1+clip_param) * advs # Trust region clipping
    entropy = dist.entropy().mean()  # Compute the mean entropy of the distribution
    return -torch.min(surr1, surr2).mean() - entropy_coef * entropy  # Add the entropy term to the policy loss

def compute_value_loss(value, states, returns, l1_loss):
    # Compute value loss
    v_pred = value(states).squeeze()
    v_target = returns.squeeze()
    value_loss = F.smooth_l1_loss(v_pred, v_target) if l1_loss else F.mse_loss(v_pred, v_target)
    return value_loss

def select_action(policy, state, device, action_min, action_max, action_std=0.001):
    state = torch.from_numpy(state).float().to(device)
    dist = policy(state)
    action = dist.sample()
    log_prob = dist.log_prob(action).sum(-1)

    # Extract mean and std of the action distribution
    action_mean = dist.mean
    action_std = dist.stddev

    # action = torch.tanh(action)  # Pass the sampled action through the tanh activation function
    # action = action + torch.normal(mean=torch.zeros_like(action), std=action_std)  # Add noise to the action
    #action = action.clamp(-1.0, 1.0)  # Clip the action to the valid range of the action space
    action = action.clamp(action_min, action_max)
    return action.cpu().detach(), log_prob.cpu().detach(), action_mean.cpu().detach(), action_std.cpu().detach()

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, log_prob, reward, next_state, done):
        self.buffer.append((state, action, log_prob, reward, next_state, done))

    def sample(self, batch_size, device):
        states, actions, log_probs, rewards, next_states, dones = zip(*random.sample(self.buffer, batch_size))
        states_t = torch.tensor(np.array(states), device=device)
        actions_t = torch.tensor(np.array(actions), device=device)
        log_probs_t = torch.tensor(np.array(log_probs), device=device)
        rewards_t = torch.tensor(np.array(rewards), device=device)
        next_states_t = torch.tensor(np.array(next_states), device=device)
        dones_t = torch.tensor(np.array(dones), device=device)
        return states_t, actions_t, log_probs_t, rewards_t, next_states_t, dones_t

    def __len__(self):
        return len(self.buffer)

    def clear(self):
        self.buffer.clear()

class MetricsRecorder:
    def __init__(self):
        self.losses = {
            "total_losses": [],
            "policy_losses": [],
            "value_losses": []
        }
        self.actions = {
            "action_means": [],
            "action_stds": []
        }
        self.episode_rewards = {
            "episode_reward_mins": [],
            "episode_reward_means": [],
            "episode_reward_maxs": []
        }
        self.learning = defaultdict(list)

    def record_losses(self, total_loss, policy_loss, value_loss):
        self.losses["total_losses"].append(total_loss)
        self.losses["policy_losses"].append(policy_loss)
        self.losses["value_losses"].append(value_loss)

    def record_actions(self, action_mean, action_std):
        self.actions["action_means"].append(action_mean)
        self.actions["action_stds"].append(action_std)
    
    def record_rewards(self, rewards):
        self.episode_rewards["episode_reward_mins"].append(min(rewards))
        self.episode_rewards["episode_reward_means"].append(sum(rewards)/len(rewards))
        self.episode_rewards["episode_reward_maxs"].append(max(rewards))

    def record_learning(self, lrs):
        for k, v in lrs.items():
            self.learning[k].append(v)

    def to_csv(self):
        import pandas as pd
        
        # Convert dictionaries to DataFrames
        df_losses = pd.DataFrame(self.losses)
        df_actions = pd.DataFrame(self.actions)
        df_rewards = pd.DataFrame(self.episode_rewards)
        df_learning = pd.DataFrame(self.learning)
        
        # Save to separate CSVs
        df_losses.to_csv('losses_metrics.csv', index=False)
        df_actions.to_csv('actions_metrics.csv', index=False)
        df_rewards.to_csv('rewards_metrics.csv', index=False)
        df_learning.to_csv('learning_metrics.csv', index=False)

def save_checkpoint(policy, value, optimizer, epoch, checkpoint_path):
    # Create checkpoint directory if it does not exist
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    checkpoint = {'epoch': epoch,
                  'policy_state_dict': policy.state_dict(),
                  'value_state_dict': value.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict()}
    torch.save(checkpoint, f"{checkpoint_path}/checkpoint_epoch{epoch}.pt")

def load_checkpoint(policy, value, optimizer, checkpoint_path, epoch=None):
    # Find the latest checkpoint file
    if epoch is None:
        checkpoint_files = sorted(glob.glob(f"{checkpoint_path}/checkpoint_epoch*.pt"))
        if len(checkpoint_files) == 0:
            raise ValueError("No checkpoint found in the specified directory.")
        checkpoint_file = checkpoint_files[-1]
    else:
        checkpoint_file = f"{checkpoint_path}/checkpoint_epoch{epoch}.pt"

    checkpoint = torch.load(checkpoint_file)
    policy.load_state_dict(checkpoint['policy_state_dict'])
    value.load_state_dict(checkpoint['value_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    return epoch

def setup_env(env_name, env_config,seed=101):
    if env_config:
        env = gym.make(env_name, config=env_config)
    else:
        env = gym.make(env_name)
    return env

def setup_networks(env, optimizer_config, hidden_size, init_type, device, epochs, attention=False):
    if isinstance(env.observation_space, gym.spaces.Dict):
        observation_space = env.observation_space.spaces['obs']
    else:
        observation_space = env.observation_space
    if isinstance(env.action_space, gym.spaces.Box):
        action_dim = env.action_space.shape[0]
    else:
        action_dim = env.action_space.n
    
    if attention:
        # TODO: add attention network
        # policy = PolicyAttentionNetwork(observation_space.shape[-1], action_dim).to(device)
        # value = ValueAttentionNetwork(observation_space.shape[-1]).to(device)    
        pass
    else:
        policy = PolicyNetwork(observation_space.shape[0], action_dim, init_type=init_type).to(device)
        value = ValueNetwork(observation_space.shape[0], hidden_size=hidden_size, init_type=init_type).to(device)

    optimizer = optim.Adam([
        {'params': policy.parameters()},
        {'params': value.parameters()}], 
        lr = optimizer_config['lr'], 
        betas=(optimizer_config['beta1'], optimizer_config['beta2']), 
        eps=optimizer_config['epsilon'], 
        weight_decay=optimizer_config['weight_decay']
        )
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    return policy, value, optimizer, scheduler

def get_progress_iterator(last_epoch, epochs, verbose):
    if verbose:
        progress_iterator = tqdm(range(last_epoch, last_epoch + epochs))
    else:
        progress_iterator = range(last_epoch, last_epoch + epochs)
    return progress_iterator

def log_rewards(rewards):
    # Log the rewards during training
    wandb.log({
        "Reward/Min": min(rewards),
        "Reward/Mean": sum(rewards) / len(rewards),
        "Reward/Max": max(rewards)})

   
class RewardScaler:
    def __init__(self):
        self.running_mean = 0
        self.running_var = 0
        self.count = 0

    def update(self, rewards):
        """Update running mean and variance."""
        self.count += len(rewards)
        batch_mean = np.mean(rewards)
        batch_var = np.var(rewards)

        # Welford's online algorithm for updating mean and variance
        delta = batch_mean - self.running_mean
        self.running_mean += delta * len(rewards) / self.count
        self.running_var += delta * (batch_mean - self.running_mean)

    def scale_rewards(self, rewards):
        """Scale rewards using the running statistics and update the statistics."""
        self.update(rewards)
        if self.running_var == 0:
            return rewards - self.running_mean
        return (rewards - self.running_mean) / (np.sqrt(self.running_var) + 1e-8)

class StateScaler:
    def __init__(self, env, sample_size=10000, by_sample=False):
        self.env = env
        self.by_sample = by_sample
        if self.by_sample:
            self.scaler = self._init_scaler(env, sample_size)
        
    def _init_scaler(self, env, sample_size):
        state_space_samples = np.array([env.observation_space.sample() for _ in range(sample_size)])
        scaler = sklearn.preprocessing.StandardScaler()
        scaler.fit(state_space_samples)
        return scaler

    def scale_state(self, state):
        if self.by_sample:
            scaled = self.scaler.transform([state])
            return scaled[0]  # Return a 1D array instead of 2D
        else:
            #return (state - self.env.observation_space.low) / (self.env.observation_space.high - self.env.observation_space.low)
            # -1 to 1 scaling
            return 2 * (state - self.env.observation_space.low) / (self.env.observation_space.high - self.env.observation_space.low) - 1

# Updating the rollout function to update the RewardScaler
def rollout_with_scaling(policy, env, device, replay_buffer, state_scaler, reward_scaler, wandb_log, metrics_recorder:MetricsRecorder):
    policy.eval()
    state, info = env.reset()
    if isinstance(state, dict):
        state = state['obs']
    scaled_state = state_scaler.scale_state(state)
    done = False
    total_rewards = 0
    steps = 0
    unscaled_rewards = []
    
    # Lists to store states, actions, and log_probs for the entire episode
    states_list = []
    actions_list = []
    log_probs_list = []
    
    while not done:
        with torch.no_grad():
            action, log_prob, action_mean, action_std = select_action(policy, scaled_state, device, env.action_space.low[0], env.action_space.high[0])
            if wandb_log:
                wandb.log({"Action/Mean": action_mean.item(), "Action/Std": action_std.item()})
            if metrics_recorder:
                metrics_recorder.record_actions(action_mean.item(), action_std.item())
        action = action.numpy()
        log_prob = log_prob.numpy()
        
        # Store state, action, and log_prob
        states_list.append(state)
        actions_list.append(action)
        log_probs_list.append(log_prob)
        
        next_state, reward, done, truncated, info = env.step(action)
        done = done or truncated
        if isinstance(next_state, dict):
            next_state = next_state['obs']
        scaled_next_state = state_scaler.scale_state(next_state)

        unscaled_rewards.append(reward)
        scaled_state = scaled_next_state
        total_rewards += reward
        steps += 1

    # Scale rewards
    if reward_scaler is None:
        scaled_rewards = unscaled_rewards
    else:
        scaled_rewards = reward_scaler.scale_rewards(unscaled_rewards)

    for i in range(len(unscaled_rewards)):
        replay_buffer.push(states_list[i], actions_list[i], log_probs_list[i], scaled_rewards[i], next_state, done)
    
    policy.train()
    return total_rewards

def eval(policy, env_name, state_scaler, env_config = {}, device='cpu', epochs=20, render=False):
    policy.eval()
    policy.to(device)
    if env_config: 
        env = gym.make(env_name, config=env_config)
    else:
        env = gym.make(env_name)
    action_min = env.action_space.low[0]
    action_max = env.action_space.high[0]
    episode_rewards = []
    actions = []
    for epoch in range(epochs):
        state, info = env.reset()
        if isinstance(state, dict):
            state = state['obs'] 
        scaled_state = state_scaler.scale_state(state)
        done = False
        total_reward = 0
        steps = 0
        while not done:
            with torch.no_grad():
                action, _ = select_action(policy, scaled_state, device, action_min, action_max)
            action = action.numpy()
            actions.append(action)
            next_state, reward, done, truncated, info = env.step(action)
            done = done or truncated
            if isinstance(next_state, dict):
                next_state = next_state['obs']
            scaled_next_state = state_scaler.scale_state(next_state)
            if render:
                env.render()
            scaled_state = scaled_next_state
            total_reward += reward
            steps += 1
        else:
            episode_rewards.append(total_reward)
    policy.train()
    return episode_rewards, actions

def train_networks(iter_num, policy, value, optimizer, scheduler, replay_buffer, device,
                  batch_size, sgd_iters, gamma, clip_param, vf_coef, entropy_coef, max_grad_norm, use_gae, tau, l1_loss, wandb_log, metrics_recorder:MetricsRecorder):
    assert(len(replay_buffer) >= batch_size)
    for sgd_iter in range(sgd_iters):
        # Sample from the replay buffer
        batch_states, batch_actions, batch_probs, batch_rewards, batch_next_states, batch_dones = replay_buffer.sample(batch_size, device=device)
    
        # Compute Advantage and Returns
        returns = []
        advs = []
        g = 0
        # Compute returns and advantages from replay buffer samples out of order
        with torch.no_grad():
            if use_gae:
                # Compute Advantage using GAE and Returns
                values = value(batch_states).squeeze().tolist()
                next_value = value(batch_next_states[-1]).item()
                masks = [1 - done.item() for done in batch_dones]
                returns = compute_gae(next_value, batch_rewards, masks, values, gamma, tau)
                advs = [ret - val for ret, val in zip(returns, values)]
            else:
                for r, state, next_state, done in zip(reversed(batch_rewards), reversed(batch_states), reversed(batch_next_states), reversed(batch_dones)):
                    mask = 1 - done.item()
                    next_value = value(next_state).item()
                    next_value = next_value * mask
                    returns.insert(0, g)
                    value_curr_state = value(state).item()
                    delta = r + gamma * next_value - value_curr_state
                    advs.insert(0, delta)
                    #g = r + gamma * next_value * mask
        
        returns = torch.tensor(returns, dtype=torch.float32).to(device)
        advs = torch.tensor(advs, dtype=torch.float32).to(device)
        advs = (advs - advs.mean()) / (advs.std() + 1e-10)
    
        # Create mini-batches
        num_samples = len(batch_rewards)
        assert(num_samples == batch_size)
        num_batches = num_samples // batch_size
        assert(num_batches == 1)
    
        optimizer.zero_grad()
        policy_loss = surrogate(policy, old_probs=batch_probs, states=batch_states,
                                actions=batch_actions, advs=advs, clip_param=clip_param,
                                entropy_coef=entropy_coef)

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

        value_loss = compute_value_loss(value, batch_states, returns, l1_loss)
        # Compute total loss and update parameters
        total_loss = policy_loss + vf_coef * value_loss

        total_loss.backward()

        # Clip the gradients to avoid exploding gradients
        value_grad_norm = nn.utils.clip_grad_norm_(value.parameters(), max_grad_norm)

        # compute activation norm
        # remove the forward hooks
        for hook in hooks:
            hook.remove()
        # compute the mean activation norm for the value network
        activation_norm = sum(activation_norms) / len(activation_norms)

        optimizer.step()
        scheduler.step()
        
        if wandb_log:
            # Log the losses and gradients to WandB
            wandb.log({"iteration":iter_num,
                      "Loss/Total": total_loss.item(), 
                      "Loss/Policy": policy_loss.item(),
                      "Loss/Value": value_loss.item(),})
            #wandb.log({"Gradients/PolicyNet": wandb.Histogram(policy.fc1.weight.grad.detach().cpu().numpy())})
            wandb.log({"Gradient_Norm/Value_Net": value_grad_norm,
                        "Activation_Norm/Value_Net": activation_norm})
            #wandb.log({"Gradients/ValueNet": wandb.Histogram(value.fc1.weight.grad.detach().cpu().numpy())})
        # log the learning rate to wandb
        lrs = {}
        for i, param_group in enumerate(optimizer.param_groups):
            lr = param_group['lr']
            lrs['learning_rate_{}'.format(i)] = lr
            if wandb_log:
                wandb.log({'learning_rate_{}'.format(i): lr})

        if metrics_recorder:
            metrics_recorder.record_losses(total_loss.item(), policy_loss.item(), value_loss.item())
            metrics_recorder.record_learning(lrs)

        iter_num += 1
    return policy, value, iter_num,

def train(env_name, env_config, attention:bool, on_policy:bool, rescaling_rewards:bool, num_rollout, epochs, batch_size, sgd_iters, gamma,
          optimizer_config, hidden_size, init_type, clip_param, vf_coef, entropy_coef, max_grad_norm, use_gae, tau, l1_loss,
          wandb_log, verbose, replay_buffer_size, replay_start_size,
          checkpoint_interval=-1, checkpoint_path=None, resume_training=False, resume_epoch=None, tune_report=False):
    print('Training PPO agent on environment: ', env_name, ' for ', epochs, ' epochs', ' and num rollout: ', num_rollout, ' and replay buffer size: ', replay_buffer_size, ' and replay start size: ', replay_start_size,
          ' with batch size: ', batch_size, ' and sgd iters: ', sgd_iters, ' and gamma: ', gamma, ' and optimizer config: ', optimizer_config,
          ' and hidden size: ', hidden_size, ' and init type: ', init_type, 
          ' and clip param: ', clip_param, ' and vf coef: ', vf_coef, ' and entropy coef: ', entropy_coef, ' and max grad norm: ', max_grad_norm,
          'and tau: ', tau, ' and l1 loss: ', l1_loss, ' and wandb log: ', wandb_log, ' and verbose: ', verbose,)
 
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Initialize WandB
    if wandb_log:
        wandb.init(project="myppo_continuous_action", name=env_name)
    metrics_recorder = MetricsRecorder()

    # Set up environment and neural networks
    env = setup_env(env_name, env_config)
    policy, value, optimizer, scheduler = setup_networks(env, optimizer_config, hidden_size=hidden_size, init_type=init_type, device=device, epochs=epochs, attention=attention)
    replay_buffer = ReplayBuffer(replay_buffer_size)
    if rescaling_rewards:
        reward_scaler = RewardScaler()
    else:
        reward_scaler = None
        click.secho('No reward scaling', fg='red', err=True)
    
    state_scaler = StateScaler(env, sample_size=10000, by_sample=False)

    if resume_training:
        last_epoch = load_checkpoint(policy, value, optimizer, checkpoint_path, epoch=resume_epoch)
    else:
        last_epoch = 0
    
    # Set up training loop
    episode_rewards = []
    total_iters = epochs * sgd_iters
    iter_num = 0
    for epoch in get_progress_iterator(last_epoch, epochs, verbose):
        if on_policy:
            replay_buffer.clear() # clear the replay buffer, all data is from the current policy
            for r in range(num_rollout):
                total_rewards = rollout_with_scaling(policy, env, device, replay_buffer, state_scaler, reward_scaler, wandb_log, metrics_recorder)
                episode_rewards.append(total_rewards)
        else: 
            # the replay buffer is filled with data from the current policy or old policies
            # improve sampling efficiency; however, keep replay buffer size reasonable small so that there is no much drift in the data distribution
            rollout_with_scaling(policy, env, device, replay_buffer, state_scaler, reward_scaler, wandb_log, metrics_recorder)
            episode_rewards.append(total_rewards)
            if len(replay_buffer) < replay_start_size:
                print('epoch', epoch, 'rollout', len(replay_buffer))
                continue
        if wandb_log:
            log_rewards(episode_rewards[-100:])
        if metrics_recorder:
            metrics_recorder.record_rewards(episode_rewards[-100:])
        _,_, iter_num = train_networks(iter_num, policy, value, optimizer, scheduler, replay_buffer, device, 
                       batch_size, sgd_iters, gamma, clip_param, vf_coef, entropy_coef, max_grad_norm, 
                       use_gae=use_gae, tau=tau, l1_loss=l1_loss, wandb_log = wandb_log, metrics_recorder=metrics_recorder)
        if checkpoint_interval > 0 and ((epoch + 1) % checkpoint_interval) == 0:
            save_checkpoint(policy, value, optimizer, epoch, checkpoint_path)
        
        # Average of last 10 episodes or all episodes if less than 10 episodes are available
        average_reward = sum(episode_rewards[-30:]) / min(30, len(episode_rewards)) 
        if tune_report:
            tune.report(mean_reward=average_reward)  # Reporting the reward to Tune
        else:
            if verbose or epoch % 10 == 0:
                print('epoch', epoch, 'average reward', average_reward, 'rollout episodes', len(episode_rewards))
    
    metrics_recorder.to_csv()
    if wandb_log:
        wandb.finish()
    return policy, value, average_reward

env_name = 'MountainCarContinuous-v0'
#env_name = 'Pendulum-v1'

config = {
    'env_name':env_name, 'env_config': None, 'on_policy': True, 'attention':False, 'rescaling_rewards': True, 'use_gae':True, 'l1_loss':False, 
    'num_rollout': 2, 'replay_start_size': 100, 'replay_buffer_size': 2048, 'hidden_size':64, 'init_type':'xavier',
    'epochs': 500, 'sgd_iters': 30, 'batch_size': 512, 'gamma': 0.99, 
    'clip_param': 0.290472058684371, 'max_grad_norm': 0.9426866418790952, 'vf_coef': 1.3438889498747817, 
    'entropy_coef': 0.000591524925410050, 'tau': 0.960078265158954, 
    'wandb_log': True, 'verbose':True, 'checkpoint_interval':100, 'checkpoint_path': Path('checkpoints') / 'myppo_onpolicy' / env_name
}

optimizer_config = { 
    "lr": 0.0009682521671192982,
    "beta1": 0.9,
    "beta2": 0.98,
    "epsilon": 1e-8,
    "weight_decay": 0.0009235827912560286
}
config.update({'optimizer_config':optimizer_config})
    
if __name__ == '__main__':
    best_config = {'num_rollout': 5, 'init_type': 'xavier', 'lr': 0.0009682521671192982, 'weight_decay': 0.0009235827912560286, 'sgd_iters': 30, 'replay_buffer_size': 2048, 'batch_size': 512, 'l1_loss': False, 'clip_param': 0.290472058684371, 'max_grad_norm': 0.9426866418790952, 'vf_coef': 1.3438889498747817, 'entropy_coef': 0.0005915249254100502, 'tau': 0.960078265158954} 
    for k,v in best_config.items():
        if k in optimizer_config.keys():
            print(f"setting optimizer {k} to {v}")
            config['optimizer_config'][k] = v
        else:
            print(f"setting {k} to {v}")
            config[k] = v
    policy, value, average_reward = train(**config, tune_report=False)
    print('train', 'average reward', average_reward)


