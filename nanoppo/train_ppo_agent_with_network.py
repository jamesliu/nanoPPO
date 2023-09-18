import torch
import os
import pickle
from nanoppo.ppo_agent_with_network import PPOAgent
from nanoppo.normalizer import Normalizer
from nanoppo.ppo_utils import compute_gae
from nanoppo.envs.point_mass2d import PointMass2DEnv 
from nanoppo.envs.point_mass1d import PointMass1DEnv
from nanoppo.environment_manager import EnvironmentManager

# Setting up the environment and the agent
#env = PointMass1DEnv()
#env = PointMass2DEnv()
env_name = "MountainCarContinuous-v0"
env = EnvironmentManager(env_name).setup_env()

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
print('state_dim', state_dim)
print('action_dim', action_dim)
n_latent_var = 128
lr = 0.0005
betas = (0.9, 0.999)
gamma = 0.99
tau = 0.95
K_epochs = 4
eps_clip = 0.2
max_timesteps = 1000
update_timestep = 200
log_interval = 20
max_episodes = 1000  # Modify this value based on how many episodes you want to train

print('env', env_name)
model_file = f"{env_name}_ppo.pth"
metrics_file = f"{env_name}_metrics.pkl"

"""
def compute_gae(next_value, rewards, masks, values, gamma=0.99, tau=0.95):
    values = values + [next_value]
    gae = 0
    returns = []
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
        gae = delta + gamma * tau * masks[step] * gae
        returns.insert(0, gae + values[step])
    #torch_returns = torch.tensor(returns, dtype=torch.float32)
    return returns
"""

# Memory for PPO
class PPOMemory:
    def __init__(self):
        self.states = []
        self.actions = []
        self.logprobs = []
        self.next_states = [] 
        self.rewards = []
        self.is_terminals = []

    def clear(self):
        del self.states[:]
        del self.actions[:]
        del self.logprobs[:]
        del self.next_states[:]
        del self.rewards[:]
        del self.is_terminals[:]

    def append(self, state, action, logprob, next_state, reward, is_terminal):
        assert(len(self.states) == len(self.actions) == len(self.logprobs) == len(self.next_states) == len(self.rewards) == len(self.is_terminals))
        self.states.append(state)
        self.actions.append(action)
        self.logprobs.append(logprob)
        self.next_states.append(next_state) 
        self.rewards.append(torch.tensor(reward, dtype=torch.float32))  # Convert to tensor here
        self.is_terminals.append(torch.tensor(is_terminal, dtype=torch.float32))  # Convert to tensor here

    def get(self):
        return torch.stack(self.states), torch.stack(self.actions), torch.stack(self.logprobs), torch.stack(self.next_states), torch.stack(self.rewards), torch.stack(self.is_terminals)

# Initialize a normalizer with the dimensionality of the state
state_normalizer = Normalizer(state_dim)
reward_normalizer = Normalizer(1)
ppo = PPOAgent(state_dim, action_dim, n_latent_var, lr, betas, gamma, K_epochs, eps_clip, state_normalizer, 
               action_low=env.action_space.low, action_high=env.action_space.high)
print(lr, betas)

# Load the best weights
if os.path.exists(model_file):
    metrics = pickle.load(open(metrics_file, 'rb'))
    best_reward = metrics['best_reward']
    start_episode = metrics['episode'] + 1
    ppo.load(model_file)
    print("Loaded best weights!")
else:
    best_reward = float('-inf')
    start_episode = 1
print("best_reward", best_reward)
print("start_episode", start_episode)
print("log_interval", log_interval)

ppo_memory = PPOMemory()

# Training loop
time_step = 0
avg_length_list = []
cumulative_reward_list = []  # Initialize cumulative reward
for episode in range(start_episode, max_episodes + start_episode):
    state, info = env.reset()
    state_normalizer.observe(state)
    state = state_normalizer.normalize(state)

    total_reward = 0
    state = torch.FloatTensor(state)
    for t in range(max_timesteps):
        action, log_prob = ppo.policy.act(state)
        action_np = action.detach().numpy()
        next_state, reward, done, truncated, _ = env.step(action_np)
        state_normalizer.observe(next_state)
        next_state = state_normalizer.normalize(next_state)

        total_reward += reward
        next_state = torch.FloatTensor(next_state)
        ppo_memory.append(state, action, log_prob, next_state, reward, done)

        state = next_state
        time_step += 1

        # update if it's time
        if time_step % update_timestep == 0:
            # Get state values for all states
            next_state = torch.FloatTensor(next_state)
            next_value = ppo.policy.get_value(next_state).detach().item()
            values = [ppo.policy.get_value(torch.FloatTensor(state)).item() for state in ppo_memory.states]
            returns = compute_gae(next_value, ppo_memory.rewards, ppo_memory.is_terminals, values, gamma=gamma, tau=tau)
            torch_returns = torch.tensor(returns, dtype=torch.float32)

            states, actions, log_probs, next_states, rewards, dones = ppo_memory.get()
            ppo.update(states, actions, returns=torch_returns, next_states=next_states, dones=dones)
            ppo_memory.clear()
            time_step = 0
        if done:
            break

    avg_length_list.append(t + 1)

    cumulative_reward_list.append(total_reward) 

    # Logging
    if episode % log_interval == 0:
        sample_length = len(avg_length_list)
        avg_length = int(sum(avg_length_list) / sample_length)
        avg_reward = float(sum(cumulative_reward_list) / sample_length)
        print('Episode {} \t sample length:{} avg length: {} \t avg reward: {} best reward: {}'.format(episode, sample_length, avg_length, avg_reward, best_reward))
        avg_length_list = []
        cumulative_reward_list = []  # Reset cumulative reward after logging
        if avg_reward > best_reward:
            print('avg_reward', avg_reward, '> best_reward', best_reward)
            best_reward = avg_reward
            metrics = {'best_reward': best_reward, 'episode':episode}
            pickle.dump(metrics, open(metrics_file, 'wb'))
            ppo.save(model_file)
            print("Saved best weights!", model_file, metrics_file)

# Load the best weights
ppo.load(model_file)
metrics = pickle.load(open(metrics_file, 'rb'))
print("Loaded best weights!")
best_reward = metrics['best_reward']
episode = metrics['episode']
print("best_reward", best_reward, 'episode', episode)
