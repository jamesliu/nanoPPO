import torch
from nanoppo.ppo_agent_with_network import PPOAgent
from nanoppo.envs.point_mass2d import PointMass2DEnv 
from nanoppo.envs.point_mass1d import PointMass1DEnv

# Setting up the environment and the agent
env = PointMass2DEnv()
#env = PointMass1DEnv()
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
print('state_dim', state_dim)
print('action_dim', action_dim)
n_latent_var = 64
lr = 0.0005
betas = (0.9, 0.999)
gamma = 0.99
K_epochs = 4
eps_clip = 0.2
max_timesteps = 200
update_timestep = 200
log_interval = 20
max_episodes = 1000  # Modify this value based on how many episodes you want to train

ppo = PPOAgent(state_dim, action_dim, n_latent_var, lr, betas, gamma, K_epochs, eps_clip)
print(lr, betas)

def compute_gae(next_value, rewards, masks, values, gamma=0.99, tau=0.95):
    values = values + [next_value]
    gae = 0
    returns = []
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
        gae = delta + gamma * tau * masks[step] * gae
        returns.insert(0, gae + values[step])
    torch_returns = torch.tensor(returns, dtype=torch.float32)
    return torch_returns

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

ppo_memory = PPOMemory()

# Training loop
time_step = 0
best_reward = float('-inf')
avg_length = 0
for episode in range(1, max_episodes + 1):
    state, info = env.reset()
    total_reward = 0
    state = torch.FloatTensor(state)
    for t in range(max_timesteps):
        action, log_prob = ppo.policy.act(state)
        next_state, reward, done, truncated, _ = env.step(action.numpy())
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
            returns = compute_gae(next_value, ppo_memory.rewards, ppo_memory.is_terminals, values)

            states, actions, log_probs, next_states, rewards, dones = ppo_memory.get()
            ppo.update(states, actions, returns=returns, next_states=next_states, dones=dones)
            ppo_memory.clear()
            time_step = 0
        if done:
            break

    avg_length += t + 1

    # Logging
    if episode % log_interval == 0:
        avg_length = int(avg_length / log_interval)
        avg_reward = int(total_reward / log_interval)
        print('Episode {} \t avg length: {} \t reward: {}'.format(episode, avg_length, avg_reward))
        avg_length = 0
        total_reward = 0
        if avg_reward > best_reward:
            best_reward = avg_reward
            ppo.save("best_weights.pth")
            print("Saved best weights!")

# Load the best weights
ppo.load("best_weights.pth")
print("Loaded best weights!")
