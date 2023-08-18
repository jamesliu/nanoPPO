import gym
import torch
import numpy as np
from nanoppo.discrete_action_ppo import PPO
import matplotlib.pyplot as plt
from tqdm import tqdm

def main():
    env = gym.make('CartPole-v1')
    action_dim = env.action_space.n

    state_dim = env.observation_space.shape[0]
    print('action space', env.action_space, action_dim)
    ppo = PPO(state_dim, action_dim)

    episode_rewards = []
    states, actions, rewards, next_states, done_flags, log_probs = [], [], [], [], [], []
    buffer_interval = 20
    print_every = 10

    for episode in tqdm(range(500), desc="Training Progress"):
        state, info = env.reset()
        total_reward = 0
        if episode % buffer_interval == 0:
            states, actions, rewards, next_states, done_flags, log_probs = [], [], [], [], [], []
        for step in range(1, 10000):
            state_tensor = torch.tensor(np.array([state]), dtype=torch.float32)
            action_prob = ppo.policy(state_tensor)
            action = torch.multinomial(action_prob, 1).item()
            next_state, reward, done_flag, truncated, _ = env.step(action)

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            done_flags.append(done_flag)
            log_prob = torch.log(action_prob[0, action])
            log_probs.append(log_prob)

            total_reward += reward

            if done_flag:
                ppo.update(states, actions, rewards, next_states, done_flags, log_probs)
                break
            state = next_state
        episode_rewards.append(total_reward)

        if episode % print_every == 0:
            avg_reward = sum(episode_rewards[-print_every:]) / print_every
            print(f"Episode {episode}: Average Reward over last {print_every} episodes: {avg_reward}")

    print("Initial rewards:", episode_rewards[:20])
    print("Final rewards:", episode_rewards[-20:])

    plt.plot(episode_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('PPO Training Rewards')
    plt.savefig(f'ppo_rewards_{buffer_interval}.png')
    plt.show()

if __name__ == '__main__':
    main()
