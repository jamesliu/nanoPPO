import torch

def compute_gae(next_value, rewards, masks, values, gamma, tau):
    values = values + [next_value]
    gae = 0
    returns = []
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
        gae = delta + gamma * tau * masks[step] * gae
        returns.insert(0, gae + values[step])
    return returns

def compute_returns_and_advantages_without_gae(
    batch_rewards, batch_states, batch_next_states, batch_dones, value, gamma
):
    returns = []
    advs = []
    g = 0  # Initialize bootstrapped return

    with torch.no_grad():
        for r, state, next_state, done in zip(
            reversed(batch_rewards),
            reversed(batch_states),
            reversed(batch_next_states),
            reversed(batch_dones),
        ):
            mask = 1 - done.item()
            next_value = value(next_state).item()
            next_value = next_value * mask
            value_curr_state = value(state).item()
            delta = r + gamma * next_value - value_curr_state
            advs.insert(0, delta)
            
            g = r + gamma * next_value * mask
            returns.insert(0, g)

    return returns, advs
