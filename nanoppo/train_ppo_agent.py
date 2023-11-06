import torch
import os
import pickle
import click
import wandb
from nanoppo.continuous_action_ppo import PPOAgent
from nanoppo.normalizer import Normalizer
from nanoppo.ppo_utils import compute_gae
from nanoppo.environment_manager import EnvironmentManager
from nanoppo.ppo_utils import get_grad_norm

# Memory for PPO
class PPOMemory:
    def __init__(self, device):
        self.states = []
        self.actions = []
        self.logprobs = []
        self.next_states = []
        self.rewards = []
        self.is_terminals = []
        self.device = device

    def clear(self):
        del self.states[:]
        del self.actions[:]
        del self.logprobs[:]
        del self.next_states[:]
        del self.rewards[:]
        del self.is_terminals[:]

    def append(self, state, action, logprob, next_state, reward, is_terminal):
        assert (
            len(self.states)
            == len(self.actions)
            == len(self.logprobs)
            == len(self.next_states)
            == len(self.rewards)
            == len(self.is_terminals)
        )
        self.states.append(state)
        self.actions.append(action)
        self.logprobs.append(logprob)
        self.next_states.append(next_state)
        self.rewards.append(
            torch.tensor(reward, dtype=torch.float32).to(self.device)
        )  # Convert to tensor here
        self.is_terminals.append(
            torch.tensor(is_terminal, dtype=torch.float32).to(self.device)
        )  # Convert to tensor here

    def get(self):
        return (
            torch.stack(self.states),
            torch.stack(self.actions),
            torch.stack(self.logprobs),
            torch.stack(self.next_states),
            torch.stack(self.rewards),
            torch.stack(self.is_terminals),
        )


def train_agent(
    env_name,
    env_config = None,
    max_episodes=500,
    stop_reward = None,
    policy_class = None,
    policy_lr=0.0005,
    value_lr=0.0005,
    betas=(0.9, 0.999),
    n_latent_var=128,
    gamma=0.99,
    tau=0.95,
    K_epochs=4,
    eps_clip=0.2,
    vl_coef=0.5,
    el_coef=0.001,
    max_timesteps=2000,
    update_timestep=200,
    checkpoint_dir="checkpoints",
    checkpoint_interval=-1,
    log_interval=-1,
    lr_scheduler=None,
    wandb_log=False,
    device=None,
    debug=False,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Setting up the environment and the agent
    env = EnvironmentManager(env_name, env_config).setup_env()
    state_dim = env.observation_space.shape[-1]
    action_dim = env.action_space.shape[0]
    print("state_dim", state_dim)
    print("action_dim", action_dim)

    checkpoint_path = os.path.join(checkpoint_dir, env_name)
    os.makedirs(checkpoint_path, exist_ok=True)
    print("env", env_name)
    model_file = f"{checkpoint_path}/models.pth"
    metrics_file = f"{checkpoint_path}/metrics.pkl"
    if wandb_log:
        wandb.init(
            project="nanoPPO",
            name=env_name,
            config={
                "policy_lr": policy_lr,
                "value_lr": value_lr,
                "betas": betas,
                "gamma": gamma,
                "tau": tau,
                "K_epochs": K_epochs,
                "eps_clip": eps_clip,
            },
        )

    # Initialize a normalizer with the dimensionality of the state
    state_normalizer = Normalizer(dim=env.observation_space.shape)
    ppo = PPOAgent(
        state_dim,
        action_dim,
        n_latent_var,
        policy_class,
        policy_lr,
        value_lr,
        betas,
        gamma,
        K_epochs,
        eps_clip,
        state_normalizer,
        action_low=env.action_space.low,
        action_high=env.action_space.high,
        vl_coef=vl_coef,
        el_coef=el_coef,
        lr_scheduler=lr_scheduler,
        device=device,
        wandb_log=wandb_log,
        debug = debug
    )
    print(policy_lr, value_lr, betas)
    print('ppo use device', ppo.device)

    # Load the best weights
    if os.path.exists(model_file):
        metrics = pickle.load(open(metrics_file, "rb"))
        best_reward = metrics["best_reward"] 
        start_episode = metrics["episode"] + 1
        ppo.load(model_file)
        print("Loaded best weights!", model_file, metrics_file)
        if stop_reward and (best_reward > stop_reward):
            print("Skipping Training: best_reward", best_reward, "> stop_reward", stop_reward)
            return ppo, model_file, metrics_file
    else:
        best_reward = float("-inf")
        start_episode = 1
    print("best_reward", best_reward)
    print("start_episode", start_episode)
    print("log_interval", log_interval)

    ppo_memory = PPOMemory(device= device)

    # Training loop
    time_step = 0
    avg_length_list = []
    cumulative_reward_list = []  # Initialize cumulative reward
    for episode in range(start_episode, max_episodes + start_episode):
        state, info = env.reset()
        state_normalizer.observe(state)
        state = state_normalizer.normalize(state)

        total_reward = 0
        state = torch.FloatTensor(state).to(device)
        for t in range(max_timesteps):
            action, log_prob = ppo.policy.act(state)

            action_np = action.detach().cpu().numpy()
            if len(action_np.shape) > 1:
                # (1, action_dim) -> (action_dim,)
                action_np = action_np.squeeze()
            next_state, reward, done, truncated, _ = env.step(action_np)
            state_normalizer.observe(next_state)
            next_state = state_normalizer.normalize(next_state)

            total_reward += reward
            next_state = torch.FloatTensor(next_state).to(device)
            ppo_memory.append(state, action, log_prob, next_state, reward, done)

            state = next_state
            time_step += 1

            # update if it's time
            if time_step % update_timestep == 0:
                try:
                    # Get state values for all states
                    next_value = ppo.policy.get_value(next_state).detach().item()
                    values = [
                        ppo.policy.get_value(state).item()
                        for state in ppo_memory.states
                    ]
                    masks = [1 - terminal.item() for terminal in ppo_memory.is_terminals]
                    returns = compute_gae(
                        next_value, ppo_memory.rewards, masks, values, gamma=gamma, tau=tau
                    )
                    torch_returns = torch.tensor(returns, dtype=torch.float32).to(device)
    
                    (
                        states,
                        actions,
                        log_probs,
                        next_states,
                        rewards,
                        dones,
                    ) = ppo_memory.get()
                    ppo.update(
                        states,
                        actions,
                        returns=torch_returns,
                        next_states=next_states,
                        dones=dones,
                    )
                    ppo_memory.clear()
                    time_step = 0
                except Exception as e:
                    print("ppo.update error")
                    print(e)
                    breakpoint()
                    raise e
            if done or truncated:
                break
        avg_length_list.append(t + 1)

        cumulative_reward_list.append(total_reward)

        num_cumulative_rewards = len(cumulative_reward_list)
        avg_reward = float(sum(cumulative_reward_list[-30:]) / 30)
        avg_length = int(sum(avg_length_list) / len(avg_length_list))
        action_mu_grad_norm = get_grad_norm(ppo.policy.action_mu.parameters())
        action_log_std_grad_norm = get_grad_norm(ppo.policy.action_log_std.parameters())
        value_grad_norm = get_grad_norm(ppo.policy.value_layer.parameters())
        # Logging
        if log_interval > 0 and (episode % log_interval == 0):
            sample_length = len(avg_length_list)
            avg_length = int(sum(avg_length_list) / sample_length)
            print(
                (
                    "Episode {} \t samples:{} avg steps: {} \t avg reward: {:.3f} \t best reward: {:.3f} \t"
                    "action_mu_grad_norm: {:.2f} \t action_log_std_grad_norm: {:.2f} \t value_grad_norm: {:.2f} \t"
                    "num cumulative rewards: {}"
                ).format(
                    episode,
                    sample_length,
                    avg_length,
                    avg_reward,
                    best_reward,
                    action_mu_grad_norm,
                    action_log_std_grad_norm,
                    value_grad_norm,
                    num_cumulative_rewards
                )
            )

        if (checkpoint_interval > 0 and (avg_reward > best_reward) and (num_cumulative_rewards > 30)):
            print("avg_reward", avg_reward, "> best_reward", best_reward)
            best_reward = avg_reward
            metrics = {"train_reward": avg_reward, "best_reward": best_reward, "episode": episode, "stop_reward":stop_reward}
            pickle.dump(metrics, open(metrics_file, "wb"))
            ppo.save(model_file)
            print("Saved best weights!", best_reward, model_file, metrics_file)
        
        if stop_reward and (avg_reward > stop_reward) and (num_cumulative_rewards > 30):
            print("avg_reward", avg_reward, "> stop_reward", stop_reward)
            best_reward = avg_reward
            metrics = {"train_reward":avg_reward, "best_reward": best_reward, "episode": episode, "stop_reward":stop_reward}
            pickle.dump(metrics, open(metrics_file, "wb"))
            ppo.save(model_file)
            print("Saved best weights!", best_reward, model_file, metrics_file)
            break

        if wandb_log:
            wandb.log(
                {
                    "avg_reward": avg_reward,
                    "best_reward": best_reward,
                    "avg_length": avg_length,
                    "action_mu_grad_norm": action_mu_grad_norm,
                    "action_log_std_grad_norm": action_log_std_grad_norm,
                    "value_grad_norm": value_grad_norm,
                }
            )
    if wandb_log:
        wandb.finish()
    return ppo, model_file, metrics_file


@click.command()
@click.option(
    "--env_name",
    default="PointMass2D-v0",
    type=click.Choice(
        ["PointMass1D-v0", "PointMass2D-v0", "Pendulum-v1", "MountainCarContinuous-v0"]
    ),
)
@click.option("--max_episodes", default=100, help="Number of training episodes.")
@click.option("--policy_lr", default=0.0005, help="Learning rate for policy network.")
@click.option("--value_lr", default=0.0005, help="Learning rate for value network.")
@click.option("--vl_coef", default=0.5, help="Value function coefficient.")
@click.option("--checkpoint_dir", default="checkpoints", help="Path to checkpoint.")
@click.option("--checkpoint_interval", default=100, help="Checkpoint interval.")
@click.option("--log_interval", default=10, help="Logging interval.")
@click.option(
    "--wandb_log", is_flag=True, default=False, help="Flag to log results to wandb."
)
def cli(
    env_name,
    max_episodes,
    policy_lr,
    value_lr,
    vl_coef,
    checkpoint_dir,
    checkpoint_interval,
    log_interval,
    wandb_log,
):
    ppo, model_file, metrics_file = train_agent(
        env_name=env_name,
        max_episodes=max_episodes,
        policy_lr=policy_lr,
        value_lr=value_lr,
        vl_coef=vl_coef,
        checkpoint_dir=checkpoint_dir,
        checkpoint_interval=checkpoint_interval,
        log_interval=log_interval,
        wandb_log=wandb_log,
        device='cpu'
    )
    # Load the best weights
    ppo.load(model_file)
    print("Loaded best weights from", model_file)
    metrics = pickle.load(open(metrics_file, "rb"))
    print("Loaded metrics from", metrics_file)
    best_reward = metrics["best_reward"]
    episode = metrics["episode"]
    print("best_reward", best_reward, "episode", episode)


if __name__ == "__main__":
    cli()
