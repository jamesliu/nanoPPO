# nanoPPO

[![PyPI](https://img.shields.io/pypi/v/nanoPPO.svg)](https://pypi.org/project/nanoPPO/)
[![Changelog](https://img.shields.io/github/v/release/jamesliu/nanoPPO?include_prereleases&label=changelog)](https://github.com/jamesliu/nanoPPO/releases)
[![Tests](https://github.com/jamesliu/nanoPPO/workflows/Test/badge.svg)](https://github.com/jamesliu/nanoPPO/actions?query=workflow%3ATest)
[![Documentation Status](https://readthedocs.org/projects/nanoPPO/badge/?version=stable)](http://nanoPPO.readthedocs.org/en/stable/?badge=stable)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/jamesliu/nanoPPO/blob/main/LICENSE)

nanoPPO is a Python package that provides a simple and efficient implementation of the Proximal Policy Optimization (PPO) algorithm for reinforcement learning. It is designed to support both continuous and discrete action spaces, making it suitable for a wide range of applications.

## Installation

You can install nanoPPO directly from PyPI using pip:

```bash
pip install nanoPPO
```

Alternatively, you can clone the repository and install from source:

```bash
git clone https://github.com/jamesliu/nanoPPO.git
cd nanoPPO
pip install .
```

## Usage

Here are examples of how to use nanoPPO to train an agent.

### On the MountaionCarContinuous-v0 environment:

```python
from nanoppo.train_ppo_agent import train_agent
import pickle

ppo, model_file, metrics_file = train_agent(
    env_name="MountainCarContinuous-v0",
    max_episodes=50,
    policy_lr=0.0005,
    value_lr=0.0005,
    vl_coef=0.5,
    checkpoint_dir="checkpoints",
    checkpoint_interval=10,
    log_interval=10,
    wandb_log=False,
)
ppo.load(model_file)
print("Loaded best weights from", model_file)
metrics = pickle.load(open(metrics_file, "rb"))
print("Loaded metrics from", metrics_file)
best_reward = metrics["best_reward"]
episode = metrics["episode"]
print("best_reward", best_reward, "episode", episode)
```

#### Use Custom LR Scheduler and Custom Policy

* Set Cosine Annealing Learning Rate Scheduler
* Set CausalAttention Policy instead of Linear Policy

```python
from nanoppo.train_ppo_agent import train_agent
from nanoppo.cosine_lr_scheduler import CosineLRScheduler
from nanoppo.policy.actor_critic_causal_attention import ActorCriticCausalAttention

lr_scheduler=CosineLRScheduler(
    learning_rate=config['cosine_lr'], 
    warmup_iters=config['cosine_warmup_iters'], 
    lr_decay_iters=config['cosine_decay_steps'], 
    min_lr=config['cosine_min_lr'])

policy_class = ActorCriticCausalAttention

ppo, model_file, metrics_file = train_agent(
    env_name=env_name,
    env_config = env_config, 
    max_episodes=config['max_episode'],
    stop_reward=config['stop_reward'],
    policy_class = policy_class,
    lr_scheduler=lr_scheduler,
    policy_lr=config['policy_lr'],
    value_lr=config['value_lr'],
    vl_coef=config['vl_coef'],
    betas = config['betas'],
    n_latent_var=config['n_latent_var'],
    gamma=config['gamma'],
    K_epochs=config['K_epochs'],
    eps_clip=config['eps_clip'],
    el_coef=config['el_coef'],
    checkpoint_dir=checkpoint_dir,
    checkpoint_interval=10,
    log_interval=10,
    wandb_log=wandb_log,
    debug=True)
```

### On the CartPole-v1 environment:

```python
from nanoppo.discrete_action_ppo import PPO
import gym

env = gym.make('CartPole-v1')
ppo = PPO(env.observation_space.shape[0], env.action_space.n)

# Training code here...
```
## Examples
See the [examples](https://github.com/jamesliu/nanoPPO/tree/main/examples) directory for more comprehensive usage examples.

examples/train_mountaincar.sh

```
python nanoppo/train_ppo_agent.py --env_name=MountainCarContinuous-v0 --policy_lr=0.0005 --value_lr=0.0005 --max_episodes=50 --vl_coef=0.5 --wandb_log
```
![mountaincar](https://github.com/jamesliu/nanoPPO/blob/main/assets/MountainCarContinuous-v0.png)

examples/train_pointmass1d.sh

```
python nanoppo/train_ppo_agent.py --env_name=PointMass1D-v0 --policy_lr=0.0005 --value_lr=0.0005 --max_episodes=50 --vl_coef=0.5 --wandb_log
```

examples/train_pointmass2d.sh

```
python nanoppo/train_ppo_agent.py --env_name=PointMass2D-v0 --policy_lr=0.0005 --value_lr=0.0005 --max_episodes=50 --vl_coef=0.5 --wandb_log
```

## Documentation

Full documentation is available [here](https://nanoppo.readthedocs.io/en/latest/).

## Contributing

We welcome contributions to nanoPPO! If you're interested in contributing, please see our [contribution guidelines](https://github.com/jamesliu/nanoPPO/blob/main/CONTRIBUTING.md).

## License

nanoPPO is licensed under the Apache License 2.0. See the [LICENSE](https://github.com/jamesliu/nanoPPO/blob/main/LICENSE) file for more details.

## Support

For support, questions, or feature requests, please open an issue on our [GitHub repository](https://github.com/jamesliu/nanoPPO/issues) or contact the maintainers.

## Changelog

See the [releases](https://github.com/jamesliu/nanoPPO/releases) page for a detailed changelog of each version.


