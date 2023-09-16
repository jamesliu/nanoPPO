# nanoPPO

[![PyPI](https://img.shields.io/pypi/v/nanoPPO.svg)](https://pypi.org/project/nanoPPO/)
[![Changelog](https://img.shields.io/github/v/release/jamesliu/nanoPPO?include_prereleases&label=changelog)](https://github.com/jamesliu/nanoPPO/releases)
[![Tests](https://github.com/jamesliu/nanoPPO/workflows/Test/badge.svg)](https://github.com/jamesliu/nanoPPO/actions?query=workflow%3ATest)
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

Here's a simple example of how to use nanoPPO to train an agent on the CartPole-v1 environment:

```python
from nanoppo.discrete_action_ppo import PPO
import gym

env = gym.make('CartPole-v1')
ppo = PPO(env.observation_space.shape[0], env.action_space.n)

# Training code here...
```

See the [examples](./examples) directory for more comprehensive usage examples.

## Documentation

Full documentation is available [here](https://nanoppo.readthedocs.io/en/latest/).

## Contributing

We welcome contributions to nanoPPO! If you're interested in contributing, please see our [contribution guidelines](./CONTRIBUTING.md) and [code of conduct](./CODE_OF_CONDUCT.md).

## License

nanoPPO is licensed under the Apache License 2.0. See the [LICENSE](./LICENSE) file for more details.

## Support

For support, questions, or feature requests, please open an issue on our [GitHub repository](https://github.com/jamesliu/nanoPPO/issues) or contact the maintainers.

## Changelog

See the [releases](https://github.com/jamesliu/nanoPPO/releases) page for a detailed changelog of each version.

