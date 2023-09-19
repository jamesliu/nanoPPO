import gym


class EnvironmentManager:
    def __init__(self, env_name, env_config: dict = None):
        self.env_name = env_name
        self.env_config = env_config

    def setup_env(self):
        if self.env_config:
            env = gym.make(self.env_name, config=self.env_config)
        else:
            env = gym.make(self.env_name)
        return env
