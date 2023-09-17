import pandas as pd
from collections import defaultdict


class MetricsRecorder:
    def __init__(self):
        self.losses = {
            "total_losses": [],
            "policy_losses": [],
            "entropy_losses": [],
            "value_losses": [],
        }
        self.actions = {"action_means": [], "action_stds": []}
        self.episode_rewards = {
            "episode_reward_mins": [],
            "episode_reward_means": [],
            "episode_reward_maxs": [],
        }
        self.learning = defaultdict(list)

    def record_losses(self, total_loss, policy_loss, entropy_loss, value_loss):
        self.losses["total_losses"].append(total_loss)
        self.losses["policy_losses"].append(policy_loss)
        self.losses["entropy_losses"].append(entropy_loss)
        self.losses["value_losses"].append(value_loss)

    def record_actions(self, action_mean, action_std):
        self.actions["action_means"].append(action_mean)
        self.actions["action_stds"].append(action_std)

    def record_rewards(self, rewards):
        self.episode_rewards["episode_reward_mins"].append(min(rewards))
        self.episode_rewards["episode_reward_means"].append(sum(rewards) / len(rewards))
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
        df_losses.to_csv("losses_metrics.csv", index=False)
        df_actions.to_csv("actions_metrics.csv", index=False)
        df_rewards.to_csv("rewards_metrics.csv", index=False)
        df_learning.to_csv("learning_metrics.csv", index=False)
