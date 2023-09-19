import wandb
import numpy as np


class WandBLogger:
    @staticmethod
    def init(project, name, config):
        wandb.init(project=project, name=name, config=config)

    @staticmethod
    def log(data):
        wandb.log(data)

    @staticmethod
    def finish():
        wandb.finish()

    @staticmethod
    def log_rewards(rewards):
        # Log the rewards during training
        wandb.log(
            {
                "Reward/Min": min(rewards),
                "Reward/Mean": sum(rewards) / len(rewards),
                "Reward/Max": max(rewards),
            }
        )

    @staticmethod
    def log_action_distribution_parameters(
        action_mean: np.ndarray, action_std: np.ndarray
    ):
        """
        Logs the parameters (mean and standard deviation) of the action distribution using WandB.

        Args:
            action_mean (torch.Tensor): Mean values of the action distribution.
            action_std (torch.Tensor): Standard deviation values of the action distribution.
        """
        log_data = {
            f"ActionDist/Action{i}_Mean": mean_val
            for i, mean_val in enumerate(action_mean.tolist())
        }
        log_data.update(
            {
                f"ActionDist/Action{i}_Std": std_val
                for i, std_val in enumerate(action_std.tolist())
            }
        )
        wandb.log(log_data)
