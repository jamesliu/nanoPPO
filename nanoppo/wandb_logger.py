import wandb

class WandBLogger:
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

