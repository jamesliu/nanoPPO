import os
import glob
import torch


class CheckpointManager:
    @staticmethod
    def save_checkpoint(policy, value, optimizer, normalizer, epoch, checkpoint_path):
        # Create checkpoint directory if it does not exist
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)
        checkpoint = {
            "epoch": epoch,
            "policy_state_dict": policy.state_dict(),
            "value_state_dict": value.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "normalizer_state": normalizer.get_state(),
        }
        torch.save(checkpoint, f"{checkpoint_path}/checkpoint_epoch{epoch}.pt")

    @staticmethod
    def load_checkpoint(
        policy, value, optimizer, normalizer, checkpoint_path, epoch=None
    ):
        # Find the latest checkpoint file
        if epoch is None:
            checkpoint_files = sorted(
                glob.glob(f"{checkpoint_path}/checkpoint_epoch*.pt")
            )
            if len(checkpoint_files) == 0:
                raise ValueError("No checkpoint found in the specified directory.")
            checkpoint_file = checkpoint_files[-1]
        else:
            checkpoint_file = f"{checkpoint_path}/checkpoint_epoch{epoch}.pt"

        checkpoint = torch.load(checkpoint_file)
        policy.load_state_dict(checkpoint["policy_state_dict"])
        value.load_state_dict(checkpoint["value_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        normalizer.set_state(checkpoint["normalizer_state"])
        epoch = checkpoint["epoch"]
        return epoch
