"""
Training loop: forward, loss, backward, validation, checkpointing.

Owner: Zhen Luo
Responsibility: Training logic, logging, checkpoint save/load
"""

import torch


class Trainer:
    """Handles the training loop for WalkerNet."""

    def __init__(self, model, train_loader, val_loader, config):
        """
        Args:
            model: WalkerNet instance
            train_loader: DataLoader for training set
            val_loader: DataLoader for validation set
            config: dict with keys:
                - lr: learning rate
                - weight_decay: weight decay
                - epochs: total epochs
                - loss: loss function name ("mse", "weighted_mse", etc.)
                - grad_clip: gradient clipping value (optional)
                - save_dir: checkpoint save directory
                - log_interval: logging frequency (steps)
        """
        # TODO: implement
        # 1. Create optimizer (AdamW)
        # 2. Create scheduler (optional)
        # 3. Create loss function
        # 4. Set up logging (tensorboard / wandb / print)
        # 5. Set up checkpointing
        pass

    def train_epoch(self):
        """Run one training epoch.

        For each batch:
            1. model(x, target_month) -> y_pred
            2. compute loss(y_pred, y_true)
            3. loss.backward(), optimizer.step()
            4. log metrics

        Returns:
            dict of training metrics (avg loss, etc.)
        """
        raise NotImplementedError

    @torch.no_grad()
    def validate(self):
        """Run validation.

        Returns:
            dict of validation metrics (avg loss, per-variable loss, etc.)
        """
        raise NotImplementedError

    def save_checkpoint(self, path, epoch, metrics):
        """Save model + optimizer state and metrics."""
        raise NotImplementedError

    def load_checkpoint(self, path):
        """Load model + optimizer state. Returns epoch and metrics."""
        raise NotImplementedError

    def train(self):
        """Full training loop: iterate epochs, validate, checkpoint, early stop."""
        raise NotImplementedError
