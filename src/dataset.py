"""
Dataset: loads physical field data and constructs training samples.

Owner: Zhen Luo
Responsibility: Data I/O, sliding window sample construction, normalization

Each sample:
  x: (L, 4, H, W)   input window (L months, 4 variables)
  y: (1, 4, H, W)   target (next month)
  target_month: int   1-12
"""

import torch
from torch.utils.data import Dataset


class WalkerDataset(Dataset):
    """Sliding-window dataset for WalkerNet.

    Given a time series of global fields, produces (input_window, target) pairs.
    Supports configurable input length L and train/val/test splitting by time.
    """

    def __init__(self, data_path, config, split="train"):
        """
        Args:
            data_path: path to data directory or file (nc / npy / npz)
            config: dict with keys:
                - L: input window length (3 or 12)
                - patch_size: spatial patch size
                - train_years: (start, end) tuple
                - val_years: (start, end) tuple
                - test_years: (start, end) tuple
                - norm: normalization method ("zscore", "minmax", etc.)
            split: "train", "val", or "test"
        """
        # TODO: implement
        # 1. Load raw data: (T, 4, H, W) where T = total months
        # 2. Split by year into train/val/test
        # 3. Compute normalization stats from training split only
        # 4. Store as tensor
        pass

    def __len__(self):
        """Number of sliding window samples."""
        raise NotImplementedError

    def __getitem__(self, idx):
        """
        Returns:
            x: (L, 4, H, W) float32
            y: (1, 4, H, W) float32
            target_month: int (1-12)
        """
        raise NotImplementedError

    def denormalize(self, y):
        """Convert normalized output back to physical units.

        Args:
            y: (B, 1, 4, H, W) normalized prediction
        Returns:
            (B, 1, 4, H, W) in original physical units
        """
        raise NotImplementedError
