"""
Utility functions: seed, config loading, parameter counting, logging.

Owner: Zhen Luo
"""

import random
import numpy as np
import torch


def set_seed(seed=42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(path):
    """Load YAML config file and return as dict.

    Args:
        path: path to yaml file
    Returns:
        dict (or OmegaConf) with all config keys
    """
    raise NotImplementedError


def count_parameters(model):
    """Count total and trainable parameters.

    Args:
        model: nn.Module
    Returns:
        (total_params, trainable_params) tuple
    """
    raise NotImplementedError
