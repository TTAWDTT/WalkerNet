"""WalkerNet 通用工具函数。

工具：
- 随机种子设置
- YAML 配置读取
- 模型参数量统计
- 设备选择
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


def set_seed(seed: int = 42, deterministic: bool = False) -> None:
    """设置随机种子

    Args:
        seed: 随机种子。
        deterministic: 是否开启更严格的确定性模式。
            设为 True 会降低某些 CUDA 算子的速度，但复现性更强。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_config(path: str | Path) -> dict[str, Any]:
    """读取 YAML 配置文件，返回普通 Python dict。

    Args:
        path: 配置文件路径，例如 ``configs/default.yaml``。

    Returns:
        配置字典。通常包含 ``data``、``model``、``training``、``logging``。

    Raises:
        FileNotFoundError: 配置文件不存在。
        ValueError: YAML 内容为空或不是 dict。
    """
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if config is None:
        raise ValueError(f"Config file is empty: {config_path}")
    if not isinstance(config, dict):
        raise ValueError(f"Config file must contain a YAML mapping/dict: {config_path}")

    return config


def count_parameters(model: torch.nn.Module) -> tuple[int, int]:
    """统计模型总参数量和可训练参数量。

    Args:
        model: PyTorch 模型。

    Returns:
        ``(total_params, trainable_params)``。
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def get_device(prefer_cuda: bool = True) -> torch.device:
    """选择训练设备。

    Args:
        prefer_cuda: 如果 CUDA 可用，是否优先使用 CUDA。

    Returns:
        ``torch.device("cuda")`` 或 ``torch.device("cpu")``。
    """
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
