"""
WalkerNet 最小共享接口约定。

这个文件只规定模型侧和数据/训练侧必须共同遵守的输入输出形式。

分工边界：
- Ziyi Zhuang 只需要实现:
    model(x, target_month, rollout_step=None) -> y_pred
- Zhen Luo 只需要保证 DataLoader 产出的 batch 符合本文件约定。

除非双方确认，否则不要随意修改这里的 shape、变量顺序和 key 名称。
"""

from __future__ import annotations

try:
    from typing import NotRequired, TypedDict
except ImportError:  # Python 3.8 compatibility on the training server.
    from typing import TypedDict

    from typing_extensions import NotRequired

import torch


# ============================================================
# 1. 变量顺序约定
# ============================================================

# 所有张量中的变量维度都必须使用这个顺序。
# 也就是 x/y/y_pred 的第 3 个维度（index=2）含义固定为：
#   0 -> tos   Sea surface temperature
#   1 -> zos   Sea surface height above geoid
#   2 -> tauu  Zonal wind stress
#   3 -> tauv  Meridional wind stress
VARIABLES = ("tos", "zos", "tauu", "tauv")
NUM_VARIABLES = 4


# ============================================================
# 2. 空间网格约定
# ============================================================

# 所有变量已经提前重网格到 1° x 1° 全球规则网格。
# 模型和数据侧都只需要处理下面这个固定空间尺寸。
GRID_H = 180
GRID_W = 360

# 网格点中心：
#   lat = -89.5, -88.5, ..., 88.5, 89.5
#   lon =   0.5,   1.5, ..., 358.5, 359.5
LAT_RANGE = (-89.5, 89.5)
LON_RANGE = (0.5, 359.5)


# ============================================================
# 3. 模型输入输出约定
# ============================================================

# Ziyi Zhuang 需要实现的 forward 形式：
#
#   y_pred = model(x, target_month, rollout_step=None)
#
# 输入：
#   x:
#       torch.float32
#       shape = (B, L, 4, 180, 360)
#       含义：归一化后的历史 L 个月、4 个变量、全球场
#
#   target_month:
#       torch.int64 / torch.long
#       shape = (B,)
#       取值范围：1-12
#       含义：预测目标 y 所在的月份，不是 x 最后一个月的月份
#
#   rollout_step:
#       torch.int64 / torch.long 或 None
#       shape = (B,)
#       含义：自回归预测时的步数
#       约定：None 等价于全 0；单步训练时可以不传
#
# 输出：
#   y_pred:
#       torch.float32
#       shape = (B, 1, 4, 180, 360)
#       含义：归一化空间中的下一月预测结果
#
# 模型侧不需要处理 valid_mask；mask 由训练/loss 侧使用。


# ============================================================
# 4. Dataset / DataLoader 输出约定
# ============================================================

# Dataset 单样本建议返回 WalkerSample。
# DataLoader 默认 collate 后应得到 WalkerBatch。


class WalkerSample(TypedDict):
    """WalkerDataset.__getitem__ 返回的单个样本。"""

    # 输入窗口，shape = (L, 4, 180, 360)，float32
    x: torch.Tensor

    # 预测目标，shape = (1, 4, 180, 360)，float32
    y: torch.Tensor

    # y 对应的月份，Python int，范围 1-12
    target_month: int

    # 有效区域 mask，shape = (4, 180, 360)，bool
    # True 表示该变量在该网格点有效；False 表示无效或缺测。
    valid_mask: torch.Tensor

    # 可选：调试/评估用，不是模型输入。
    time_index: NotRequired[int]
    target_time: NotRequired[object]


class WalkerBatch(TypedDict):
    """DataLoader collate 后传给 trainer 的 batch。"""

    # shape = (B, L, 4, 180, 360)，float32
    x: torch.Tensor

    # shape = (B, 1, 4, 180, 360)，float32
    y: torch.Tensor

    # shape = (B,)，int64
    target_month: torch.Tensor

    # shape = (B, 4, 180, 360)，bool
    valid_mask: torch.Tensor

    # 可选：调试/评估用，不是模型输入。
    time_index: NotRequired[torch.Tensor]
    target_time: NotRequired[object]


# ============================================================
# 5. NaN / 缺测值约定
# ============================================================

# 传给模型的 x 不能包含 NaN 或 Inf。
# 如果某些位置无效或缺测，数据侧需要：
#   1. 在 valid_mask 中标记为 False
#   2. 在 x/y 张量中用 FILL_VALUE 填充
#
# 训练侧计算 loss 时应使用 valid_mask 忽略无效位置。
FILL_VALUE = 0.0
