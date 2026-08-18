"""生成并检查 WalkerDataset 的归一化统计量。

用法：
    python scripts/data/inspect_norm_stats.py --config configs/examples/mixed5.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset import WalkerDataset  # noqa: E402
from src.interfaces import VARIABLES  # noqa: E402
from src.utils import load_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect WalkerNet normalization statistics")
    parser.add_argument("--config", required=True, help="训练配置文件")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    data_config = config["data"]
    dataset = WalkerDataset(data_config["path"], config, split="train")

    print(f"norm={dataset.norm} scope={dataset.norm_scope}")
    if dataset.norm == "none":
        print("normalization disabled")
        return

    names = ("mean", "std") if dataset.norm == "zscore" else ("min", "max")
    for source_idx, source_name in enumerate(dataset.source_names):
        print(f"source={source_name}")
        for stat_name in names:
            values = dataset.norm_stats[stat_name]
            if values.ndim == 2:
                values = values[source_idx]
            fields = " ".join(
                f"{variable}={float(value):.8g}" for variable, value in zip(VARIABLES, values)
            )
            print(f"  {stat_name}: {fields}")

        # 用该 source 的第一个训练样本做一次端到端往返，防止统计量算对了但选错行。
        sample_position = int(np.where(dataset.sample_indices[:, 0] == source_idx)[0][0])
        sample = dataset[sample_position]
        restored = dataset.denormalize(sample["x"], source_idx)
        target_t = int(sample["time_index"])
        raw = torch.from_numpy(
            np.array(dataset.source_payloads[source_idx]["data"][target_t - dataset.L : target_t], copy=True)
        ).float()
        finite = torch.isfinite(raw)
        max_error = float((restored[finite] - raw[finite]).abs().max().item())
        print(f"  roundtrip_max_abs_error={max_error:.8g}")


if __name__ == "__main__":
    main()
