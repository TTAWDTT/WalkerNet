"""Mixed-source Dataset tests.

Run standalone:
    python tests/test_dataset_mixed_sources.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset import WalkerDataset  # noqa: E402


def _payload(source_offset: float) -> dict:
    data = np.full((36, 4, 180, 360), source_offset, dtype=np.float32)
    data[:, 0, :20, :] = np.nan
    years = np.repeat(np.arange(2000, 2003, dtype=np.int32), 12)
    months = np.tile(np.arange(1, 13, dtype=np.int8), 3)
    return {
        "data": data,
        "years": years,
        "months": months,
        "lat": np.arange(-89.5, 90.0, 1.0, dtype=np.float64),
        "lon": np.arange(0.5, 360.0, 1.0, dtype=np.float64),
        "valid_mask": torch.from_numpy(np.isfinite(data).any(axis=0)),
    }


def test_mixed_source_dataset_keeps_sample_internal_source_consistent():
    payloads = {
        "S1": _payload(1.0),
        "S2": _payload(2.0),
    }
    original_loader = WalkerDataset._load_or_get_cache

    def fake_loader(cls, data_path, data_config):
        return payloads[Path(data_path).name]

    WalkerDataset._load_or_get_cache = classmethod(fake_loader)
    try:
        config = {
            "data": {
                "path": "/unused/S1",
                "sources": [
                    {"name": "S1", "path": "/unused/S1"},
                    {"name": "S2", "path": "/unused/S2"},
                ],
                "L": 3,
                "H": 180,
                "W": 360,
                "variables": ["tos", "zos", "tauu", "tauv"],
                "norm": "none",
                "target_steps": 4,
                "train_years": [2000, 2001],
                "val_years": [2002, 2002],
                "test_years": [2002, 2002],
            }
        }
        dataset = WalkerDataset(None, config, split="train")
        assert dataset.source_names == ("S1", "S2")
        assert dataset.sample_indices.shape[1] == 2
        assert set(dataset.sample_indices[:, 0].tolist()) == {0, 1}

        first_s2_pos = int(np.where(dataset.sample_indices[:, 0] == 1)[0][0])
        sample = dataset[first_s2_pos]
        assert sample["source_id"] == "S2"
        assert sample["source_index"] == 1
        assert sample["x"].shape == (3, 4, 180, 360)
        assert sample["y_rollout"].shape == (4, 4, 180, 360)
        assert sample["target_months"].shape == (4,)
        assert torch.nan_to_num(sample["x"]).mean() > 0.0

        batch = next(iter(DataLoader(dataset, batch_size=2, shuffle=False)))
        assert batch["x"].shape == (2, 3, 4, 180, 360)
        assert batch["y_rollout"].shape == (2, 4, 4, 180, 360)
        assert batch["valid_mask"].shape == (2, 4, 180, 360)
        assert batch["source_index"].shape == (2,)
    finally:
        WalkerDataset._load_or_get_cache = original_loader


def _run_all():
    test_mixed_source_dataset_keeps_sample_internal_source_consistent()
    print("All mixed-source Dataset tests passed.")


if __name__ == "__main__":
    _run_all()
