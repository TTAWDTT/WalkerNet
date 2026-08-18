from __future__ import annotations

from argparse import Namespace

import numpy as np
import torch

from scripts.cnop.compute_tos_zos_cnop import (
    NeutralCase,
    build_domain_mask,
    cnop_objective,
    select_diverse_candidates,
)


class _DatasetStub:
    def __init__(self) -> None:
        self.source_payloads = [
            {
                "lat": np.asarray([-70.0, -30.0, 0.0, 30.0, 70.0], dtype=np.float32),
                "lon": np.asarray([0.0, 100.0, 120.0, 200.0, 290.0, 300.0], dtype=np.float32),
                "valid_mask": torch.ones((4, 5, 6), dtype=torch.bool),
            }
        ]


CASE = NeutralCase(0, "test", 12, 2000, 0.0, 0.0)


def test_basin_masks_are_disjoint_within_common_latitudes() -> None:
    dataset = _DatasetStub()
    pacific = build_domain_mask(dataset, CASE, "pacific", (-20, 20), (120, 290), torch.device("cpu"))
    remote = build_domain_mask(dataset, CASE, "atlantic_indian", (-20, 20), (120, 290), torch.device("cpu"))

    assert not torch.any(pacific & remote)
    assert pacific[0, 0, 2, 2]
    assert pacific[0, 0, 2, 4]
    assert remote[0, 0, 2, 0]
    assert not pacific[0, 0, 0].any()


def test_global_mask_contains_both_basin_masks() -> None:
    dataset = _DatasetStub()
    global_mask = build_domain_mask(dataset, CASE, "global", (-20, 20), (120, 290), torch.device("cpu"))
    pacific = build_domain_mask(dataset, CASE, "pacific", (-20, 20), (120, 290), torch.device("cpu"))
    remote = build_domain_mask(dataset, CASE, "atlantic_indian", (-20, 20), (120, 290), torch.device("cpu"))

    assert torch.all(global_mask | ~pacific)
    assert torch.all(global_mask | ~remote)


def test_late_three_month_objective_uses_leads_10_to_12() -> None:
    forecast = torch.arange(1.0, 13.0)
    baseline = torch.zeros(12)
    args = Namespace(objective_mode="late_3m_delta", objective_lead=12, horizon=12, objective_temperature=0.25)

    assert torch.isclose(cnop_objective(forecast, baseline, args), torch.tensor(11.0))


def test_candidate_selection_removes_nearly_identical_fields() -> None:
    candidates = [
        {"delta_norm": np.asarray([1.0, 0.0]), "objective": 3.0},
        {"delta_norm": np.asarray([0.999, 0.001]), "objective": 2.9},
        {"delta_norm": np.asarray([0.0, 1.0]), "objective": 2.0},
    ]

    selected = select_diverse_candidates(candidates, top_k=3, max_cosine_similarity=0.98)

    assert [item["objective"] for item in selected] == [3.0, 2.0]
