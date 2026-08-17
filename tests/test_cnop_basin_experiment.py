from __future__ import annotations

from argparse import Namespace

import numpy as np
import torch

from scripts.cnop.compute_tos_zos_cnop import (
    NeutralCase,
    build_domain_mask,
    cnop_objective,
    load_warm_start_deltas,
    select_diverse_candidates,
)
from scripts.cnop.basin_domains import numpy_basin_region


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
    assert torch.equal(global_mask, pacific | remote)
    assert not global_mask[0, 0, 0].any()
    assert not global_mask[0, 0, -1].any()


def test_numpy_and_torch_basin_masks_match() -> None:
    dataset = _DatasetStub()
    payload = dataset.source_payloads[0]
    expected = numpy_basin_region(payload["lat"], payload["lon"], "global", (-60.0, 60.0))
    actual = build_domain_mask(
        dataset,
        CASE,
        "global",
        (-20, 20),
        (120, 290),
        torch.device("cpu"),
        (-60.0, 60.0),
    )[0, 0].numpy()

    assert np.array_equal(actual, expected)


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


def test_warm_starts_include_regional_fields_and_three_mixtures(tmp_path) -> None:
    pacific_dir = tmp_path / "pacific"
    remote_dir = tmp_path / "atlantic_indian"
    pacific_dir.mkdir()
    remote_dir.mkdir()
    pacific = np.ones((2, 3, 4), dtype=np.float32)
    remote = np.full((2, 3, 4), 2.0, dtype=np.float32)
    pacific_path = pacific_dir / "case.npz"
    remote_path = remote_dir / "case.npz"
    np.savez_compressed(pacific_path, delta_norm=pacific)
    np.savez_compressed(remote_path, delta_norm=remote)

    warm_starts = load_warm_start_deltas([str(pacific_path), str(remote_path)])

    assert [label for label, _ in warm_starts] == [
        "pacific",
        "atlantic_indian",
        "mix:pacific+atlantic_indian",
        "mix:3pacific+atlantic_indian",
        "mix:pacific+3atlantic_indian",
    ]
    expected = [pacific, remote, pacific + remote, 0.75 * pacific + 0.25 * remote, 0.25 * pacific + 0.75 * remote]
    for (_, actual), target in zip(warm_starts, expected, strict=True):
        assert np.allclose(actual.numpy(), target)
