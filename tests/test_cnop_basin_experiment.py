from __future__ import annotations

from argparse import Namespace
import sys

import numpy as np
import torch

from scripts.cnop.summarize_constraint_scale_pilot import main as summarize_constraint_scale_pilot

from scripts.cnop.compute_tos_zos_cnop import (
    NeutralCase,
    accepts_objective_update,
    build_domain_mask,
    cnop_objective,
    load_warm_start_deltas,
    select_diverse_candidates,
)
from scripts.cnop.basin_domains import numpy_basin_region
from scripts.cnop.evaluate_basin_zero_state_gradient import scale_zero_state_gradient_to_event_radius
from scripts.cnop.plot_basin_cnop_gradient_comparison import load_domain_comparison


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


def test_accepted_adam_policy_rejects_objective_regressions() -> None:
    assert accepts_objective_update(0.5, 0.5)
    assert accepts_objective_update(0.5, 0.49, tolerance=0.01)
    assert not accepts_objective_update(0.5, 0.49)


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


def test_zero_state_gradient_is_scaled_to_event_radius_after_patch_upsampling() -> None:
    """The linear baseline uses the same full-resolution masked norm as CNOP."""

    class _NormNoneDataset:
        norm = "none"

    args = Namespace(
        constraint_mode="event_l2",
        event_constraint_l2=3.0,
        event_constraint_normalization="dataset_zscore_equal_rms",
        event_constraint_equalization_rms=torch.ones(2),
        perturb_grid="patch",
        max_abs=100.0,
    )
    gradient = torch.ones((1, 2, 1, 1))
    mask = torch.ones((1, 2, 2, 2), dtype=torch.bool)
    direction, projected, clipped = scale_zero_state_gradient_to_event_radius(
        gradient,
        dataset=_NormNoneDataset(),
        case=CASE,
        mask=mask,
        target_hw=(2, 2),
        args=args,
    )
    full = torch.nn.functional.interpolate(direction, size=(2, 2), mode="bilinear", align_corners=False)
    assert projected
    assert not clipped
    assert torch.isclose(torch.sqrt((full.square() * mask).sum()), torch.tensor(3.0))


def test_gradient_plot_loader_cross_checks_csv_npz_and_budget(tmp_path) -> None:
    domain = "pacific"
    combined = tmp_path / "combined" / domain
    gradient = tmp_path / "gradient_baseline" / domain
    random_controls = tmp_path / "random_controls"
    combined.mkdir(parents=True)
    gradient.mkdir(parents=True)
    random_controls.mkdir()
    (combined / "cnop_summary.csv").write_text(
        "source,target_year,best_objective,lead_delta,constraint_norm,constraint_radius,constraint_ratio\n"
        "GFDL-ESM4,1995,1.2,1.3,2.0,2.0,1.0\n",
        encoding="utf-8",
    )
    (gradient / "gradient_summary.csv").write_text(
        "source,target_year,objective_mode,objective,lead_delta,constraint_norm,constraint_radius,constraint_ratio\n"
        "GFDL-ESM4,1995,late_3m_delta,0.8,0.9,2.0,2.0,1.0\n",
        encoding="utf-8",
    )
    np.savez_compressed(
        gradient / "case_GFDL-ESM4_1995.npz",
        objective=np.asarray(0.8, dtype=np.float32),
        lead_delta=np.asarray(0.9, dtype=np.float32),
        constraint_norm=np.asarray(2.0, dtype=np.float32),
        constraint_radius=np.asarray(2.0, dtype=np.float32),
        constraint_ratio=np.asarray(1.0, dtype=np.float32),
        projected=np.asarray(True),
    )
    (random_controls / "pacific.csv").write_text(
        "objective,lead_delta\n0.1,0.2\n0.3,0.4\n",
        encoding="utf-8",
    )
    result = load_domain_comparison(tmp_path, domain, "objective", 1.0e-5, 1.0e-5)
    assert result.cnop_value == 1.2
    assert result.gradient_value == 0.8
    assert np.allclose(result.random_values, [0.1, 0.3])


def test_constraint_scale_pilot_summary_keeps_paired_methods(tmp_path, monkeypatch) -> None:
    case_dir = tmp_path / "scale_0p10" / "GFDL-ESM4_1995"
    (case_dir / "cnop").mkdir(parents=True)
    (case_dir / "gradient").mkdir()
    (case_dir / "cnop" / "cnop_summary.csv").write_text(
        "source,target_year,best_objective,constraint_ratio\nGFDL-ESM4,1995,0.6,1.0\n",
        encoding="utf-8",
    )
    (case_dir / "gradient" / "gradient_summary.csv").write_text(
        "objective,constraint_ratio\n0.5,1.0\n",
        encoding="utf-8",
    )
    (case_dir / "random_controls.csv").write_text(
        "objective\n0.1\n0.2\n0.3\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(sys, "argv", ["summarize", "--experiment-dir", str(tmp_path)])
    summarize_constraint_scale_pilot()

    summary = np.genfromtxt(
        tmp_path / "summary" / "constraint_scale_pilot_by_scale.csv",
        delimiter=",",
        names=True,
        dtype=None,
        encoding="utf-8",
    )
    assert np.isclose(summary["constraint_scale"], 0.1)
    assert summary["n_cases"] == 1
    assert np.isclose(summary["mean_cnop_minus_gradient"], 0.1)
