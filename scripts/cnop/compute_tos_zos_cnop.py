"""Compute TOS/ZOS CNOP-like perturbations for a trained WalkerNet model.

The script selects neutral January-December target years, uses the previous
January-December window as model input, and optimizes a bounded perturbation on
the final input month. Only ``tos`` and ``zos`` are perturbed by default.

Example:
    python scripts/cnop/compute_tos_zos_cnop.py \
        --config configs/examples/mixed5.yaml \
        --checkpoint /path/to/checkpoints/best_skill.pt \
        --split test --device cuda --num-cases 10 --steps 80 \
        --output-dir outputs/cnop_tos_zos_best
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.utils.checkpoint
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dataset import WalkerDataset
from src.metrics import compute_nino34
from src.model import WalkerNet
from src.utils import load_config, set_seed


VARIABLES = ("tos", "zos", "tauu", "tauv")
PERTURB_VARIABLES = ("tos", "zos")


@dataclass(frozen=True)
class NeutralCase:
    """A target Jan-Dec year whose observed Nino3.4 anomaly is weak."""

    source_idx: int
    source_name: str
    target_t: int
    target_year: int
    neutral_score: float
    observed_max_3m_abs: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="WalkerNet TOS/ZOS CNOP optimization")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", type=str, default="test", choices=("train", "val", "test"))
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output-dir", type=str, default="outputs/cnop_tos_zos")
    parser.add_argument("--num-cases", type=int, default=10)
    parser.add_argument(
        "--case-year-range",
        type=str,
        default="",
        help="Optional inclusive target-year range, e.g. 1851,2014. If set, neutral cases are selected from all sources instead of the split sample list.",
    )
    parser.add_argument("--case-source-name", type=str, default="", help="Run one specific case from this source.")
    parser.add_argument("--case-target-year", type=int, default=0, help="Run one specific Jan-Dec target year.")
    parser.add_argument("--horizon", type=int, default=12, help="Rollout months for the CNOP objective.")
    parser.add_argument("--steps", type=int, default=80, help="Projected-gradient optimization steps per case.")
    parser.add_argument("--lr", type=float, default=0.08)
    parser.add_argument("--num-starts", type=int, default=16, help="Number of initial perturbations optimized per case.")
    parser.add_argument("--start-index-offset", type=int, default=0, help="Global start-index offset used by parallel shards.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of locally optimal CNOP candidates saved per case.")
    parser.add_argument(
        "--candidate-max-cosine-similarity",
        type=float,
        default=0.98,
        help="Greedy top-k candidates must not exceed this cosine similarity.",
    )
    parser.add_argument("--random-init-scale", type=float, default=0.02, help="Normalized random perturbation scale for nonzero starts.")
    parser.add_argument("--lbfgs-steps", type=int, default=0, help="Optional projected L-BFGS refinement iterations for top-k candidates.")
    parser.add_argument("--lbfgs-lr", type=float, default=0.5)
    parser.add_argument("--epsilon-tos", type=float, default=0.1, help="Normalized RMS radius for TOS perturbation.")
    parser.add_argument("--epsilon-zos", type=float, default=0.1, help="Normalized RMS radius for ZOS perturbation.")
    parser.add_argument(
        "--constraint-mode",
        type=str,
        default="normalized_rms",
        choices=("normalized_rms", "relative_initial_l2", "event_l2"),
        help="Perturbation norm constraint. event_l2 uses an ENSO-event based joint TOS/ZOS L2 radius.",
    )
    parser.add_argument("--relative-l2-fraction", type=float, default=0.1)
    parser.add_argument(
        "--constraint-file",
        type=str,
        default="",
        help="JSON produced by scripts/cnop/compute_cnop_constraint.py.",
    )
    parser.add_argument("--constraint-scale", type=float, default=1.0, help="Scale applied to event_l2 constraint.")
    parser.add_argument("--max-abs", type=float, default=2.0, help="Elementwise normalized perturbation clip.")
    parser.add_argument("--neutral-threshold", type=float, default=0.5)
    parser.add_argument(
        "--domain",
        type=str,
        default="tropical_pacific",
        choices=("tropical_pacific", "pacific", "indian", "atlantic_indian", "global"),
    )
    parser.add_argument("--perturb-grid", type=str, default="patch", choices=("patch", "full"))
    parser.add_argument("--perturb-patch-size", type=int, default=4)
    parser.add_argument("--lat-bounds", type=str, default="-20,20")
    parser.add_argument("--lon-bounds", type=str, default="120,290")
    parser.add_argument(
        "--basin-lat-bounds",
        type=str,
        default="-60,60",
        help="Latitude range used by Pacific and Atlantic-Indian basin-sector masks.",
    )
    parser.add_argument(
        "--objective-mode",
        type=str,
        default="softmax_3m",
        choices=("softmax_3m", "lead_delta", "late_3m_delta", "delayed_lead_delta"),
        help="delayed_lead_delta maximizes lead_delta while penalizing excessive early response.",
    )
    parser.add_argument("--objective-lead", type=int, default=12, help="1-based forecast lead used by objective-mode=lead_delta.")
    parser.add_argument("--objective-temperature", type=float, default=0.25)
    parser.add_argument("--delay-early-leads", type=int, default=3, help="Number of initial leads included in delayed-onset penalty.")
    parser.add_argument("--delay-early-threshold", type=float, default=0.2, help="Absolute early perturbed-minus-baseline Nino3.4 threshold in degC.")
    parser.add_argument("--delay-penalty-weight", type=float, default=2.0, help="Weight on the delayed-onset early-response penalty.")
    parser.add_argument("--smoothness-weight", type=float, default=0.001)
    parser.set_defaults(amp=True, checkpoint_rollout=True)
    parser.add_argument("--amp", dest="amp", action="store_true")
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    parser.add_argument("--checkpoint-rollout", dest="checkpoint_rollout", action="store_true")
    parser.add_argument("--no-checkpoint-rollout", dest="checkpoint_rollout", action="store_false")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def parse_bounds(value: str) -> tuple[float, float]:
    parts = [float(item.strip()) for item in value.split(",") if item.strip()]
    if len(parts) != 2:
        raise ValueError(f"bounds must contain two comma-separated numbers, got {value!r}")
    return min(parts), max(parts)


def parse_year_range(value: str) -> tuple[int, int] | None:
    if not value.strip():
        return None
    parts = [int(item.strip()) for item in value.split(",") if item.strip()]
    if len(parts) != 2:
        raise ValueError(f"year range must contain two comma-separated years, got {value!r}")
    return min(parts), max(parts)


def load_model(config: dict[str, Any], checkpoint_path: str, device: torch.device) -> tuple[WalkerNet, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = WalkerNet(config).to(device)
    model.load_state_dict(checkpoint["model"])
    for param in model.parameters():
        param.requires_grad_(False)
    model.eval()
    return model, checkpoint


def compute_source_nino34_climatology(dataset: WalkerDataset) -> np.ndarray:
    """Compute source-wise monthly Nino3.4 climatology from training years."""
    train_start, train_end = dataset.data_config["train_years"]
    climatology = np.zeros((len(dataset.source_payloads), 13), dtype=np.float32)
    for source_idx, payload in enumerate(dataset.source_payloads):
        years = payload["years"]
        months = payload["months"]
        train_mask = (years >= int(train_start)) & (years <= int(train_end))
        tos = np.asarray(payload["data"][:, 0], dtype=np.float32)
        nino = compute_nino34_numpy(tos, np.asarray(payload["lat"]), np.asarray(payload["lon"]))
        for month in range(1, 13):
            month_mask = train_mask & (months == month)
            climatology[source_idx, month] = float(np.nanmean(nino[month_mask]))
    return climatology


def compute_nino34_numpy(data: np.ndarray, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Compute Nino3.4 for ``data`` with shape ``(T, H, W)``."""
    lat_mask = (lat >= -5.0) & (lat <= 5.0)
    lon_mask = (lon >= 190.0) & (lon <= 240.0)
    region = data[:, lat_mask, :][:, :, lon_mask]
    lon_mean = np.nanmean(region, axis=2)
    weights = np.cos(np.deg2rad(lat[lat_mask])).astype(np.float64)
    weights = weights / weights.sum()
    return np.nansum(lon_mean * weights[None, :], axis=1).astype(np.float32)


def three_month_mean(values: torch.Tensor) -> torch.Tensor:
    if values.numel() < 3:
        return values
    return torch.stack([values[i - 2 : i + 1].mean() for i in range(2, values.numel())])


def three_month_mean_np(values: np.ndarray) -> np.ndarray:
    if values.size < 3:
        return values
    return np.asarray([values[i - 2 : i + 1].mean() for i in range(2, values.size)], dtype=np.float32)


def select_neutral_cases(
    dataset: WalkerDataset,
    climatology: np.ndarray,
    num_cases: int,
    horizon: int,
    threshold: float,
    case_year_range: tuple[int, int] | None = None,
) -> list[NeutralCase]:
    """Select Jan-Dec target windows with weak observed Nino3.4 anomaly."""
    candidates: list[NeutralCase] = []
    if case_year_range is None:
        candidate_indices = [(int(item[0]), int(item[1])) for item in dataset.sample_indices]
    else:
        start_year, end_year = case_year_range
        candidate_indices = []
        for source_idx, payload in enumerate(dataset.source_payloads):
            years = payload["years"]
            months = payload["months"]
            for target_t in np.where(months == 1)[0]:
                year = int(years[target_t])
                if start_year <= year <= end_year:
                    candidate_indices.append((source_idx, int(target_t)))

    for source_idx, target_t in candidate_indices:
        payload = dataset.source_payloads[source_idx]
        years = payload["years"]
        months = payload["months"]
        if int(months[target_t]) != 1:
            continue
        final_t = target_t + horizon - 1
        if final_t >= len(years):
            continue
        if int(months[final_t]) != 12 or int(years[final_t]) != int(years[target_t]):
            continue
        if target_t - dataset.L < 0 or int(months[target_t - dataset.L]) != 1:
            continue

        tos = np.asarray(payload["data"][target_t : target_t + horizon, 0], dtype=np.float32)
        raw = compute_nino34_numpy(tos, np.asarray(payload["lat"]), np.asarray(payload["lon"]))
        anomaly = raw - climatology[source_idx, months[target_t : target_t + horizon]]
        rolling = three_month_mean_np(anomaly)
        max_abs = float(np.nanmax(np.abs(rolling)))
        candidates.append(
            NeutralCase(
                source_idx=source_idx,
                source_name=dataset.source_names[source_idx],
                target_t=target_t,
                target_year=int(years[target_t]),
                neutral_score=max_abs,
                observed_max_3m_abs=max_abs,
            )
        )

    below = [case for case in candidates if case.neutral_score <= threshold]
    pool = below if len(below) >= num_cases else candidates
    pool = sorted(pool, key=lambda case: (case.neutral_score, case.source_name, case.target_year))
    return pool[:num_cases]


def select_specific_case(
    dataset: WalkerDataset,
    climatology: np.ndarray,
    source_name: str,
    target_year: int,
    horizon: int,
) -> NeutralCase:
    """Select exactly one Jan-Dec case by source and target year."""

    source_to_idx = {name: idx for idx, name in enumerate(dataset.source_names)}
    if source_name not in source_to_idx:
        raise ValueError(f"Unknown source {source_name!r}; available={list(dataset.source_names)}")
    source_idx = source_to_idx[source_name]
    payload = dataset.source_payloads[source_idx]
    years = np.asarray(payload["years"])
    months = np.asarray(payload["months"])
    starts = np.where((years == int(target_year)) & (months == 1))[0]
    if starts.size == 0:
        raise ValueError(f"No January start found for source={source_name} target_year={target_year}")
    target_t = int(starts[0])
    final_t = target_t + horizon - 1
    if final_t >= len(years) or int(months[final_t]) != 12 or int(years[final_t]) != int(target_year):
        raise ValueError(f"source={source_name} target_year={target_year} is not a complete Jan-Dec window")
    if target_t - dataset.L < 0 or int(months[target_t - dataset.L]) != 1:
        raise ValueError(f"source={source_name} target_year={target_year} does not have previous Jan-Dec input")

    tos = np.asarray(payload["data"][target_t : target_t + horizon, 0], dtype=np.float32)
    raw = compute_nino34_numpy(tos, np.asarray(payload["lat"]), np.asarray(payload["lon"]))
    anomaly = raw - climatology[source_idx, months[target_t : target_t + horizon]]
    rolling = three_month_mean_np(anomaly)
    max_abs = float(np.nanmax(np.abs(rolling)))
    return NeutralCase(
        source_idx=source_idx,
        source_name=source_name,
        target_t=target_t,
        target_year=int(target_year),
        neutral_score=max_abs,
        observed_max_3m_abs=max_abs,
    )


def build_domain_mask(
    dataset: WalkerDataset,
    case: NeutralCase,
    domain: str,
    lat_bounds: tuple[float, float],
    lon_bounds: tuple[float, float],
    device: torch.device,
    basin_lat_bounds: tuple[float, float] = (-60.0, 60.0),
) -> torch.Tensor:
    """Return perturbation mask with shape ``(1, 2, H, W)``."""
    payload = dataset.source_payloads[case.source_idx]
    lat = torch.as_tensor(payload["lat"], dtype=torch.float32, device=device)
    lon = torch.as_tensor(payload["lon"], dtype=torch.float32, device=device)
    lon_360 = torch.remainder(lon, 360.0)
    if domain == "global":
        region = torch.ones((lat.numel(), lon.numel()), dtype=torch.bool, device=device)
    elif domain == "tropical_pacific":
        lat_mask = (lat >= lat_bounds[0]) & (lat <= lat_bounds[1])
        lon_mask = (lon_360 >= lon_bounds[0]) & (lon_360 <= lon_bounds[1])
        region = lat_mask[:, None] & lon_mask[None, :]
    else:
        # Transparent longitude-sector definitions keep the basin experiments
        # reproducible without a runtime dependency on external shapefiles.
        lat_mask = (lat >= basin_lat_bounds[0]) & (lat <= basin_lat_bounds[1])
        pacific_lon = (lon_360 >= 120.0) & (lon_360 <= 290.0)
        if domain == "pacific":
            lon_mask = pacific_lon
        elif domain == "indian":
            # Indian Ocean sector: 20E-120E, restricted to the common basin
            # latitude band. This excludes the Atlantic for a clean comparison.
            lon_mask = (lon_360 >= 20.0) & (lon_360 < 120.0)
        else:
            # Complement of the Pacific sector, retained for the historical
            # Atlantic+Indian diagnostic mode.
            lon_mask = ~pacific_lon
        region = lat_mask[:, None] & lon_mask[None, :]

    valid = dataset.source_payloads[case.source_idx]["valid_mask"][:2].to(device=device, dtype=torch.bool)
    return (valid & region[None]).unsqueeze(0)


def make_case_input(dataset: WalkerDataset, case: NeutralCase, device: torch.device) -> torch.Tensor:
    """Create normalized input window for a selected case."""
    payload = dataset.source_payloads[case.source_idx]
    raw = np.array(payload["data"][case.target_t - dataset.L : case.target_t], dtype=np.float32, copy=True)
    x = torch.from_numpy(raw).to(device=device, dtype=torch.float32).unsqueeze(0)
    x = dataset._normalize_tensor(x)  # noqa: SLF001 - script intentionally reuses Dataset normalization
    return torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


def target_month_tensor(dataset: WalkerDataset, case: NeutralCase, step: int, device: torch.device) -> torch.Tensor:
    payload = dataset.source_payloads[case.source_idx]
    month = int(payload["months"][case.target_t + step])
    return torch.tensor([month], dtype=torch.long, device=device)


def rollout_nino_anomaly(
    model: torch.nn.Module,
    dataset: WalkerDataset,
    case: NeutralCase,
    x_norm: torch.Tensor,
    climatology: torch.Tensor,
    horizon: int,
    trained_rollout_steps: int,
    lat: torch.Tensor,
    lon: torch.Tensor,
    use_amp: bool = False,
    use_checkpoint: bool = False,
) -> torch.Tensor:
    """Differentiable 12-month WalkerNet rollout returning Nino3.4 anomalies."""
    values: list[torch.Tensor] = []
    window = x_norm
    for step in range(horizon):
        target_month = target_month_tensor(dataset, case, step, x_norm.device)
        rollout_step = torch.tensor(
            [min(step, trained_rollout_steps - 1)],
            dtype=torch.long,
            device=x_norm.device,
        )
        with torch.cuda.amp.autocast(enabled=use_amp and x_norm.device.type == "cuda"):
            if use_checkpoint and torch.is_grad_enabled() and window.requires_grad:
                def _forward(inp: torch.Tensor) -> torch.Tensor:
                    return model(inp, target_month, rollout_step=rollout_step)

                try:
                    pred_norm = torch.utils.checkpoint.checkpoint(_forward, window, use_reentrant=False)
                except TypeError:
                    pred_norm = torch.utils.checkpoint.checkpoint(_forward, window)
            else:
                pred_norm = model(window, target_month, rollout_step=rollout_step)
            pred_phys = dataset.denormalize(pred_norm)
            raw = compute_nino34(pred_phys[:, 0, 0], lat, lon)
        clim = climatology[case.source_idx, int(target_month.item())]
        values.append(raw[0] - clim)
        window = torch.cat([window[:, 1:], pred_norm], dim=1)
    return torch.stack(values)


def apply_delta(x_norm: torch.Tensor, delta: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Apply TOS/ZOS perturbation to the last input month."""
    x_work = x_norm.clone()
    masked_delta = delta * mask.to(dtype=delta.dtype)
    x_work[:, -1, 0] = x_work[:, -1, 0] + masked_delta[:, 0]
    x_work[:, -1, 1] = x_work[:, -1, 1] + masked_delta[:, 1]
    return x_work


def expand_delta(delta_param: torch.Tensor, target_hw: tuple[int, int], perturb_grid: str) -> torch.Tensor:
    """Map optimized perturbation parameters to full-resolution fields."""
    if perturb_grid == "full":
        return delta_param
    return F.interpolate(delta_param, size=target_hw, mode="bilinear", align_corners=False)


def project_delta_param(
    delta_param: torch.Tensor,
    mask: torch.Tensor,
    eps: tuple[float, float],
    max_abs: float,
    target_hw: tuple[int, int],
    perturb_grid: str,
    args: argparse.Namespace,
    dataset: WalkerDataset | None = None,
    x0: torch.Tensor | None = None,
    case: NeutralCase | None = None,
) -> None:
    """Project perturbation parameters using the full-resolution perturbation norm."""
    with torch.no_grad():
        full_delta = expand_delta(delta_param, target_hw, perturb_grid)
        mask_f = mask.to(dtype=delta_param.dtype)
        if args.constraint_mode == "event_l2":
            if dataset is None or case is None:
                raise ValueError("event_l2 constraint requires dataset and case")
            radius = float(args.event_constraint_l2)
            delta_dimless = normalized_delta_to_event_constraint_tensor(dataset, case.source_idx, full_delta, args)
            delta_norm = torch.sqrt((delta_dimless.square() * mask_f).sum()).clamp_min(1.0e-12)
            if float(delta_norm.item()) > radius:
                delta_param.mul_(radius / delta_norm)
        elif args.constraint_mode == "relative_initial_l2":
            if dataset is None or x0 is None:
                raise ValueError("relative_initial_l2 constraint requires dataset and x0")
            delta_phys = normalized_delta_to_physical_tensor(dataset, full_delta)
            x0_phys = dataset.denormalize(x0)[:, -1, :2]
            for idx in range(2):
                delta_norm = torch.sqrt((delta_phys[:, idx].square() * mask_f[:, idx]).sum())
                initial_norm = torch.sqrt((x0_phys[:, idx].square() * mask_f[:, idx]).sum()).clamp_min(1.0e-6)
                radius = float(args.relative_l2_fraction) * initial_norm
                if float(delta_norm.item()) > float(radius.item()):
                    delta_param[:, idx].mul_(radius / delta_norm)
        else:
            for idx, radius in enumerate(eps):
                denom = mask_f[:, idx].sum().clamp_min(1.0)
                rms = torch.sqrt((full_delta[:, idx].square() * mask_f[:, idx]).sum() / denom)
                if float(rms.item()) > radius:
                    delta_param[:, idx].mul_(radius / rms)
        delta_param.clamp_(min=-max_abs, max=max_abs)


def smoothness_penalty(delta: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask_f = mask.to(dtype=delta.dtype)
    dx = delta[..., :, 1:] - delta[..., :, :-1]
    mx = mask_f[..., :, 1:] * mask_f[..., :, :-1]
    dy = delta[..., 1:, :] - delta[..., :-1, :]
    my = mask_f[..., 1:, :] * mask_f[..., :-1, :]
    return ((dx.square() * mx).sum() + (dy.square() * my).sum()) / (mx.sum() + my.sum()).clamp_min(1.0)


def delayed_early_penalty(
    nino_anom: torch.Tensor,
    baseline_nino: torch.Tensor,
    args: argparse.Namespace,
) -> torch.Tensor:
    """Penalize an excessive early CNOP response while retaining differentiability."""

    lead_count = min(max(int(getattr(args, "delay_early_leads", 3)), 1), int(nino_anom.numel()))
    threshold = max(float(getattr(args, "delay_early_threshold", 0.2)), 1.0e-6)
    early_delta = nino_anom[:lead_count] - baseline_nino[:lead_count]
    excess = torch.relu(torch.abs(early_delta) - threshold)
    return excess.square().mean()


def cnop_objective(nino_anom: torch.Tensor, baseline_nino: torch.Tensor, args: argparse.Namespace) -> torch.Tensor:
    """Return the selected CNOP objective."""
    if args.objective_mode == "lead_delta":
        lead_idx = min(max(int(args.objective_lead), 1), int(args.horizon)) - 1
        return nino_anom[lead_idx] - baseline_nino[lead_idx]

    if args.objective_mode == "delayed_lead_delta":
        lead_idx = min(max(int(args.objective_lead), 1), int(args.horizon)) - 1
        lead_gain = nino_anom[lead_idx] - baseline_nino[lead_idx]
        penalty = delayed_early_penalty(nino_anom, baseline_nino, args)
        return lead_gain - float(getattr(args, "delay_penalty_weight", 2.0)) * penalty

    if args.objective_mode == "late_3m_delta":
        if nino_anom.numel() < 3:
            return nino_anom.mean() - baseline_nino.mean()
        return nino_anom[-3:].mean() - baseline_nino[-3:].mean()

    rolling = three_month_mean(nino_anom)
    if rolling.numel() == 0:
        rolling = nino_anom
    temp = max(float(args.objective_temperature), 1e-6)
    return temp * torch.logsumexp(rolling / temp, dim=0)


def initialize_delta_param(
    shape: tuple[int, int, int, int],
    x0: torch.Tensor,
    mask: torch.Tensor,
    eps: tuple[float, float],
    args: argparse.Namespace,
    dataset: WalkerDataset,
    target_hw: tuple[int, int],
    start_idx: int,
    seed: int,
    case: NeutralCase,
) -> torch.Tensor:
    """初始化一个候选扰动；start 0 从零开始，其余 start 随机初始化。"""

    delta_param = torch.zeros(shape, dtype=x0.dtype, device=x0.device)
    if start_idx > 0 and float(args.random_init_scale) > 0:
        generator = torch.Generator(device=x0.device)
        generator.manual_seed(int(seed))
        delta_param.normal_(mean=0.0, std=float(args.random_init_scale), generator=generator)
    project_delta_param(
        delta_param,
        mask,
        eps,
        float(args.max_abs),
        target_hw,
        args.perturb_grid,
        args,
        dataset=dataset,
        x0=x0,
        case=case,
    )
    return delta_param.requires_grad_(True)


def evaluate_delta(
    model: torch.nn.Module,
    dataset: WalkerDataset,
    case: NeutralCase,
    x0: torch.Tensor,
    delta: torch.Tensor,
    mask: torch.Tensor,
    climatology: torch.Tensor,
    args: argparse.Namespace,
    trained_rollout_steps: int,
    lat: torch.Tensor,
    lon: torch.Tensor,
    baseline_nino: torch.Tensor,
    baseline_3m: torch.Tensor,
    use_checkpoint: bool,
) -> dict[str, Any]:
    """对一个完整分辨率扰动计算目标值和 Niño3.4 响应。"""

    x_pert = apply_delta(x0, delta, mask)
    nino = rollout_nino_anomaly(
        model,
        dataset,
        case,
        x_pert,
        climatology,
        args.horizon,
        trained_rollout_steps,
        lat,
        lon,
        use_amp=args.amp,
        use_checkpoint=use_checkpoint,
    )
    rolling = three_month_mean(nino)
    objective = cnop_objective(nino, baseline_nino, args)
    early_penalty = delayed_early_penalty(nino, baseline_nino, args)
    lead_idx = min(max(int(args.objective_lead), 1), int(args.horizon)) - 1
    return {
        "nino": nino,
        "three_month": rolling,
        "objective": objective,
        "early_penalty": early_penalty,
        "lead_nino": nino[lead_idx],
        "lead_delta": nino[lead_idx] - baseline_nino[lead_idx],
        "max_3m": rolling.max(),
        "mean_3m": rolling.mean(),
        "gain_max_3m": rolling.max() - baseline_3m.max(),
    }


def optimize_single_start(
    model: torch.nn.Module,
    dataset: WalkerDataset,
    case: NeutralCase,
    x0: torch.Tensor,
    mask: torch.Tensor,
    climatology: torch.Tensor,
    args: argparse.Namespace,
    trained_rollout_steps: int,
    lat: torch.Tensor,
    lon: torch.Tensor,
    baseline_nino: torch.Tensor,
    baseline_3m: torch.Tensor,
    target_hw: tuple[int, int],
    param_hw: tuple[int, int],
    eps: tuple[float, float],
    start_idx: int,
    seed: int,
) -> dict[str, Any]:
    """从一个初值出发做 projected Adam，上山寻找一个局部 CNOP。"""

    delta_param = initialize_delta_param(
        (1, 2, param_hw[0], param_hw[1]),
        x0,
        mask,
        eps,
        args,
        dataset,
        target_hw,
        start_idx,
        seed,
        case,
    )
    optimizer = torch.optim.Adam([delta_param], lr=float(args.lr))
    history: list[dict[str, float]] = []

    for step in range(1, int(args.steps) + 1):
        optimizer.zero_grad(set_to_none=True)
        delta = expand_delta(delta_param, target_hw, args.perturb_grid)
        metrics = evaluate_delta(
            model,
            dataset,
            case,
            x0,
            delta,
            mask,
            climatology,
            args,
            trained_rollout_steps,
            lat,
            lon,
            baseline_nino,
            baseline_3m,
            use_checkpoint=args.checkpoint_rollout,
        )
        penalty = smoothness_penalty(delta, mask) * float(args.smoothness_weight)
        loss = -metrics["objective"] + penalty
        loss.backward()
        optimizer.step()
        project_delta_param(
            delta_param,
            mask,
            eps,
            float(args.max_abs),
            target_hw,
            args.perturb_grid,
            args,
            dataset=dataset,
            x0=x0,
            case=case,
        )

        if step == 1 or step == int(args.steps) or step % max(1, int(args.steps) // 10) == 0:
            history.append(
                {
                    "step": float(step),
                    "objective": float(metrics["objective"].detach().cpu().item()),
                    "early_penalty": float(metrics["early_penalty"].detach().cpu().item()),
                    "lead_delta": float(metrics["lead_delta"].detach().cpu().item()),
                    "max_3m": float(metrics["max_3m"].detach().cpu().item()),
                    "mean_3m": float(metrics["mean_3m"].detach().cpu().item()),
                    "loss": float(loss.detach().cpu().item()),
                }
            )

    with torch.no_grad():
        final_delta = expand_delta(delta_param, target_hw, args.perturb_grid) * mask.to(dtype=x0.dtype)
        final_metrics = evaluate_delta(
            model,
            dataset,
            case,
            x0,
            final_delta,
            mask,
            climatology,
            args,
            trained_rollout_steps,
            lat,
            lon,
            baseline_nino,
            baseline_3m,
            use_checkpoint=False,
        )
        constraint_norm, constraint_radius = measure_constraint_norm(
            dataset,
            case,
            x0,
            final_delta,
            mask,
            eps,
            args,
        )

    return {
        "start_idx": start_idx,
        "seed": seed,
        "delta_norm": final_delta.detach().cpu().numpy()[0],
        "delta_param": delta_param.detach(),
        "delta_phys": normalized_delta_to_physical(dataset, final_delta.detach())[0],
        "final_nino": final_metrics["nino"].detach().cpu().numpy(),
        "final_3m": final_metrics["three_month"].detach().cpu().numpy(),
        "objective": float(final_metrics["objective"].detach().cpu().item()),
        "early_penalty": float(final_metrics["early_penalty"].detach().cpu().item()),
        "lead_nino": float(final_metrics["lead_nino"].detach().cpu().item()),
        "lead_delta": float(final_metrics["lead_delta"].detach().cpu().item()),
        "cnop_max_3m": float(final_metrics["max_3m"].detach().cpu().item()),
        "gain_max_3m": float(final_metrics["gain_max_3m"].detach().cpu().item()),
        "constraint_norm": constraint_norm,
        "constraint_radius": constraint_radius,
        "constraint_ratio": constraint_norm / max(constraint_radius, 1.0e-12),
        "history": history,
    }


def refine_with_lbfgs(
    model: torch.nn.Module,
    dataset: WalkerDataset,
    case: NeutralCase,
    x0: torch.Tensor,
    mask: torch.Tensor,
    climatology: torch.Tensor,
    args: argparse.Namespace,
    trained_rollout_steps: int,
    lat: torch.Tensor,
    lon: torch.Tensor,
    baseline_nino: torch.Tensor,
    baseline_3m: torch.Tensor,
    target_hw: tuple[int, int],
    eps: tuple[float, float],
    initial_delta_param: torch.Tensor,
) -> torch.Tensor:
    """用 projected L-BFGS 对 Adam 找到的候选扰动做局部精修。"""

    delta_param = initial_delta_param.detach().clone().requires_grad_(True)
    optimizer = torch.optim.LBFGS(
        [delta_param],
        lr=float(args.lbfgs_lr),
        max_iter=int(args.lbfgs_steps),
        line_search_fn="strong_wolfe",
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad(set_to_none=True)
        delta = expand_delta(delta_param, target_hw, args.perturb_grid)
        metrics = evaluate_delta(
            model,
            dataset,
            case,
            x0,
            delta,
            mask,
            climatology,
            args,
            trained_rollout_steps,
            lat,
            lon,
            baseline_nino,
            baseline_3m,
            use_checkpoint=args.checkpoint_rollout,
        )
        penalty = smoothness_penalty(delta, mask) * float(args.smoothness_weight)
        loss = -metrics["objective"] + penalty
        loss.backward()
        return loss

    optimizer.step(closure)
    project_delta_param(
        delta_param,
        mask,
        eps,
        float(args.max_abs),
        target_hw,
        args.perturb_grid,
        args,
        dataset=dataset,
        x0=x0,
        case=case,
    )
    return delta_param.detach()


def candidate_from_delta_param(
    model: torch.nn.Module,
    dataset: WalkerDataset,
    case: NeutralCase,
    x0: torch.Tensor,
    mask: torch.Tensor,
    climatology: torch.Tensor,
    args: argparse.Namespace,
    trained_rollout_steps: int,
    lat: torch.Tensor,
    lon: torch.Tensor,
    baseline_nino: torch.Tensor,
    baseline_3m: torch.Tensor,
    target_hw: tuple[int, int],
    delta_param: torch.Tensor,
    template: dict[str, Any],
) -> dict[str, Any]:
    """精修后重新评估候选扰动，并继承 start/history 元信息。"""

    with torch.no_grad():
        full_delta = expand_delta(delta_param, target_hw, args.perturb_grid) * mask.to(dtype=x0.dtype)
        metrics = evaluate_delta(
            model,
            dataset,
            case,
            x0,
            full_delta,
            mask,
            climatology,
            args,
            trained_rollout_steps,
            lat,
            lon,
            baseline_nino,
            baseline_3m,
            use_checkpoint=False,
        )
    candidate = dict(template)
    candidate.update(
        {
            "delta_norm": full_delta.detach().cpu().numpy()[0],
            "delta_param": delta_param.detach(),
            "delta_phys": normalized_delta_to_physical(dataset, full_delta.detach())[0],
            "final_nino": metrics["nino"].detach().cpu().numpy(),
            "final_3m": metrics["three_month"].detach().cpu().numpy(),
            "objective": float(metrics["objective"].detach().cpu().item()),
            "lead_nino": float(metrics["lead_nino"].detach().cpu().item()),
            "lead_delta": float(metrics["lead_delta"].detach().cpu().item()),
            "cnop_max_3m": float(metrics["max_3m"].detach().cpu().item()),
            "gain_max_3m": float(metrics["gain_max_3m"].detach().cpu().item()),
            "refined_with_lbfgs": True,
        }
    )
    return candidate


def optimize_case(
    model: torch.nn.Module,
    dataset: WalkerDataset,
    case: NeutralCase,
    climatology: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
    trained_rollout_steps: int,
) -> dict[str, Any]:
    payload = dataset.source_payloads[case.source_idx]
    lat = torch.as_tensor(payload["lat"], dtype=torch.float32, device=device)
    lon = torch.as_tensor(payload["lon"], dtype=torch.float32, device=device)
    x0 = make_case_input(dataset, case, device)
    mask = build_domain_mask(
        dataset,
        case,
        args.domain,
        parse_bounds(args.lat_bounds),
        parse_bounds(args.lon_bounds),
        device,
        parse_bounds(args.basin_lat_bounds),
    )

    with torch.no_grad():
        baseline_nino = rollout_nino_anomaly(
            model,
            dataset,
            case,
            x0,
            climatology,
            args.horizon,
            trained_rollout_steps,
            lat,
            lon,
            use_amp=args.amp,
            use_checkpoint=False,
        )
        baseline_3m = three_month_mean(baseline_nino)

    target_hw = (x0.shape[-2], x0.shape[-1])
    if args.perturb_grid == "patch":
        patch_size = max(1, int(args.perturb_patch_size))
        param_hw = (math.ceil(target_hw[0] / patch_size), math.ceil(target_hw[1] / patch_size))
    else:
        param_hw = target_hw
    eps = (float(args.epsilon_tos), float(args.epsilon_zos))
    num_starts = max(1, int(args.num_starts))
    top_k = max(1, min(int(args.top_k), num_starts))
    candidates: list[dict[str, Any]] = []

    start_offset = max(0, int(args.start_index_offset))
    for local_start_idx in range(num_starts):
        start_idx = start_offset + local_start_idx
        start_seed = int(args.seed) + case.source_idx * 100_000 + case.target_t * 10 + start_idx
        candidate = optimize_single_start(
            model,
            dataset,
            case,
            x0,
            mask,
            climatology,
            args,
            trained_rollout_steps,
            lat,
            lon,
            baseline_nino,
            baseline_3m,
            target_hw,
            param_hw,
            eps,
            start_idx,
            start_seed,
        )
        candidates.append(candidate)
        print(
            f"  start {local_start_idx + 1}/{num_starts} global_idx={start_idx}: "
            f"objective={candidate['objective']:.4f} "
            f"lead_delta={candidate['lead_delta']:.4f} "
            f"cnop_max_3m={candidate['cnop_max_3m']:.4f} "
            f"gain={candidate['gain_max_3m']:.4f}",
            flush=True,
        )

    candidates = sorted(candidates, key=lambda item: (item["objective"], item["cnop_max_3m"]), reverse=True)
    top_candidates = select_diverse_candidates(
        candidates,
        top_k,
        float(args.candidate_max_cosine_similarity),
    )
    if int(args.lbfgs_steps) > 0:
        refined_candidates: list[dict[str, Any]] = []
        for candidate in top_candidates:
            refined_param = refine_with_lbfgs(
                model,
                dataset,
                case,
                x0,
                mask,
                climatology,
                args,
                trained_rollout_steps,
                lat,
                lon,
                baseline_nino,
                baseline_3m,
                target_hw,
                eps,
                candidate["delta_param"],
            )
            refined_candidates.append(
                candidate_from_delta_param(
                    model,
                    dataset,
                    case,
                    x0,
                    mask,
                    climatology,
                    args,
                    trained_rollout_steps,
                    lat,
                    lon,
                    baseline_nino,
                    baseline_3m,
                    target_hw,
                    refined_param,
                    candidate,
                )
            )
        refined_start_indices = {item["start_idx"] for item in refined_candidates}
        other_candidates = [item for item in candidates if item["start_idx"] not in refined_start_indices]
        candidates = sorted(refined_candidates + other_candidates, key=lambda item: (item["objective"], item["cnop_max_3m"]), reverse=True)
        top_candidates = select_diverse_candidates(
            candidates,
            top_k,
            float(args.candidate_max_cosine_similarity),
        )
    for rank, candidate in enumerate(top_candidates, start=1):
        candidate["rank"] = rank
    best = top_candidates[0]
    lead_idx = min(max(int(args.objective_lead), 1), int(args.horizon)) - 1

    return {
        "case": case,
        "x0": x0.detach().cpu().numpy()[0],
        "delta_norm": best["delta_norm"],
        "delta_phys": best["delta_phys"],
        "baseline_nino": baseline_nino.detach().cpu().numpy(),
        "final_nino": best["final_nino"],
        "baseline_3m": baseline_3m.detach().cpu().numpy(),
        "final_3m": best["final_3m"],
        "history": best["history"],
        "candidates": candidates,
        "top_candidates": top_candidates,
        "mask_count": int(mask.sum().detach().cpu().item()),
        "baseline_max_3m": float(baseline_3m.max().detach().cpu().item()),
        "baseline_lead_nino": float(baseline_nino[lead_idx].detach().cpu().item()),
        "cnop_lead_nino": best["lead_nino"],
        "lead_delta": best["lead_delta"],
        "cnop_max_3m": best["cnop_max_3m"],
        "gain_max_3m": best["gain_max_3m"],
        "best_start_idx": best["start_idx"],
        "best_objective": best["objective"],
        "domain": args.domain,
    }


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    """Return cosine similarity between two TOS/ZOS perturbation fields."""

    left_flat = np.asarray(left, dtype=np.float64).reshape(-1)
    right_flat = np.asarray(right, dtype=np.float64).reshape(-1)
    denominator = np.linalg.norm(left_flat) * np.linalg.norm(right_flat)
    if denominator <= 1.0e-12:
        return 1.0 if np.linalg.norm(left_flat - right_flat) <= 1.0e-12 else 0.0
    return float(np.dot(left_flat, right_flat) / denominator)


def select_diverse_candidates(
    candidates: list[dict[str, Any]],
    top_k: int,
    max_cosine_similarity: float,
) -> list[dict[str, Any]]:
    """Greedily retain strong candidates with genuinely different structures."""

    selected: list[dict[str, Any]] = []
    for candidate in candidates:
        if all(
            cosine_similarity(candidate["delta_norm"], previous["delta_norm"]) <= max_cosine_similarity
            for previous in selected
        ):
            selected.append(candidate)
        if len(selected) >= top_k:
            break
    return selected or candidates[:1]


def normalized_delta_to_physical(dataset: WalkerDataset, delta: torch.Tensor) -> np.ndarray:
    """Convert normalized TOS/ZOS perturbation to physical units."""
    return normalized_delta_to_physical_tensor(dataset, delta).detach().cpu().numpy()


def normalized_delta_to_physical_tensor(dataset: WalkerDataset, delta: torch.Tensor) -> torch.Tensor:
    """Convert normalized TOS/ZOS perturbation to physical units as a tensor."""
    if dataset.norm == "none":
        return delta
    if dataset.norm == "zscore":
        std = dataset._std[:2].to(device=delta.device, dtype=delta.dtype)  # noqa: SLF001
        std = std.view(1, 2, *([1] * (delta.ndim - 2)))
        return delta * std
    if dataset.norm == "minmax":
        min_value = dataset._min[:2].to(device=delta.device, dtype=delta.dtype)  # noqa: SLF001
        max_value = dataset._max[:2].to(device=delta.device, dtype=delta.dtype)  # noqa: SLF001
        scale = max_value - min_value
        scale = scale.view(1, 2, *([1] * (delta.ndim - 2)))
        return delta * scale
    raise ValueError(f"Unsupported normalization: {dataset.norm}")


def prepare_event_l2_constraint(dataset: WalkerDataset, args: argparse.Namespace) -> None:
    """Load event-based constraint metadata and prepare source-wise scaling tensors."""

    if args.constraint_mode != "event_l2":
        args.event_constraint_l2 = 0.0
        return
    if not args.constraint_file:
        raise ValueError("--constraint-file is required when --constraint-mode event_l2")

    constraint_path = Path(args.constraint_file)
    with constraint_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    radius = float(payload["constraint_l2"]) * float(args.constraint_scale)
    if radius <= 0:
        raise ValueError(f"event_l2 constraint must be positive, got {radius}")

    normalization = str(payload.get("normalization", "december_anomaly_train_std_equal_rms"))
    equalization = payload.get("equalization_rms_before_rescale", {})
    equalization_rms = torch.tensor(
        [
            float(equalization.get("tos", 1.0)),
            float(equalization.get("zos", 1.0)),
        ],
        dtype=torch.float32,
    ).clamp_min(1.0e-12)

    args.event_constraint_l2 = radius
    args.event_constraint_normalization = normalization
    args.event_constraint_equalization_rms = equalization_rms

    if normalization == "december_anomaly_train_std_equal_rms":
        train_start, train_end = dataset.data_config["train_years"]
        std_rows: list[np.ndarray] = []
        for payload_item in dataset.source_payloads:
            years = np.asarray(payload_item["years"])
            months = np.asarray(payload_item["months"])
            mask = (years >= int(train_start)) & (years <= int(train_end)) & (months == 12)
            fields = np.asarray(payload_item["data"][mask, :2], dtype=np.float32)
            mean_field = np.nanmean(fields, axis=0)
            std_scalar = np.nanstd(fields - mean_field[None], axis=(0, 2, 3)).astype(np.float32)
            std_scalar = np.where(std_scalar > 1.0e-12, std_scalar, 1.0).astype(np.float32)
            std_rows.append(std_scalar)
        args.event_constraint_december_std = torch.from_numpy(np.stack(std_rows, axis=0).astype(np.float32))
    elif normalization == "dataset_zscore_equal_rms":
        args.event_constraint_december_std = None
    else:
        raise ValueError(f"Unsupported event constraint normalization: {normalization}")

    print(
        f"[constraint] event_l2 radius={radius:.6f} "
        f"scale={float(args.constraint_scale):.4f} file={constraint_path}",
        flush=True,
    )


def normalized_delta_to_event_constraint_tensor(
    dataset: WalkerDataset,
    source_idx: int,
    delta: torch.Tensor,
    args: argparse.Namespace,
) -> torch.Tensor:
    """Convert delta to the same dimensionless space used by compute_cnop_constraint.py."""

    equalization = args.event_constraint_equalization_rms.to(device=delta.device, dtype=delta.dtype).view(1, 2, 1, 1)
    if args.event_constraint_normalization == "dataset_zscore_equal_rms":
        return delta / equalization

    delta_phys = normalized_delta_to_physical_tensor(dataset, delta)
    december_std = args.event_constraint_december_std[source_idx].to(device=delta.device, dtype=delta.dtype).view(1, 2, 1, 1)
    return delta_phys / december_std / equalization


def measure_constraint_norm(
    dataset: WalkerDataset,
    case: NeutralCase,
    x0: torch.Tensor,
    delta: torch.Tensor,
    mask: torch.Tensor,
    eps: tuple[float, float],
    args: argparse.Namespace,
) -> tuple[float, float]:
    """Measure the optimized perturbation in the same space as its hard constraint."""

    mask_f = mask.to(device=delta.device, dtype=delta.dtype)
    if args.constraint_mode == "event_l2":
        dimless = normalized_delta_to_event_constraint_tensor(dataset, case.source_idx, delta, args)
        norm = torch.sqrt((dimless.square() * mask_f).sum())
        return float(norm.item()), float(args.event_constraint_l2)
    if args.constraint_mode == "relative_initial_l2":
        delta_phys = normalized_delta_to_physical_tensor(dataset, delta)
        initial_phys = dataset.denormalize(x0)[:, -1, :2]
        ratios = []
        for idx in range(2):
            delta_norm = torch.sqrt((delta_phys[:, idx].square() * mask_f[:, idx]).sum())
            initial_norm = torch.sqrt((initial_phys[:, idx].square() * mask_f[:, idx]).sum()).clamp_min(1.0e-12)
            ratios.append(delta_norm / initial_norm)
        return float(torch.stack(ratios).max().item()), float(args.relative_l2_fraction)

    rms_values = []
    for idx in range(2):
        denominator = mask_f[:, idx].sum().clamp_min(1.0)
        rms_values.append(torch.sqrt((delta[:, idx].square() * mask_f[:, idx]).sum() / denominator) / eps[idx])
    return float(torch.stack(rms_values).max().item()), 1.0


def write_case_npz(output_dir: Path, result: dict[str, Any], dataset: WalkerDataset) -> Path:
    case: NeutralCase = result["case"]
    path = output_dir / f"case_{case.source_name}_{case.target_year}.npz"
    top_candidates = result.get("top_candidates", [])
    np.savez_compressed(
        path,
        delta_norm=result["delta_norm"],
        delta_phys=result["delta_phys"],
        top_delta_norm=np.stack([item["delta_norm"] for item in top_candidates], axis=0) if top_candidates else np.empty((0, 2, 0, 0)),
        top_delta_phys=np.stack([item["delta_phys"] for item in top_candidates], axis=0) if top_candidates else np.empty((0, 2, 0, 0)),
        top_cnop_nino=np.stack([item["final_nino"] for item in top_candidates], axis=0) if top_candidates else np.empty((0,)),
        top_cnop_3m=np.stack([item["final_3m"] for item in top_candidates], axis=0) if top_candidates else np.empty((0,)),
        top_objective=np.asarray([item["objective"] for item in top_candidates], dtype=np.float32),
        top_early_penalty=np.asarray([item["early_penalty"] for item in top_candidates], dtype=np.float32),
        top_lead_nino=np.asarray([item["lead_nino"] for item in top_candidates], dtype=np.float32),
        top_lead_delta=np.asarray([item["lead_delta"] for item in top_candidates], dtype=np.float32),
        top_cnop_max_3m=np.asarray([item["cnop_max_3m"] for item in top_candidates], dtype=np.float32),
        top_gain_max_3m=np.asarray([item["gain_max_3m"] for item in top_candidates], dtype=np.float32),
        top_constraint_norm=np.asarray([item["constraint_norm"] for item in top_candidates], dtype=np.float32),
        top_constraint_radius=np.asarray([item["constraint_radius"] for item in top_candidates], dtype=np.float32),
        top_constraint_ratio=np.asarray([item["constraint_ratio"] for item in top_candidates], dtype=np.float32),
        top_start_idx=np.asarray([item["start_idx"] for item in top_candidates], dtype=np.int32),
        baseline_nino=result["baseline_nino"],
        cnop_nino=result["final_nino"],
        baseline_3m=result["baseline_3m"],
        cnop_3m=result["final_3m"],
        lat=np.asarray(dataset.source_payloads[case.source_idx]["lat"]),
        lon=np.asarray(dataset.source_payloads[case.source_idx]["lon"]),
    )
    return path


def serializable_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    """去掉大数组和 Tensor，只保留便于 JSON/CSV 记录的候选元信息。"""

    return {
        "rank": candidate.get("rank"),
        "start_idx": candidate["start_idx"],
        "seed": candidate["seed"],
        "objective": candidate["objective"],
        "early_penalty": candidate["early_penalty"],
        "lead_nino": candidate["lead_nino"],
        "lead_delta": candidate["lead_delta"],
        "cnop_max_3m": candidate["cnop_max_3m"],
        "gain_max_3m": candidate["gain_max_3m"],
        "constraint_norm": candidate["constraint_norm"],
        "constraint_radius": candidate["constraint_radius"],
        "constraint_ratio": candidate["constraint_ratio"],
        "refined_with_lbfgs": bool(candidate.get("refined_with_lbfgs", False)),
        "history": candidate.get("history", []),
    }


def write_summary_csv(output_dir: Path, results: list[dict[str, Any]]) -> Path:
    path = output_dir / "cnop_summary.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "source",
                "target_year",
                "target_t",
                "observed_max_3m_abs",
                "baseline_max_3m",
                "baseline_lead_nino",
                "cnop_lead_nino",
                "lead_delta",
                "cnop_max_3m",
                "gain_max_3m",
                "best_start_idx",
                "best_objective",
                "best_early_penalty",
                "constraint_norm",
                "constraint_radius",
                "constraint_ratio",
                "mask_count",
            ]
        )
        for result in results:
            case: NeutralCase = result["case"]
            writer.writerow(
                [
                    case.source_name,
                    case.target_year,
                    case.target_t,
                    case.observed_max_3m_abs,
                    result["baseline_max_3m"],
                    result["baseline_lead_nino"],
                    result["cnop_lead_nino"],
                    result["lead_delta"],
                    result["cnop_max_3m"],
                    result["gain_max_3m"],
                    result["best_start_idx"],
                    result["best_objective"],
                    result["top_candidates"][0]["early_penalty"],
                    result["top_candidates"][0]["constraint_norm"],
                    result["top_candidates"][0]["constraint_radius"],
                    result["top_candidates"][0]["constraint_ratio"],
                    result["mask_count"],
                ]
            )
    return path


def write_candidate_summary_csv(output_dir: Path, results: list[dict[str, Any]]) -> Path:
    path = output_dir / "cnop_candidate_summary.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "source",
                "target_year",
                "rank",
                "start_idx",
                "seed",
                "objective",
                "early_penalty",
                "baseline_lead_nino",
                "cnop_lead_nino",
                "lead_delta",
                "baseline_max_3m",
                "cnop_max_3m",
                "gain_max_3m",
                "constraint_norm",
                "constraint_radius",
                "constraint_ratio",
                "refined_with_lbfgs",
            ]
        )
        for result in results:
            case: NeutralCase = result["case"]
            for candidate in result.get("top_candidates", []):
                writer.writerow(
                    [
                        case.source_name,
                        case.target_year,
                        candidate.get("rank"),
                        candidate["start_idx"],
                        candidate["seed"],
                        candidate["objective"],
                        candidate["early_penalty"],
                        result["baseline_lead_nino"],
                        candidate["lead_nino"],
                        candidate["lead_delta"],
                        result["baseline_max_3m"],
                        candidate["cnop_max_3m"],
                        candidate["gain_max_3m"],
                        candidate["constraint_norm"],
                        candidate["constraint_radius"],
                        candidate["constraint_ratio"],
                        bool(candidate.get("refined_with_lbfgs", False)),
                    ]
                )
    return path


def maybe_plot_results(output_dir: Path, results: list[dict[str, Any]], dataset: WalkerDataset) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"plot skipped: {exc}", flush=True)
        return

    summary_sources = [f"{r['case'].source_name}-{r['case'].target_year}" for r in results]
    gains = [r["gain_max_3m"] for r in results]
    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.bar(range(len(gains)), gains, color="#3A7CA5")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(range(len(gains)))
    ax.set_xticklabels(summary_sources, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("CNOP gain in max 3-month Nino3.4 anomaly")
    ax.set_title("WalkerNet TOS/ZOS CNOP response across neutral cases")
    fig.tight_layout()
    fig.savefig(output_dir / "cnop_gain_summary.png", dpi=180)
    plt.close(fig)

    if not results:
        return
    best = max(results, key=lambda row: row["gain_max_3m"])
    case: NeutralCase = best["case"]
    payload = dataset.source_payloads[case.source_idx]
    lat = np.asarray(payload["lat"])
    lon = np.asarray(payload["lon"])
    lon2, lat2 = np.meshgrid(lon, lat)

    fig, axes = plt.subplots(2, 2, figsize=(11, 6.5))
    for idx, (name, unit) in enumerate((("TOS CNOP", "physical"), ("ZOS CNOP", "physical"))):
        ax = axes[0, idx]
        data = best["delta_phys"][idx]
        vmax = float(np.nanpercentile(np.abs(data), 99))
        mesh = ax.pcolormesh(lon2, lat2, data, cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="auto")
        ax.set_title(f"{name} perturbation ({unit})")
        ax.set_xlim(100, 300)
        ax.set_ylim(-30, 30)
        fig.colorbar(mesh, ax=ax, orientation="horizontal", fraction=0.08, pad=0.08)

    months = np.arange(1, len(best["baseline_nino"]) + 1)
    axes[1, 0].plot(months, best["baseline_nino"], label="baseline", linewidth=2)
    axes[1, 0].plot(months, best["final_nino"], label="CNOP", linewidth=2)
    axes[1, 0].axhline(0.5, color="red", linestyle="--", linewidth=1, alpha=0.7)
    axes[1, 0].set_title("Monthly Nino3.4 anomaly forecast")
    axes[1, 0].set_xlabel("Target-year month")
    axes[1, 0].set_ylabel("Nino3.4 anomaly")
    axes[1, 0].legend()

    m3 = np.arange(3, len(best["baseline_nino"]) + 1)
    axes[1, 1].plot(m3, best["baseline_3m"], label="baseline 3m", linewidth=2)
    axes[1, 1].plot(m3, best["final_3m"], label="CNOP 3m", linewidth=2)
    axes[1, 1].axhline(0.5, color="red", linestyle="--", linewidth=1, alpha=0.7)
    axes[1, 1].set_title("3-month mean Nino3.4 anomaly")
    axes[1, 1].set_xlabel("Lead month")
    axes[1, 1].legend()

    fig.suptitle(f"Best CNOP case: {case.source_name} target {case.target_year}", fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_dir / "best_case_cnop_maps_and_nino.png", dpi=190)
    plt.close(fig)


def write_method_json(output_dir: Path, args: argparse.Namespace, checkpoint: dict[str, Any], cases: list[NeutralCase]) -> None:
    payload = {
        "config": args.config,
        "checkpoint": args.checkpoint,
        "checkpoint_epoch": checkpoint.get("epoch"),
        "split": args.split,
        "case_year_range": args.case_year_range or None,
        "num_cases": args.num_cases,
        "horizon": args.horizon,
        "steps": args.steps,
        "lr": args.lr,
        "num_starts": args.num_starts,
        "start_index_offset": args.start_index_offset,
        "top_k": args.top_k,
        "candidate_max_cosine_similarity": args.candidate_max_cosine_similarity,
        "random_init_scale": args.random_init_scale,
        "lbfgs_steps": args.lbfgs_steps,
        "lbfgs_lr": args.lbfgs_lr,
        "epsilon_tos": args.epsilon_tos,
        "epsilon_zos": args.epsilon_zos,
        "constraint_mode": args.constraint_mode,
        "relative_l2_fraction": args.relative_l2_fraction,
        "constraint_file": args.constraint_file or None,
        "constraint_scale": args.constraint_scale,
        "event_constraint_l2": getattr(args, "event_constraint_l2", None),
        "max_abs": args.max_abs,
        "domain": args.domain,
        "basin_lat_bounds": args.basin_lat_bounds,
        "perturb_grid": args.perturb_grid,
        "perturb_patch_size": args.perturb_patch_size,
        "lat_bounds": args.lat_bounds,
        "lon_bounds": args.lon_bounds,
        "amp": args.amp,
        "checkpoint_rollout": args.checkpoint_rollout,
        "smoothness_weight": args.smoothness_weight,
        "objective_mode": args.objective_mode,
        "objective_lead": args.objective_lead,
        "delay_early_leads": getattr(args, "delay_early_leads", 3),
        "delay_early_threshold": getattr(args, "delay_early_threshold", 0.2),
        "delay_penalty_weight": getattr(args, "delay_penalty_weight", 2.0),
        "objective": "delayed_lead_delta maximizes perturbed-minus-baseline Nino3.4 at objective_lead while penalizing squared early excess above delay_early_threshold; lead_delta is the unpenalized counterpart",
        "basin_mask_definition": {
            "pacific": "valid ocean, 120E-290E, within basin_lat_bounds",
            "indian": "valid ocean, 20E-120E, within basin_lat_bounds",
            "atlantic_indian": "valid ocean outside 120E-290E, within basin_lat_bounds",
            "global": "all valid ocean cells",
        },
        "selected_cases": [
            {
                "source": case.source_name,
                "target_year": case.target_year,
                "target_t": case.target_t,
                "observed_max_3m_abs": case.observed_max_3m_abs,
            }
            for case in cases
        ],
    }
    (output_dir / "method.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(args.config)
    dataset = WalkerDataset(config["data"]["path"], config, split=args.split)
    prepare_event_l2_constraint(dataset, args)
    model, checkpoint = load_model(config, args.checkpoint, device)
    trained_rollout_steps = int(config.get("training", {}).get("rollout_steps", args.horizon))
    climatology_np = compute_source_nino34_climatology(dataset)
    climatology = torch.from_numpy(climatology_np).to(device=device, dtype=torch.float32)

    if args.case_source_name or int(args.case_target_year) > 0:
        if not args.case_source_name or int(args.case_target_year) <= 0:
            raise ValueError("--case-source-name and --case-target-year must be set together")
        cases = [
            select_specific_case(
                dataset,
                climatology_np,
                args.case_source_name,
                int(args.case_target_year),
                args.horizon,
            )
        ]
    else:
        case_year_range = parse_year_range(args.case_year_range)
        cases = select_neutral_cases(
            dataset,
            climatology_np,
            args.num_cases,
            args.horizon,
            args.neutral_threshold,
            case_year_range=case_year_range,
        )
    if not cases:
        raise ValueError("No neutral Jan-Dec cases found")
    write_method_json(output_dir, args, checkpoint, cases)

    print(
        f"checkpoint_epoch={checkpoint.get('epoch')} split={args.split} "
        f"cases={len(cases)} horizon={args.horizon} steps={args.steps} device={device}",
        flush=True,
    )
    for case in cases:
        print(
            f"selected case source={case.source_name} target_year={case.target_year} "
            f"observed_max_3m_abs={case.observed_max_3m_abs:.4f}",
            flush=True,
        )

    results: list[dict[str, Any]] = []
    for idx, case in enumerate(cases, start=1):
        print(f"optimize case {idx}/{len(cases)} {case.source_name} {case.target_year}", flush=True)
        result = optimize_case(model, dataset, case, climatology, args, device, trained_rollout_steps)
        write_case_npz(output_dir, result, dataset)
        (output_dir / f"case_{case.source_name}_{case.target_year}_history.json").write_text(
            json.dumps(result["history"], indent=2),
            encoding="utf-8",
        )
        (output_dir / f"case_{case.source_name}_{case.target_year}_candidates.json").write_text(
            json.dumps([serializable_candidate(item) for item in result["top_candidates"]], indent=2),
            encoding="utf-8",
        )
        print(
            f"case {case.source_name} {case.target_year}: "
            f"baseline_max_3m={result['baseline_max_3m']:.4f} "
            f"cnop_max_3m={result['cnop_max_3m']:.4f} "
            f"gain={result['gain_max_3m']:.4f} "
            f"best_start={result['best_start_idx']}",
            flush=True,
        )
        results.append(result)

    summary_path = write_summary_csv(output_dir, results)
    candidate_summary_path = write_candidate_summary_csv(output_dir, results)
    maybe_plot_results(output_dir, results, dataset)
    print(f"wrote {summary_path}", flush=True)
    print(f"wrote {candidate_summary_path}", flush=True)
    print(f"wrote figures to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
