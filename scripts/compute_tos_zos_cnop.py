"""Compute TOS/ZOS CNOP-like perturbations for a trained WalkerNet model.

The script selects neutral January-December target years, uses the previous
January-December window as model input, and optimizes a bounded perturbation on
the final input month. Only ``tos`` and ``zos`` are perturbed by default.

Example:
    python scripts/compute_tos_zos_cnop.py \
        --config configs/server_3090_mixed5_ddp8.yaml \
        --checkpoint /mnt/sda/WalkerNet/checkpoints_mixed5_enso18_simple_loss_corr_ddp8/best_skill.pt \
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

ROOT = Path(__file__).resolve().parents[1]
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
    parser.add_argument("--horizon", type=int, default=12, help="Rollout months for the CNOP objective.")
    parser.add_argument("--steps", type=int, default=80, help="Projected-gradient optimization steps per case.")
    parser.add_argument("--lr", type=float, default=0.08)
    parser.add_argument("--epsilon-tos", type=float, default=0.1, help="Normalized RMS radius for TOS perturbation.")
    parser.add_argument("--epsilon-zos", type=float, default=0.1, help="Normalized RMS radius for ZOS perturbation.")
    parser.add_argument("--max-abs", type=float, default=2.0, help="Elementwise normalized perturbation clip.")
    parser.add_argument("--neutral-threshold", type=float, default=0.5)
    parser.add_argument("--domain", type=str, default="tropical_pacific", choices=("tropical_pacific", "global"))
    parser.add_argument("--perturb-grid", type=str, default="patch", choices=("patch", "full"))
    parser.add_argument("--perturb-patch-size", type=int, default=4)
    parser.add_argument("--lat-bounds", type=str, default="-20,20")
    parser.add_argument("--lon-bounds", type=str, default="120,290")
    parser.add_argument("--objective-temperature", type=float, default=0.25)
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


def build_domain_mask(
    dataset: WalkerDataset,
    case: NeutralCase,
    domain: str,
    lat_bounds: tuple[float, float],
    lon_bounds: tuple[float, float],
    device: torch.device,
) -> torch.Tensor:
    """Return perturbation mask with shape ``(1, 2, H, W)``."""
    payload = dataset.source_payloads[case.source_idx]
    lat = torch.as_tensor(payload["lat"], dtype=torch.float32, device=device)
    lon = torch.as_tensor(payload["lon"], dtype=torch.float32, device=device)
    if domain == "global":
        region = torch.ones((lat.numel(), lon.numel()), dtype=torch.bool, device=device)
    else:
        lat_mask = (lat >= lat_bounds[0]) & (lat <= lat_bounds[1])
        lon_mask = (lon >= lon_bounds[0]) & (lon <= lon_bounds[1])
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
) -> None:
    """Project perturbation parameters using the full-resolution perturbation norm."""
    with torch.no_grad():
        full_delta = expand_delta(delta_param, target_hw, perturb_grid)
        mask_f = mask.to(dtype=delta_param.dtype)
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


def cnop_objective(nino_anom: torch.Tensor, temperature: float) -> torch.Tensor:
    """Soft maximum over target-year 3-month Nino3.4 anomaly."""
    rolling = three_month_mean(nino_anom)
    if rolling.numel() == 0:
        rolling = nino_anom
    temp = max(float(temperature), 1e-6)
    return temp * torch.logsumexp(rolling / temp, dim=0)


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

    target_hw = (x0.shape[-2], x0.shape[-1])
    if args.perturb_grid == "patch":
        patch_size = max(1, int(args.perturb_patch_size))
        param_hw = (math.ceil(target_hw[0] / patch_size), math.ceil(target_hw[1] / patch_size))
    else:
        param_hw = target_hw
    delta_param = torch.zeros((1, 2, param_hw[0], param_hw[1]), dtype=x0.dtype, device=device, requires_grad=True)
    optimizer = torch.optim.Adam([delta_param], lr=float(args.lr))
    eps = (float(args.epsilon_tos), float(args.epsilon_zos))
    history: list[dict[str, float]] = []

    for step in range(1, int(args.steps) + 1):
        optimizer.zero_grad(set_to_none=True)
        delta = expand_delta(delta_param, target_hw, args.perturb_grid)
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
            use_checkpoint=args.checkpoint_rollout,
        )
        objective = cnop_objective(nino, args.objective_temperature)
        penalty = smoothness_penalty(delta, mask) * float(args.smoothness_weight)
        loss = -objective + penalty
        loss.backward()
        optimizer.step()
        project_delta_param(delta_param, mask, eps, float(args.max_abs), target_hw, args.perturb_grid)

        if step == 1 or step == int(args.steps) or step % max(1, int(args.steps) // 10) == 0:
            with torch.no_grad():
                rolling = three_month_mean(nino)
                history.append(
                    {
                        "step": float(step),
                        "objective": float(objective.detach().cpu().item()),
                        "max_3m": float(rolling.max().detach().cpu().item()),
                        "mean_3m": float(rolling.mean().detach().cpu().item()),
                        "loss": float(loss.detach().cpu().item()),
                    }
                )

    with torch.no_grad():
        final_delta = expand_delta(delta_param, target_hw, args.perturb_grid) * mask.to(dtype=x0.dtype)
        final_x = apply_delta(x0, final_delta, mask)
        final_nino = rollout_nino_anomaly(
            model,
            dataset,
            case,
            final_x,
            climatology,
            args.horizon,
            trained_rollout_steps,
            lat,
            lon,
            use_amp=args.amp,
            use_checkpoint=False,
        )
        baseline_3m = three_month_mean(baseline_nino)
        final_3m = three_month_mean(final_nino)

    delta_np = final_delta.detach().cpu().numpy()[0]
    delta_phys = normalized_delta_to_physical(dataset, final_delta.detach())[0]
    return {
        "case": case,
        "x0": x0.detach().cpu().numpy()[0],
        "delta_norm": delta_np,
        "delta_phys": delta_phys,
        "baseline_nino": baseline_nino.detach().cpu().numpy(),
        "final_nino": final_nino.detach().cpu().numpy(),
        "baseline_3m": baseline_3m.detach().cpu().numpy(),
        "final_3m": final_3m.detach().cpu().numpy(),
        "history": history,
        "mask_count": int(mask.sum().detach().cpu().item()),
        "baseline_max_3m": float(baseline_3m.max().detach().cpu().item()),
        "cnop_max_3m": float(final_3m.max().detach().cpu().item()),
        "gain_max_3m": float((final_3m.max() - baseline_3m.max()).detach().cpu().item()),
    }


def normalized_delta_to_physical(dataset: WalkerDataset, delta: torch.Tensor) -> np.ndarray:
    """Convert normalized TOS/ZOS perturbation to physical units."""
    if dataset.norm == "none":
        return delta.detach().cpu().numpy()
    if dataset.norm == "zscore":
        std = dataset._std[:2].to(device=delta.device, dtype=delta.dtype)  # noqa: SLF001
        std = std.view(1, 2, *([1] * (delta.ndim - 2)))
        return (delta * std).detach().cpu().numpy()
    if dataset.norm == "minmax":
        min_value = dataset._min[:2].to(device=delta.device, dtype=delta.dtype)  # noqa: SLF001
        max_value = dataset._max[:2].to(device=delta.device, dtype=delta.dtype)  # noqa: SLF001
        scale = max_value - min_value
        scale = scale.view(1, 2, *([1] * (delta.ndim - 2)))
        return (delta * scale).detach().cpu().numpy()
    raise ValueError(f"Unsupported normalization: {dataset.norm}")


def write_case_npz(output_dir: Path, result: dict[str, Any], dataset: WalkerDataset) -> Path:
    case: NeutralCase = result["case"]
    path = output_dir / f"case_{case.source_name}_{case.target_year}.npz"
    np.savez_compressed(
        path,
        delta_norm=result["delta_norm"],
        delta_phys=result["delta_phys"],
        baseline_nino=result["baseline_nino"],
        cnop_nino=result["final_nino"],
        baseline_3m=result["baseline_3m"],
        cnop_3m=result["final_3m"],
        lat=np.asarray(dataset.source_payloads[case.source_idx]["lat"]),
        lon=np.asarray(dataset.source_payloads[case.source_idx]["lon"]),
    )
    return path


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
                "cnop_max_3m",
                "gain_max_3m",
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
                    result["cnop_max_3m"],
                    result["gain_max_3m"],
                    result["mask_count"],
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
        "epsilon_tos": args.epsilon_tos,
        "epsilon_zos": args.epsilon_zos,
        "max_abs": args.max_abs,
        "domain": args.domain,
        "perturb_grid": args.perturb_grid,
        "perturb_patch_size": args.perturb_patch_size,
        "lat_bounds": args.lat_bounds,
        "lon_bounds": args.lon_bounds,
        "amp": args.amp,
        "checkpoint_rollout": args.checkpoint_rollout,
        "smoothness_weight": args.smoothness_weight,
        "objective": "softmax over target-year 3-month mean Nino3.4 anomaly",
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
    model, checkpoint = load_model(config, args.checkpoint, device)
    trained_rollout_steps = int(config.get("training", {}).get("rollout_steps", args.horizon))
    climatology_np = compute_source_nino34_climatology(dataset)
    climatology = torch.from_numpy(climatology_np).to(device=device, dtype=torch.float32)

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
        print(
            f"case {case.source_name} {case.target_year}: "
            f"baseline_max_3m={result['baseline_max_3m']:.4f} "
            f"cnop_max_3m={result['cnop_max_3m']:.4f} "
            f"gain={result['gain_max_3m']:.4f}",
            flush=True,
        )
        results.append(result)

    summary_path = write_summary_csv(output_dir, results)
    maybe_plot_results(output_dir, results, dataset)
    print(f"wrote {summary_path}", flush=True)
    print(f"wrote figures to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
