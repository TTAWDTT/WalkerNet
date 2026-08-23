"""Plot a 10-case CNOP lead-12 overview figure.

Each row is one neutral target year. Columns show:

1. rank-1 initial delta TOS;
2. lead-12 TOS truth or response, controlled by ``--second-column``;
3. baseline lead-12 TOS/SSTA with the Nino3.4 index annotated;
4. perturbed lead-12 TOS/SSTA with the Nino3.4 index annotated.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.cnop.compute_tos_zos_cnop import compute_nino34_numpy  # noqa: E402
from scripts.cnop.plot_cnop_monthly_response import (  # noqa: E402
    NINO34_BOX,
    apply_delta,
    load_case_npz,
    load_model,
    make_case,
    make_case_input,
    month_labels,
    rollout_fields,
    setup_axis,
    smooth_field,
)
from src.dataset import WalkerDataset  # noqa: E402
from src.utils import load_config  # noqa: E402


TOS_CMAP = LinearSegmentedColormap.from_list(
    "overview_tos",
    ["#4B56A6", "#8FC7D9", "#F7F3D0", "#F0A35A", "#B61732"],
    N=256,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot 10 CNOP cases at lead 12.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
    )
    parser.add_argument("--cnop-dir", type=Path, default=Path("outputs/cnop_relative_l2_3pct_lead12_0704"))
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="CSV with source,target_year,cnop_dir and optional scale; supports rows from different CNOP runs.",
    )
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--candidate-rank", type=int, default=1)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--lead-month", type=int, default=12, help="1-based forecast lead to plot.")
    parser.add_argument(
        "--second-column",
        choices=("truth", "response"),
        default="truth",
        help="第二列画真值场，还是画扰动积分减去原始积分的响应场。",
    )
    parser.add_argument(
        "--tos-mode",
        choices=("anomaly", "raw"),
        default="anomaly",
        help="第二/三/四列画相对训练气候态的 SSTA anomaly，还是原始 TOS。",
    )
    parser.add_argument(
        "--forecast-climatology",
        choices=("train", "split", "all", "none"),
        default="train",
        help="模型预报场用模型自己的 lead/month 气候态订正；none 时退回观测月气候态。",
    )
    parser.add_argument("--forecast-climatology-cache", type=Path, default=None)
    parser.add_argument("--climatology-batch-size", type=int, default=2)
    parser.add_argument("--trained-rollout-steps", type=int, default=0)
    parser.add_argument("--smooth-sigma", type=float, default=0.8)
    parser.add_argument("--require-cases", type=int, default=10)
    parser.add_argument("--max-cases", type=int, default=10)
    parser.add_argument("--dpi", type=int, default=320)
    parser.add_argument("--output", type=Path, default=Path("figures/cnop_ten_case_lead12_3pct.png"))
    return parser.parse_args()


def set_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.linewidth": 0.55,
            "axes.titlesize": 8,
            "axes.labelsize": 7,
            "xtick.labelsize": 5.5,
            "ytick.labelsize": 5.5,
            "savefig.bbox": "tight",
            "savefig.dpi": 320,
        }
    )


def read_summary_rows(cnop_dir: Path, max_cases: int) -> list[dict[str, str]]:
    summary_path = cnop_dir / "cnop_summary.csv"
    with summary_path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return rows[:max_cases]


def read_manifest_rows(manifest_path: Path, max_cases: int) -> list[dict[str, str]]:
    """Resolve selected case rows from one or more CNOP output directories."""

    with manifest_path.open("r", newline="", encoding="utf-8") as handle:
        selections = list(csv.DictReader(handle))
    required = {"source", "target_year", "cnop_dir"}
    if not selections or not required.issubset(selections[0]):
        raise ValueError(f"{manifest_path} must contain columns: {', '.join(sorted(required))}")
    rows: list[dict[str, str]] = []
    for selection in selections[:max_cases]:
        source = selection["source"]
        year = int(selection["target_year"])
        cnop_dir = Path(selection["cnop_dir"])
        with (cnop_dir / "cnop_summary.csv").open("r", newline="", encoding="utf-8") as handle:
            matches = [
                row
                for row in csv.DictReader(handle)
                if row["source"] == source and int(row["target_year"]) == year
            ]
        if len(matches) != 1:
            raise ValueError(f"Expected one summary row for {source} {year} in {cnop_dir}, found {len(matches)}")
        row = matches[0]
        row["cnop_dir"] = str(cnop_dir)
        row["scale"] = selection.get("scale", "")
        rows.append(row)
    return rows


def symmetric_limit(values: list[np.ndarray], fallback: float, percentile: float = 99.0) -> float:
    if not values:
        return fallback
    stacked = np.concatenate([np.ravel(np.asarray(value, dtype=np.float32)) for value in values])
    vmax = float(np.nanpercentile(np.abs(stacked), percentile))
    return max(vmax, fallback)


def field_limit(values: list[np.ndarray], fallback: float, percentile: float = 99.0) -> float:
    if not values:
        return fallback
    stacked = np.concatenate([np.ravel(np.asarray(value, dtype=np.float32)) for value in values])
    vmax = float(np.nanpercentile(np.abs(stacked), percentile))
    return max(vmax, fallback)


def monthly_tos_climatology(dataset: WalkerDataset, source_idx: int, month: int) -> np.ndarray:
    """用训练年同月气候态把 TOS 转成 SSTA，使图和 Niño3.4 anomaly 对齐。"""

    payload = dataset.source_payloads[source_idx]
    train_start, train_end = dataset.data_config["train_years"]
    years = np.asarray(payload["years"])
    months = np.asarray(payload["months"])
    mask = (years >= int(train_start)) & (years <= int(train_end)) & (months == int(month))
    if not np.any(mask):
        source = dataset.source_names[source_idx]
        raise ValueError(f"No climatology samples for source={source} month={month}")
    return np.nanmean(np.asarray(payload["data"][mask, 0], dtype=np.float32), axis=0)


def valid_climatology_starts(dataset: WalkerDataset, source_idx: int, horizon: int, mode: str, split: str) -> list[int]:
    """Return target indices used to estimate the model forecast climatology."""

    payload = dataset.source_payloads[source_idx]
    years = np.asarray(payload["years"])
    starts = range(dataset.L, len(years) - horizon + 1)
    if mode == "all":
        return list(starts)

    year_key = "train_years" if mode == "train" else f"{split}_years"
    start_year, end_year = dataset.data_config[year_key]
    selected: list[int] = []
    for target_t in starts:
        target_years = years[target_t : target_t + horizon]
        if np.all((target_years >= int(start_year)) & (target_years <= int(end_year))):
            selected.append(int(target_t))
    return selected


def make_source_batch_input(
    dataset: WalkerDataset,
    source_idx: int,
    target_indices: list[int],
    device: torch.device,
) -> torch.Tensor:
    payload = dataset.source_payloads[source_idx]
    raw = np.asarray([payload["data"][target_t - dataset.L : target_t] for target_t in target_indices], dtype=np.float32)
    x = torch.from_numpy(raw).to(device=device, dtype=torch.float32)
    x = dataset._normalize_tensor(x)  # noqa: SLF001 - plotting script intentionally mirrors Dataset preprocessing
    return torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


def rollout_tos_batch(
    model: torch.nn.Module,
    dataset: WalkerDataset,
    source_idx: int,
    target_indices: list[int],
    x_norm: torch.Tensor,
    horizon: int,
    trained_rollout_steps: int,
) -> torch.Tensor:
    """Return denormalized TOS predictions with shape ``(B, lead, H, W)``."""

    payload = dataset.source_payloads[source_idx]
    months = np.asarray(payload["months"])
    preds: list[torch.Tensor] = []
    window = x_norm
    batch_size = x_norm.shape[0]
    for step in range(horizon):
        target_months = [int(months[target_t + step]) for target_t in target_indices]
        target_month = torch.tensor(target_months, dtype=torch.long, device=x_norm.device)
        rollout_step = torch.full(
            (batch_size,),
            min(step, trained_rollout_steps - 1),
            dtype=torch.long,
            device=x_norm.device,
        )
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=x_norm.device.type == "cuda"):
            pred_norm = model(window, target_month, rollout_step=rollout_step)
            pred_tos = dataset.denormalize(pred_norm)[:, 0, 0]
        preds.append(pred_tos.detach().cpu())
        window = torch.cat([window[:, 1:], pred_norm], dim=1)
    return torch.stack(preds, dim=1)


def compute_forecast_tos_climatology(
    model: torch.nn.Module,
    dataset: WalkerDataset,
    source_indices: list[int],
    horizon: int,
    trained_rollout_steps: int,
    device: torch.device,
    mode: str,
    split: str,
    batch_size: int,
) -> dict[int, np.ndarray]:
    """Compute model TOS climatology grouped by source, forecast lead, and calendar month."""

    climatology_by_source: dict[int, np.ndarray] = {}
    for source_idx in source_indices:
        payload = dataset.source_payloads[source_idx]
        starts = valid_climatology_starts(dataset, source_idx, horizon, mode, split)
        source = dataset.source_names[source_idx]
        if not starts:
            raise ValueError(f"No starts available for forecast climatology: source={source}, mode={mode}")

        h, w = payload["data"].shape[-2:]
        sums = np.zeros((horizon, 13, h, w), dtype=np.float32)
        counts = np.zeros((horizon, 13), dtype=np.int32)
        months = np.asarray(payload["months"])
        print(f"[forecast-clim] {source}: {len(starts)} starts, mode={mode}", flush=True)
        for offset in range(0, len(starts), batch_size):
            batch_starts = starts[offset : offset + batch_size]
            x_batch = make_source_batch_input(dataset, source_idx, batch_starts, device)
            preds = rollout_tos_batch(model, dataset, source_idx, batch_starts, x_batch, horizon, trained_rollout_steps).numpy()
            for batch_idx, target_t in enumerate(batch_starts):
                for lead_idx in range(horizon):
                    month = int(months[target_t + lead_idx])
                    sums[lead_idx, month] += preds[batch_idx, lead_idx]
                    counts[lead_idx, month] += 1

        climatology = np.full_like(sums, np.nan, dtype=np.float32)
        for lead_idx in range(horizon):
            for month in range(1, 13):
                if counts[lead_idx, month] > 0:
                    climatology[lead_idx, month] = sums[lead_idx, month] / float(counts[lead_idx, month])
        climatology_by_source[source_idx] = climatology
    return climatology_by_source


def load_or_compute_forecast_climatology(
    model: torch.nn.Module,
    dataset: WalkerDataset,
    source_indices: list[int],
    horizon: int,
    trained_rollout_steps: int,
    device: torch.device,
    mode: str,
    split: str,
    batch_size: int,
    cache_path: Path,
) -> dict[int, np.ndarray]:
    source_indices = sorted(set(int(item) for item in source_indices))
    if cache_path.exists():
        with np.load(cache_path) as data:
            cached_sources = [int(item) for item in data["source_indices"]]
            if (
                int(data["horizon"]) == horizon
                and str(data["mode"]) == mode
                and str(data["split"]) == split
                and all(source_idx in cached_sources for source_idx in source_indices)
            ):
                clim = np.asarray(data["climatology"], dtype=np.float32)
                print(f"[forecast-clim] using cache {cache_path}", flush=True)
                return {source_idx: clim[cached_sources.index(source_idx)] for source_idx in source_indices}

    climatology = compute_forecast_tos_climatology(
        model,
        dataset,
        source_indices,
        horizon,
        trained_rollout_steps,
        device,
        mode,
        split,
        max(1, batch_size),
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        source_indices=np.asarray(source_indices, dtype=np.int64),
        climatology=np.stack([climatology[source_idx] for source_idx in source_indices], axis=0),
        horizon=np.asarray(horizon, dtype=np.int64),
        mode=np.asarray(mode),
        split=np.asarray(split),
    )
    print(f"[forecast-clim] wrote cache {cache_path}", flush=True)
    return climatology


def add_nino34_box(ax: plt.Axes) -> None:
    lon_min, lon_max, lat_min, lat_max = NINO34_BOX
    ax.plot(
        [lon_min, lon_max, lon_max, lon_min, lon_min],
        [lat_min, lat_min, lat_max, lat_max, lat_min],
        color="#111827",
        lw=0.65,
    )


def add_panel_label(ax: plt.Axes, text: str, loc: str = "upper left") -> None:
    xy = (0.015, 0.97) if loc == "upper left" else (0.985, 0.97)
    ha = "left" if loc == "upper left" else "right"
    ax.text(
        *xy,
        text,
        transform=ax.transAxes,
        ha=ha,
        va="top",
        fontsize=6.6,
        color="#111827",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 1.3},
    )


def apply_ocean_mask(field: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    """Hide invalid land cells before plotting; model outputs there are meaningless."""

    return np.where(valid_mask, field, np.nan)


def main() -> None:
    args = parse_args()
    if args.lead_month < 1 or args.lead_month > args.horizon:
        raise ValueError(f"--lead-month must be in [1, {args.horizon}], got {args.lead_month}")

    rows = read_manifest_rows(args.manifest, args.max_cases) if args.manifest else read_summary_rows(args.cnop_dir, args.max_cases)
    if args.require_cases and len(rows) < args.require_cases:
        raise ValueError(
            f"Selected input only has {len(rows)} cases; "
            f"expected at least {args.require_cases}. Re-run CNOP with --num-cases {args.require_cases}, "
            "or pass --require-cases 0 to draw the available cases."
        )

    set_style()
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    config = load_config(args.config)
    dataset = WalkerDataset(config["data"]["path"], config, split=args.split)
    model, _checkpoint = load_model(config, args.checkpoint, device)
    trained_rollout_steps = int(args.trained_rollout_steps or config.get("training", {}).get("rollout_steps", args.horizon))
    lead_idx = args.lead_month - 1
    forecast_climatology: dict[int, np.ndarray] = {}
    if args.tos_mode == "anomaly" and args.forecast_climatology != "none":
        source_indices = [dataset.source_names.index(row["source"]) for row in rows]
        cache_path = args.forecast_climatology_cache
        if cache_path is None:
            cache_root = args.manifest.parent if args.manifest else args.cnop_dir
            cache_path = cache_root / f"forecast_tos_climatology_{args.forecast_climatology}_h{args.horizon}.npz"
        forecast_climatology = load_or_compute_forecast_climatology(
            model,
            dataset,
            source_indices,
            args.horizon,
            trained_rollout_steps,
            device,
            args.forecast_climatology,
            args.split,
            args.climatology_batch_size,
            cache_path,
        )

    cases: list[dict[str, object]] = []
    perturb_fields: list[np.ndarray] = []
    response_fields: list[np.ndarray] = []
    lead_fields: list[np.ndarray] = []

    for row in rows:
        source = row["source"]
        year = int(row["target_year"])
        target_t = int(row["target_t"])
        observed = float(row["observed_max_3m_abs"])
        case = make_case(dataset, source, year, target_t, observed)
        case_cnop_dir = Path(row.get("cnop_dir", str(args.cnop_dir)))
        delta_norm, _npz_path = load_case_npz(case_cnop_dir, source, year, args.candidate_rank)

        x0 = make_case_input(dataset, case, device)
        delta = torch.from_numpy(delta_norm).to(device=device, dtype=x0.dtype).unsqueeze(0)
        x_pert = apply_delta(x0, delta, torch.ones_like(delta, dtype=torch.bool))
        with torch.no_grad():
            perturb = (dataset.denormalize(x_pert)[:, -1, :2] - dataset.denormalize(x0)[:, -1, :2])[0]
        perturb_tos = perturb[0].detach().cpu().numpy()

        baseline = rollout_fields(model, dataset, case, x0, args.horizon, trained_rollout_steps).numpy()
        perturbed = rollout_fields(model, dataset, case, x_pert, args.horizon, trained_rollout_steps).numpy()
        payload = dataset.source_payloads[case.source_idx]
        truth = np.asarray(payload["data"][case.target_t : case.target_t + args.horizon], dtype=np.float32)
        tos_valid = payload["valid_mask"][0].cpu().numpy().astype(bool)
        response_tos = perturbed[lead_idx, 0] - baseline[lead_idx, 0]
        truth_tos = truth[lead_idx, 0]
        baseline_tos = baseline[lead_idx, 0]
        perturbed_tos = perturbed[lead_idx, 0]

        lat = np.asarray(payload["lat"], dtype=np.float64)
        lon = np.asarray(payload["lon"], dtype=np.float64)
        labels = month_labels(case, dataset, args.horizon)
        if args.tos_mode == "anomaly":
            lead_calendar_month = int(payload["months"][case.target_t + lead_idx])
            observed_climatology = monthly_tos_climatology(dataset, case.source_idx, lead_calendar_month)
            truth_tos = truth_tos - observed_climatology
            if forecast_climatology:
                model_climatology = forecast_climatology[case.source_idx][lead_idx, lead_calendar_month]
                baseline_tos = baseline_tos - model_climatology
                perturbed_tos = perturbed_tos - model_climatology
            else:
                baseline_tos = baseline_tos - observed_climatology
                perturbed_tos = perturbed_tos - observed_climatology
        truth_nino = float(compute_nino34_numpy(truth_tos[None], lat, lon)[0])
        baseline_nino = float(compute_nino34_numpy(baseline_tos[None], lat, lon)[0])
        perturbed_nino = float(compute_nino34_numpy(perturbed_tos[None], lat, lon)[0])

        perturb_tos = apply_ocean_mask(perturb_tos, tos_valid)
        response_tos = apply_ocean_mask(response_tos, tos_valid)
        truth_tos = apply_ocean_mask(truth_tos, tos_valid)
        baseline_tos = apply_ocean_mask(baseline_tos, tos_valid)
        perturbed_tos = apply_ocean_mask(perturbed_tos, tos_valid)

        perturb_plot = smooth_field(perturb_tos, args.smooth_sigma)
        response_plot = smooth_field(response_tos, args.smooth_sigma)
        truth_plot = smooth_field(truth_tos, args.smooth_sigma)
        baseline_plot = smooth_field(baseline_tos, args.smooth_sigma)
        perturbed_plot = smooth_field(perturbed_tos, args.smooth_sigma)

        perturb_fields.append(perturb_plot)
        response_fields.append(response_plot)
        lead_fields.extend([truth_plot, baseline_plot, perturbed_plot])
        cases.append(
            {
                "source": source,
                "year": year,
                "scale": row.get("scale", ""),
                "label": labels[lead_idx],
                "lat": lat,
                "lon": lon,
                "perturb": perturb_plot,
                "response": response_plot,
                "truth": truth_plot,
                "truth_nino": truth_nino,
                "baseline": baseline_plot,
                "perturbed": perturbed_plot,
                "baseline_nino": baseline_nino,
                "perturbed_nino": perturbed_nino,
            }
        )

    nrows = len(cases)
    fig, axes = plt.subplots(nrows, 4, figsize=(11.7, 1.58 * nrows + 0.9), squeeze=False)
    fig.subplots_adjust(left=0.055, right=0.94, top=0.955, bottom=0.055, wspace=0.05, hspace=0.11)

    perturb_vmax = symmetric_limit(perturb_fields, fallback=0.01)
    response_vmax = symmetric_limit(response_fields, fallback=0.01)
    tos_vmax = field_limit(lead_fields, fallback=0.3)
    perturb_levels = np.linspace(-perturb_vmax, perturb_vmax, 25)
    response_levels = np.linspace(-response_vmax, response_vmax, 25)
    tos_levels = np.linspace(-tos_vmax, tos_vmax, 31)

    tos_label = "SSTA" if args.tos_mode == "anomaly" else "TOS"
    second_title = f"Observed lead-{args.lead_month} {tos_label}" if args.second_column == "truth" else f"Lead-{args.lead_month} TOS response"
    col_titles = (
        "Initial delta TOS",
        second_title,
        f"Baseline lead-{args.lead_month} {tos_label}",
        f"Perturbed lead-{args.lead_month} {tos_label}",
    )
    mappables = [None, None, None]
    for row_idx, item in enumerate(cases):
        lat = item["lat"]
        lon = item["lon"]
        scale_label = f", scale={item['scale']}" if item["scale"] else ""
        row_label = f"{item['source']} {item['year']}{scale_label}\n{item['label']}"
        second_field = item["truth"] if args.second_column == "truth" else item["response"]
        fields = (item["perturb"], second_field, item["baseline"], item["perturbed"])
        if args.second_column == "truth":
            levels = (perturb_levels, tos_levels, tos_levels, tos_levels)
            cmaps = ("RdBu_r", TOS_CMAP, TOS_CMAP, TOS_CMAP)
        else:
            levels = (perturb_levels, response_levels, tos_levels, tos_levels)
            cmaps = ("RdBu_r", "RdBu_r", TOS_CMAP, TOS_CMAP)
        for col_idx in range(4):
            ax = axes[row_idx, col_idx]
            setup_axis(ax, show_xticks=row_idx == nrows - 1, show_yticks=col_idx == 0)
            mappable = ax.contourf(lon, lat, fields[col_idx], levels=levels[col_idx], cmap=cmaps[col_idx], extend="both")
            ax.contour(lon, lat, fields[col_idx], levels=[0.0], colors="#263238", linewidths=0.22, alpha=0.5)
            add_nino34_box(ax)
            if row_idx == 0:
                ax.set_title(col_titles[col_idx], pad=3)
            if col_idx == 0:
                ax.set_ylabel(row_label, rotation=0, ha="right", va="center", labelpad=42, fontsize=7.1)
                mappables[0] = mappable
            elif col_idx == 1:
                mappables[1] = mappable
                if args.second_column == "truth":
                    add_panel_label(ax, f"Nino3.4={item['truth_nino']:+.2f}")
            else:
                mappables[2] = mappable
                nino_value = item["baseline_nino"] if col_idx == 2 else item["perturbed_nino"]
                add_panel_label(ax, f"Nino3.4={nino_value:+.2f}")

    cb_delta_ax = fig.add_axes([0.945, 0.69, 0.011, 0.22])
    cb_resp_ax = fig.add_axes([0.945, 0.405, 0.011, 0.22])
    cb_tos_ax = fig.add_axes([0.945, 0.12, 0.011, 0.22])
    fig.colorbar(mappables[0], cax=cb_delta_ax).set_label("delta TOS", fontsize=7)
    fig.colorbar(mappables[1], cax=cb_resp_ax).set_label(f"truth {tos_label}" if args.second_column == "truth" else "response", fontsize=7)
    fig.colorbar(mappables[2], cax=cb_tos_ax).set_label(tos_label, fontsize=7)

    fig.suptitle(f"Rank-{args.candidate_rank} CNOP cases: lead-{args.lead_month} {tos_label} and Nino3.4", fontsize=10, y=0.992)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=args.dpi)
    plt.close(fig)
    print(args.output)


if __name__ == "__main__":
    main()
