"""Sample neutral CNOP cases whose baseline forecast is close to truth.

筛选逻辑：
1. 目标年必须是完整的一月到十二月，且上一年一月到十二月可作为输入。
2. truth 的 Niño3.4 anomaly 三月滑动平均不超过 neutral threshold，表示没有 ENSO。
3. 对每个 neutral case 做 baseline 12 个月滚动预报，用 Niño3.4 anomaly RMSE
   衡量 baseline 与 truth 的接近程度，取最接近的若干个。
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.cnop.compute_tos_zos_cnop import (  # noqa: E402
    NeutralCase,
    compute_nino34_numpy,
    compute_source_nino34_climatology,
    make_case_input,
    three_month_mean_np,
)
from scripts.cnop.plot_cnop_monthly_response import load_model, rollout_fields  # noqa: E402
from scripts.cnop.plot_cnop_ten_case_lead12 import load_or_compute_forecast_climatology  # noqa: E402
from src.dataset import WalkerDataset  # noqa: E402
from src.utils import load_config, set_seed  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample neutral cases with close WalkerNet baseline.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-cases", type=int, default=64)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--neutral-threshold", type=float, default=0.5)
    parser.add_argument("--case-year-range", type=str, default="")
    parser.add_argument("--max-per-source", type=int, default=0)
    parser.add_argument("--forecast-climatology-cache", type=Path, required=True)
    parser.add_argument("--climatology-batch-size", type=int, default=2)
    parser.add_argument("--trained-rollout-steps", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def parse_year_range(value: str) -> tuple[int, int] | None:
    if not value.strip():
        return None
    parts = [int(item.strip()) for item in value.split(",") if item.strip()]
    if len(parts) != 2:
        raise ValueError(f"year range must contain two comma-separated years, got {value!r}")
    return min(parts), max(parts)


def candidate_starts(dataset: WalkerDataset, year_range: tuple[int, int] | None) -> list[tuple[int, int]]:
    starts: list[tuple[int, int]] = []
    if year_range is None:
        for source_idx, target_t in dataset.sample_indices:
            starts.append((int(source_idx), int(target_t)))
        return starts

    start_year, end_year = year_range
    for source_idx, payload in enumerate(dataset.source_payloads):
        years = np.asarray(payload["years"])
        months = np.asarray(payload["months"])
        for target_t in np.where(months == 1)[0]:
            year = int(years[target_t])
            if start_year <= year <= end_year:
                starts.append((source_idx, int(target_t)))
    return starts


def complete_jan_dec_case(dataset: WalkerDataset, source_idx: int, target_t: int, horizon: int) -> bool:
    payload = dataset.source_payloads[source_idx]
    years = np.asarray(payload["years"])
    months = np.asarray(payload["months"])
    final_t = target_t + horizon - 1
    return (
        target_t - dataset.L >= 0
        and final_t < len(years)
        and int(months[target_t]) == 1
        and int(months[final_t]) == 12
        and int(years[final_t]) == int(years[target_t])
        and int(months[target_t - dataset.L]) == 1
    )


def nino_truth_anomaly(dataset: WalkerDataset, climatology: np.ndarray, source_idx: int, target_t: int, horizon: int) -> np.ndarray:
    payload = dataset.source_payloads[source_idx]
    months = np.asarray(payload["months"])[target_t : target_t + horizon]
    tos = np.asarray(payload["data"][target_t : target_t + horizon, 0], dtype=np.float32)
    raw = compute_nino34_numpy(tos, np.asarray(payload["lat"]), np.asarray(payload["lon"]))
    return raw - climatology[source_idx, months]


def nino_forecast_anomaly(fields: np.ndarray, model_climatology: np.ndarray, months: np.ndarray, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    tos = np.asarray(fields[:, 0], dtype=np.float32).copy()
    for lead_idx, month in enumerate(months):
        tos[lead_idx] = tos[lead_idx] - np.asarray(model_climatology[lead_idx, int(month)], dtype=np.float32)
    return compute_nino34_numpy(tos, lat, lon)


def load_dataset(config: dict[str, Any], split: str) -> WalkerDataset:
    data_config = config.get("data", config)
    return WalkerDataset(data_config.get("data_path"), config, split=split)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = load_config(args.config)
    device = torch.device(args.device)

    dataset = load_dataset(config, args.split)
    model, _checkpoint = load_model(config, args.checkpoint, device)
    trained_rollout_steps = int(args.trained_rollout_steps or config.get("training", {}).get("rollout_steps", args.horizon))
    observed_climatology = compute_source_nino34_climatology(dataset)
    forecast_climatology = load_or_compute_forecast_climatology(
        model,
        dataset,
        list(range(len(dataset.source_names))),
        args.horizon,
        trained_rollout_steps,
        device,
        "train",
        args.split,
        args.climatology_batch_size,
        args.forecast_climatology_cache,
    )

    rows: list[dict[str, Any]] = []
    starts = candidate_starts(dataset, parse_year_range(args.case_year_range))
    for idx, (source_idx, target_t) in enumerate(starts, start=1):
        if not complete_jan_dec_case(dataset, source_idx, target_t, args.horizon):
            continue
        payload = dataset.source_payloads[source_idx]
        months = np.asarray(payload["months"])[target_t : target_t + args.horizon]
        year = int(payload["years"][target_t])
        truth = nino_truth_anomaly(dataset, observed_climatology, source_idx, target_t, args.horizon)
        truth_3m = three_month_mean_np(truth)
        observed_max_abs = float(np.nanmax(np.abs(truth_3m)))
        if observed_max_abs > args.neutral_threshold:
            continue

        case = NeutralCase(
            source_idx=source_idx,
            source_name=dataset.source_names[source_idx],
            target_t=target_t,
            target_year=year,
            neutral_score=observed_max_abs,
            observed_max_3m_abs=observed_max_abs,
        )
        x0 = make_case_input(dataset, case, device)
        baseline = rollout_fields(model, dataset, case, x0, args.horizon, trained_rollout_steps).numpy()
        baseline_nino = nino_forecast_anomaly(
            baseline,
            forecast_climatology[source_idx],
            months,
            np.asarray(payload["lat"]),
            np.asarray(payload["lon"]),
        )
        diff = baseline_nino - truth
        baseline_rmse = float(np.sqrt(np.nanmean(diff * diff)))
        baseline_mae = float(np.nanmean(np.abs(diff)))
        baseline_lead12_abs_error = float(abs(baseline_nino[args.horizon - 1] - truth[args.horizon - 1]))
        baseline_3m = three_month_mean_np(baseline_nino)

        rows.append(
            {
                "rank_source_order": idx,
                "source": dataset.source_names[source_idx],
                "source_idx": source_idx,
                "target_year": year,
                "target_t": target_t,
                "observed_max_3m_abs": observed_max_abs,
                "truth_lead12_nino": float(truth[args.horizon - 1]),
                "baseline_lead12_nino": float(baseline_nino[args.horizon - 1]),
                "baseline_lead12_abs_error": baseline_lead12_abs_error,
                "truth_max_3m": float(np.nanmax(truth_3m)),
                "baseline_max_3m": float(np.nanmax(baseline_3m)),
                "baseline_truth_nino_rmse": baseline_rmse,
                "baseline_truth_nino_mae": baseline_mae,
                "selection_score": baseline_rmse + 0.25 * baseline_lead12_abs_error,
            }
        )
        if len(rows) % 25 == 0:
            print(f"[sample] neutral candidates scored: {len(rows)}", flush=True)

    if len(rows) < args.num_cases:
        raise RuntimeError(f"Only {len(rows)} neutral candidates found; need {args.num_cases}.")

    rows = sorted(rows, key=lambda row: (row["selection_score"], row["baseline_truth_nino_rmse"], row["observed_max_3m_abs"]))
    selected: list[dict[str, Any]] = []
    source_counts: dict[str, int] = {}
    max_per_source = int(args.max_per_source)
    for row in rows:
        source = str(row["source"])
        if max_per_source > 0 and source_counts.get(source, 0) >= max_per_source:
            continue
        selected.append(row)
        source_counts[source] = source_counts.get(source, 0) + 1
        if len(selected) >= args.num_cases:
            break
    if len(selected) < args.num_cases:
        selected = rows[: args.num_cases]

    all_path = args.output_dir / "neutral_baseline_candidates.csv"
    selected_path = args.output_dir / "selected_cases.csv"
    fields = list(rows[0].keys())
    with all_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with selected_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(selected)

    case_lines = [f'{row["source"]} {int(row["target_year"])} {idx % 8}' for idx, row in enumerate(selected)]
    (args.output_dir / "selected_cases_for_bash.txt").write_text("\n".join(case_lines) + "\n", encoding="utf-8")
    print(f"[sample] candidates={len(rows)} selected={len(selected)}", flush=True)
    print(f"[sample] selected source counts={source_counts}", flush=True)
    print(f"[sample] wrote {selected_path}", flush=True)


if __name__ == "__main__":
    main()
