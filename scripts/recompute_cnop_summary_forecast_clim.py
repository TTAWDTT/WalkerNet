"""Recompute CNOP Niño3.4 summaries with model forecast climatology.

CNOP optimization can use any anomaly reference, because the lead-delta objective
subtracts baseline from perturbed forecasts. For reporting, however, model
forecasts should be converted to anomaly using the model-world forecast
climatology, while observations still use observed climatology.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.compute_tos_zos_cnop import compute_nino34_numpy, three_month_mean_np  # noqa: E402
from scripts.plot_cnop_monthly_response import apply_delta, load_model, make_case, make_case_input, rollout_fields  # noqa: E402
from src.dataset import WalkerDataset  # noqa: E402
from src.utils import load_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recompute CNOP summaries with forecast climatology.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cnop-dir", type=Path, required=True)
    parser.add_argument("--forecast-climatology-cache", type=Path, required=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--lead-month", type=int, default=12)
    parser.add_argument("--trained-rollout-steps", type=int, default=0)
    parser.add_argument("--summary-name", type=str, default="cnop_summary_forecast_clim.csv")
    parser.add_argument("--candidate-summary-name", type=str, default="cnop_candidate_summary_forecast_clim.csv")
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_forecast_climatology(cache_path: Path) -> tuple[list[int], np.ndarray]:
    with np.load(cache_path) as data:
        return [int(item) for item in np.asarray(data["source_indices"]).tolist()], np.asarray(data["climatology"], dtype=np.float32)


def nino_anomaly_series(
    tos_fields: np.ndarray,
    model_climatology: np.ndarray,
    target_months: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
) -> np.ndarray:
    values: list[float] = []
    for lead_idx in range(tos_fields.shape[0]):
        month = int(target_months[lead_idx])
        anomaly_field = tos_fields[lead_idx] - model_climatology[lead_idx, month]
        values.append(float(compute_nino34_numpy(anomaly_field[None], lat, lon)[0]))
    return np.asarray(values, dtype=np.float32)


def write_summary(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
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
        "mask_count",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fieldnames})


def write_candidate_summary(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "source",
        "target_year",
        "rank",
        "start_idx",
        "objective",
        "baseline_lead_nino",
        "cnop_lead_nino",
        "lead_delta",
        "baseline_max_3m",
        "cnop_max_3m",
        "gain_max_3m",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fieldnames})


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    config = load_config(args.config)
    dataset = WalkerDataset(config["data"]["path"], config, split=args.split)
    model, checkpoint = load_model(config, args.checkpoint, device)
    trained_rollout_steps = int(args.trained_rollout_steps or config.get("training", {}).get("rollout_steps", args.horizon))
    lead_idx = min(max(int(args.lead_month), 1), int(args.horizon)) - 1

    cached_sources, climatology_stack = load_forecast_climatology(args.forecast_climatology_cache)
    source_to_cache_idx = {source_idx: idx for idx, source_idx in enumerate(cached_sources)}

    summary_rows = read_rows(args.cnop_dir / "cnop_summary.csv")
    fixed_rows: list[dict[str, object]] = []
    fixed_candidate_rows: list[dict[str, object]] = []

    print(f"[recompute] checkpoint_epoch={checkpoint.get('epoch')} rows={len(summary_rows)}", flush=True)
    for row in summary_rows:
        source = row["source"]
        year = int(row["target_year"])
        target_t = int(row["target_t"])
        observed = float(row["observed_max_3m_abs"])
        case = make_case(dataset, source, year, target_t, observed)
        payload = dataset.source_payloads[case.source_idx]
        lat = np.asarray(payload["lat"], dtype=np.float64)
        lon = np.asarray(payload["lon"], dtype=np.float64)
        months = np.asarray(payload["months"][target_t : target_t + args.horizon], dtype=np.int64)
        model_clim = climatology_stack[source_to_cache_idx[case.source_idx]]

        x0 = make_case_input(dataset, case, device)
        baseline = rollout_fields(model, dataset, case, x0, args.horizon, trained_rollout_steps).numpy()
        baseline_nino = nino_anomaly_series(baseline[:, 0], model_clim, months, lat, lon)
        baseline_3m = three_month_mean_np(baseline_nino)
        baseline_max_3m = float(np.nanmax(baseline_3m))
        baseline_lead = float(baseline_nino[lead_idx])

        npz_path = args.cnop_dir / f"case_{source}_{year}.npz"
        with np.load(npz_path) as npz:
            top_delta_norm = np.asarray(npz["top_delta_norm"], dtype=np.float32)
            top_start_idx = np.asarray(npz["top_start_idx"], dtype=np.int32)
            top_objective = np.asarray(npz["top_objective"], dtype=np.float32)

        best_row: dict[str, object] | None = None
        for rank, delta_norm in enumerate(top_delta_norm, start=1):
            delta = torch.from_numpy(delta_norm).to(device=device, dtype=x0.dtype).unsqueeze(0)
            x_pert = apply_delta(x0, delta, torch.ones_like(delta, dtype=torch.bool))
            perturbed = rollout_fields(model, dataset, case, x_pert, args.horizon, trained_rollout_steps).numpy()
            cnop_nino = nino_anomaly_series(perturbed[:, 0], model_clim, months, lat, lon)
            cnop_3m = three_month_mean_np(cnop_nino)
            cnop_max_3m = float(np.nanmax(cnop_3m))
            cnop_lead = float(cnop_nino[lead_idx])
            candidate_row = {
                "source": source,
                "target_year": year,
                "rank": rank,
                "start_idx": int(top_start_idx[rank - 1]),
                "objective": float(top_objective[rank - 1]),
                "baseline_lead_nino": baseline_lead,
                "cnop_lead_nino": cnop_lead,
                "lead_delta": cnop_lead - baseline_lead,
                "baseline_max_3m": baseline_max_3m,
                "cnop_max_3m": cnop_max_3m,
                "gain_max_3m": cnop_max_3m - baseline_max_3m,
            }
            fixed_candidate_rows.append(candidate_row)
            if rank == 1:
                best_row = candidate_row

        if best_row is None:
            raise RuntimeError(f"No candidates found for {source} {year}")
        fixed_rows.append(
            {
                "source": source,
                "target_year": year,
                "target_t": target_t,
                "observed_max_3m_abs": observed,
                "baseline_max_3m": best_row["baseline_max_3m"],
                "baseline_lead_nino": best_row["baseline_lead_nino"],
                "cnop_lead_nino": best_row["cnop_lead_nino"],
                "lead_delta": best_row["lead_delta"],
                "cnop_max_3m": best_row["cnop_max_3m"],
                "gain_max_3m": best_row["gain_max_3m"],
                "best_start_idx": best_row["start_idx"],
                "best_objective": best_row["objective"],
                "mask_count": row["mask_count"],
            }
        )
        print(
            f"[recompute] {source} {year}: baseline_lead={baseline_lead:+.3f} "
            f"cnop_lead={best_row['cnop_lead_nino']:+.3f} delta={best_row['lead_delta']:+.3f}",
            flush=True,
        )

    write_summary(args.cnop_dir / args.summary_name, fixed_rows)
    write_candidate_summary(args.cnop_dir / args.candidate_summary_name, fixed_candidate_rows)
    print(args.cnop_dir / args.summary_name, flush=True)
    print(args.cnop_dir / args.candidate_summary_name, flush=True)


if __name__ == "__main__":
    main()
