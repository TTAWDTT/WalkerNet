"""Audit lead-specific CNOP forecast maps against the reported Niño3.4 values.

The audit is deliberately numerical and precedes plotting.  It verifies that
truth uses observed source/month climatology and both forecast branches use one
shared model source/lead/month climatology; it then compares the spatial
Niño3.4 means with the selected summary table.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.cnop.compute_tos_zos_cnop import compute_nino34_numpy  # noqa: E402
from scripts.cnop.forecast_field_climatology import (  # noqa: E402
    load_or_compute_forecast_field_climatology,
    monthly_observed_field_climatology,
)
from scripts.cnop.plot_cnop_monthly_response import (  # noqa: E402
    apply_delta,
    load_case_npz,
    load_model,
    make_case,
    make_case_input,
    rollout_fields,
)
from src.dataset import WalkerDataset  # noqa: E402
from src.utils import load_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit CNOP lead-forecast field and summary consistency.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--forecast-climatology-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--lead-month", type=int, default=12)
    parser.add_argument("--trained-rollout-steps", type=int, default=0)
    parser.add_argument("--climatology-batch-size", type=int, default=2)
    parser.add_argument("--summary-name", default="cnop_summary.csv")
    return parser.parse_args()


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {"source", "target_year", "cnop_dir"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"{path} must contain source,target_year,cnop_dir")
    return rows


def summary_row(cnop_dir: Path, source: str, year: int, name: str) -> dict[str, str]:
    with (cnop_dir / name).open("r", newline="", encoding="utf-8") as handle:
        matches = [row for row in csv.DictReader(handle) if row["source"] == source and int(row["target_year"]) == year]
    if len(matches) != 1:
        raise ValueError(f"Expected one {source} {year} row in {cnop_dir / name}, found {len(matches)}")
    return matches[0]


def optional_float(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    return float(value) if value not in ("", None) else float("nan")


def main() -> None:
    args = parse_args()
    selections = read_manifest(args.manifest)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    config = load_config(args.config)
    dataset = WalkerDataset(config["data"]["path"], config, split=args.split)
    model, _checkpoint = load_model(config, args.checkpoint, device)
    source_indices = [dataset.source_names.index(row["source"]) for row in selections]
    trained_rollout_steps = int(args.trained_rollout_steps or config.get("training", {}).get("rollout_steps", args.horizon))
    forecast_climatology = load_or_compute_forecast_field_climatology(
        model,
        dataset,
        source_indices,
        args.horizon,
        trained_rollout_steps,
        device,
        "train",
        args.split,
        args.climatology_batch_size,
        args.forecast_climatology_cache,
    )
    lead_idx = min(max(args.lead_month, 1), args.horizon) - 1
    audited: list[dict[str, object]] = []
    for selection in selections:
        source = selection["source"]
        year = int(selection["target_year"])
        cnop_dir = Path(selection["cnop_dir"])
        summary = summary_row(cnop_dir, source, year, selection.get("summary_name", args.summary_name))
        case = make_case(
            dataset,
            source,
            year,
            int(summary["target_t"]),
            float(summary["observed_max_3m_abs"]),
        )
        payload = dataset.source_payloads[case.source_idx]
        target_months = np.asarray(payload["months"][case.target_t : case.target_t + args.horizon], dtype=np.int64)
        observed_climatology = monthly_observed_field_climatology(dataset, case.source_idx, target_months)
        model_climatology = forecast_climatology[case.source_idx]
        lat = np.asarray(payload["lat"], dtype=np.float64)
        lon = np.asarray(payload["lon"], dtype=np.float64)
        truth = np.asarray(payload["data"][case.target_t : case.target_t + args.horizon], dtype=np.float32)
        x0 = make_case_input(dataset, case, device)
        delta_norm, _ = load_case_npz(cnop_dir, source, year, 1)
        delta = torch.from_numpy(delta_norm).to(device=device, dtype=x0.dtype).unsqueeze(0)
        x_perturbed = apply_delta(x0, delta, torch.ones_like(delta, dtype=torch.bool))
        baseline = rollout_fields(model, dataset, case, x0, args.horizon, trained_rollout_steps).numpy()
        perturbed = rollout_fields(model, dataset, case, x_perturbed, args.horizon, trained_rollout_steps).numpy()
        month = int(target_months[lead_idx])
        truth_nino = float(compute_nino34_numpy((truth[lead_idx, 0] - observed_climatology[lead_idx, 0])[None], lat, lon)[0])
        baseline_nino = float(compute_nino34_numpy((baseline[lead_idx, 0] - model_climatology[lead_idx, month, 0])[None], lat, lon)[0])
        perturbed_nino = float(compute_nino34_numpy((perturbed[lead_idx, 0] - model_climatology[lead_idx, month, 0])[None], lat, lon)[0])
        reported_baseline = optional_float(summary, "baseline_lead_nino")
        reported_perturbed = optional_float(summary, "cnop_lead_nino")
        audited.append(
            {
                "scale": selection.get("scale", ""),
                "source": source,
                "target_year": year,
                "target_t": case.target_t,
                "lead_month": args.lead_month,
                "calendar_month": month,
                "truth_nino34_anomaly": truth_nino,
                "baseline_nino34_anomaly": baseline_nino,
                "perturbed_nino34_anomaly": perturbed_nino,
                "response_nino34": perturbed_nino - baseline_nino,
                "reported_baseline_nino34": reported_baseline,
                "reported_perturbed_nino34": reported_perturbed,
                "baseline_abs_error": abs(baseline_nino - reported_baseline) if np.isfinite(reported_baseline) else float("nan"),
                "perturbed_abs_error": abs(perturbed_nino - reported_perturbed) if np.isfinite(reported_perturbed) else float("nan"),
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(audited[0]))
        writer.writeheader()
        writer.writerows(audited)
    print(args.output)
    for row in audited:
        print(
            f"[audit] scale={row['scale']} {row['source']} {row['target_year']}: "
            f"truth={row['truth_nino34_anomaly']:+.3f}, base={row['baseline_nino34_anomaly']:+.3f}, "
            f"pert={row['perturbed_nino34_anomaly']:+.3f}, "
            f"summary-errors=({row['baseline_abs_error']:.3g},{row['perturbed_abs_error']:.3g})",
            flush=True,
        )


if __name__ == "__main__":
    main()
