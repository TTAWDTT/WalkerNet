"""Evaluate equal-radius random TOS/ZOS perturbations for one basin experiment."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.cnop.compute_tos_zos_cnop import (  # noqa: E402
    build_domain_mask,
    compute_source_nino34_climatology,
    evaluate_delta,
    expand_delta,
    load_model,
    make_case_input,
    measure_constraint_norm,
    normalized_delta_to_event_constraint_tensor,
    prepare_event_l2_constraint,
    rollout_nino_anomaly,
    select_specific_case,
    three_month_mean,
)
from src.dataset import WalkerDataset  # noqa: E402
from src.utils import load_config, set_seed  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--constraint-file", required=True)
    parser.add_argument("--constraint-scale", type=float, default=0.1)
    parser.add_argument("--domain", choices=("pacific", "atlantic_indian", "global"), required=True)
    parser.add_argument("--case-source-name", default="GFDL-ESM4")
    parser.add_argument("--case-target-year", type=int, default=1995)
    parser.add_argument("--num-controls", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1042)
    parser.add_argument("--basin-lat-bounds", default="-60,60")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    args.constraint_mode = "event_l2"
    args.relative_l2_fraction = 0.0
    args.epsilon_tos = 0.1
    args.epsilon_zos = 0.1
    args.max_abs = 2.0
    args.horizon = 12
    args.objective_mode = "late_3m_delta"
    args.objective_lead = 12
    args.objective_temperature = 0.25
    args.amp = True

    device = torch.device(args.device)
    config = load_config(args.config)
    dataset = WalkerDataset(config["data"]["path"], config, split="test")
    prepare_event_l2_constraint(dataset, args)
    model, checkpoint = load_model(config, args.checkpoint, device)
    trained_rollout_steps = int(config["training"]["rollout_steps"])
    climatology_np = compute_source_nino34_climatology(dataset)
    climatology = torch.from_numpy(climatology_np).to(device=device, dtype=torch.float32)
    case = select_specific_case(dataset, climatology_np, args.case_source_name, args.case_target_year, args.horizon)
    x0 = make_case_input(dataset, case, device)
    lat = torch.as_tensor(dataset.source_payloads[case.source_idx]["lat"], device=device, dtype=torch.float32)
    lon = torch.as_tensor(dataset.source_payloads[case.source_idx]["lon"], device=device, dtype=torch.float32)
    basin_lat_bounds = tuple(float(item) for item in args.basin_lat_bounds.split(","))
    mask = build_domain_mask(dataset, case, args.domain, (-20, 20), (120, 290), device, basin_lat_bounds)

    with torch.no_grad():
        baseline_nino = rollout_nino_anomaly(
            model, dataset, case, x0, climatology, args.horizon, trained_rollout_steps, lat, lon, use_amp=True
        )
        baseline_3m = three_month_mean(baseline_nino)

    target_hw = (x0.shape[-2], x0.shape[-1])
    param_hw = (math.ceil(target_hw[0] / 4), math.ceil(target_hw[1] / 4))
    rows: list[dict[str, float | int | str]] = []
    for control_idx in range(args.num_controls):
        generator = torch.Generator(device=device)
        generator.manual_seed(args.seed + control_idx)
        delta_param = torch.randn((1, 2, *param_hw), generator=generator, device=device, dtype=x0.dtype)
        full_delta = expand_delta(delta_param, target_hw, "patch")
        dimless = normalized_delta_to_event_constraint_tensor(dataset, case.source_idx, full_delta, args)
        norm = torch.sqrt((dimless.square() * mask.to(dimless.dtype)).sum()).clamp_min(1.0e-12)
        delta_param.mul_(float(args.event_constraint_l2) / norm)
        full_delta = expand_delta(delta_param, target_hw, "patch") * mask.to(x0.dtype)
        constraint_norm, constraint_radius = measure_constraint_norm(
            dataset, case, x0, full_delta, mask, (0.1, 0.1), args
        )
        with torch.no_grad():
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
        rows.append(
            {
                "domain": args.domain,
                "control_idx": control_idx,
                "seed": args.seed + control_idx,
                "objective": float(metrics["objective"].item()),
                "lead_delta": float(metrics["lead_delta"].item()),
                "cnop_max_3m": float(metrics["max_3m"].item()),
                "gain_max_3m": float(metrics["gain_max_3m"].item()),
                "constraint_norm": constraint_norm,
                "constraint_radius": constraint_radius,
                "constraint_ratio": constraint_norm / constraint_radius,
            }
        )
        if (control_idx + 1) % 16 == 0:
            print(f"[random] {args.domain} {control_idx + 1}/{args.num_controls}", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[random] checkpoint_epoch={checkpoint.get('epoch')} wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
