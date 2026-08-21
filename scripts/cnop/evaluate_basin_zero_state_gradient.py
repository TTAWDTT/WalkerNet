"""Evaluate the zero-state local-gradient baseline for a basin CNOP case.

This is deliberately a *single* first-order baseline.  The objective gradient
is evaluated at an unperturbed input (``delta = 0``), represented on the same
45 x 90 TOS/ZOS patch grid as the CNOP search.  That direction is then scaled
to the same event-L2 perturbation budget and clipped with the same normalized
``max_abs`` bound before one 12-month rollout is evaluated.

The output schema keeps the physical fields and Niño3.4 series needed for a
matched CNOP--gradient--random comparison.  It must not be interpreted as an
iterative gradient-ascent method: taking more than this one local direction
would no longer be the requested zero-state linear baseline.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
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
    build_domain_mask,
    compute_source_nino34_climatology,
    evaluate_delta,
    expand_delta,
    load_model,
    make_case_input,
    measure_constraint_norm,
    normalized_delta_to_event_constraint_tensor,
    normalized_delta_to_physical,
    parse_bounds,
    prepare_event_l2_constraint,
    rollout_nino_anomaly,
    select_specific_case,
    three_month_mean,
)
from src.dataset import WalkerDataset  # noqa: E402
from src.utils import load_config, set_seed  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a zero-state local-gradient CNOP baseline.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--constraint-file", required=True)
    parser.add_argument("--constraint-scale", type=float, default=0.1)
    parser.add_argument("--domain", choices=("pacific", "atlantic_indian", "global"), required=True)
    parser.add_argument("--case-source-name", default="GFDL-ESM4")
    parser.add_argument("--case-target-year", type=int, default=1995)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--basin-lat-bounds", default="-60,60")
    parser.add_argument("--perturb-grid", choices=("patch",), default="patch")
    parser.add_argument("--perturb-patch-size", type=int, default=4)
    parser.add_argument("--max-abs", type=float, default=2.0)
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    parser.set_defaults(amp=True)
    return parser.parse_args()


def scale_zero_state_gradient_to_event_radius(
    gradient_param: torch.Tensor,
    *,
    dataset: WalkerDataset,
    case: NeutralCase,
    mask: torch.Tensor,
    target_hw: tuple[int, int],
    args: argparse.Namespace,
) -> tuple[torch.Tensor, bool, bool]:
    """Turn a patch-parameter gradient into the matched finite-amplitude delta.

    The L2 norm is evaluated after bilinear upsampling and with the basin/ocean
    mask, exactly as for the CNOP hard event-L2 constraint.  Elementwise
    clipping is applied after radius scaling, matching ``project_delta_param``.
    The returned flags state whether radius projection and clipping changed the
    local direction; clipping can leave a final norm below the nominal radius.
    """

    if args.constraint_mode != "event_l2":
        raise ValueError("The matched workshop gradient baseline requires constraint_mode=event_l2")
    if not torch.isfinite(gradient_param).all():
        raise ValueError("Zero-state objective gradient contains non-finite values")

    direction = gradient_param.detach().clone()
    full_direction = expand_delta(direction, target_hw, args.perturb_grid)
    dimless = normalized_delta_to_event_constraint_tensor(dataset, case.source_idx, full_direction, args)
    norm = torch.sqrt((dimless.square() * mask.to(dimless.dtype)).sum())
    if float(norm.item()) <= 1.0e-12:
        raise ValueError("Zero-state objective gradient has zero norm inside the requested basin mask")

    radius = float(args.event_constraint_l2)
    direction.mul_(radius / norm)
    before_clip = direction.clone()
    direction.clamp_(min=-float(args.max_abs), max=float(args.max_abs))
    clipped = not torch.equal(direction, before_clip)
    return direction, True, clipped


def result_row(
    case: NeutralCase,
    metrics: dict[str, Any],
    baseline_3m: torch.Tensor,
    constraint_norm: float,
    constraint_radius: float,
    projected: bool,
    max_abs_clipped: bool,
    mask: torch.Tensor,
) -> dict[str, Any]:
    return {
        "method": "zero_state_local_gradient",
        "source": case.source_name,
        "target_year": case.target_year,
        "target_t": case.target_t,
        "objective_mode": "late_3m_delta",
        "objective": float(metrics["objective"].item()),
        "baseline_max_3m": float(baseline_3m.max().item()),
        "baseline_lead_nino": float(metrics["lead_nino"].item() - metrics["lead_delta"].item()),
        "gradient_lead_nino": float(metrics["lead_nino"].item()),
        "lead_delta": float(metrics["lead_delta"].item()),
        "gradient_max_3m": float(metrics["max_3m"].item()),
        "gain_max_3m": float(metrics["gain_max_3m"].item()),
        "constraint_norm": constraint_norm,
        "constraint_radius": constraint_radius,
        "constraint_ratio": constraint_norm / max(constraint_radius, 1.0e-12),
        "projected": projected,
        "max_abs_clipped": max_abs_clipped,
        "mask_count": int(mask[0, 0].sum().item()),
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    args.constraint_mode = "event_l2"
    args.relative_l2_fraction = 0.0
    args.epsilon_tos = 0.1
    args.epsilon_zos = 0.1
    args.horizon = 12
    args.objective_mode = "late_3m_delta"
    args.objective_lead = 12
    args.objective_temperature = 0.25
    args.smoothness_weight = 0.0
    args.checkpoint_rollout = False

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
    mask = build_domain_mask(
        dataset,
        case,
        args.domain,
        (-20.0, 20.0),
        (120.0, 290.0),
        device,
        parse_bounds(args.basin_lat_bounds),
    )
    target_hw = (x0.shape[-2], x0.shape[-1])
    param_hw = tuple(math.ceil(size / int(args.perturb_patch_size)) for size in target_hw)
    if param_hw != (45, 90):
        raise ValueError(f"Workshop baseline requires a 45x90 patch grid, got {param_hw} from target={target_hw}")

    with torch.no_grad():
        baseline_nino = rollout_nino_anomaly(
            model, dataset, case, x0, climatology, args.horizon, trained_rollout_steps, lat, lon, use_amp=args.amp
        )
        baseline_3m = three_month_mean(baseline_nino)

    # One local derivative at delta=0; no iterative optimization is performed.
    delta_param = torch.zeros((1, 2, *param_hw), dtype=x0.dtype, device=device, requires_grad=True)
    zero_delta = expand_delta(delta_param, target_hw, args.perturb_grid)
    zero_metrics = evaluate_delta(
        model, dataset, case, x0, zero_delta, mask, climatology, args, trained_rollout_steps,
        lat, lon, baseline_nino, baseline_3m, use_checkpoint=False,
    )
    zero_metrics["objective"].backward()
    if delta_param.grad is None:
        raise RuntimeError("No gradient was produced for the zero-state patch parameters")
    direction_param, projected, max_abs_clipped = scale_zero_state_gradient_to_event_radius(
        delta_param.grad, dataset=dataset, case=case, mask=mask, target_hw=target_hw, args=args
    )
    final_delta = expand_delta(direction_param, target_hw, args.perturb_grid) * mask.to(dtype=x0.dtype)
    constraint_norm, constraint_radius = measure_constraint_norm(
        dataset, case, x0, final_delta, mask, (args.epsilon_tos, args.epsilon_zos), args
    )
    with torch.no_grad():
        metrics = evaluate_delta(
            model, dataset, case, x0, final_delta, mask, climatology, args, trained_rollout_steps,
            lat, lon, baseline_nino, baseline_3m, use_checkpoint=False,
        )

    row = result_row(
        case, metrics, baseline_3m, constraint_norm, constraint_radius, projected, max_abs_clipped, mask
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    npz_path = args.output_dir / f"case_{case.source_name}_{case.target_year}.npz"
    np.savez_compressed(
        npz_path,
        method=np.asarray("zero_state_local_gradient"),
        delta_norm=final_delta.detach().cpu().numpy()[0],
        delta_phys=normalized_delta_to_physical(dataset, final_delta.detach())[0],
        raw_gradient_param=delta_param.grad.detach().cpu().numpy()[0],
        baseline_nino=baseline_nino.detach().cpu().numpy(),
        gradient_nino=metrics["nino"].detach().cpu().numpy(),
        baseline_3m=baseline_3m.detach().cpu().numpy(),
        gradient_3m=metrics["three_month"].detach().cpu().numpy(),
        objective=np.asarray(row["objective"], dtype=np.float32),
        lead_delta=np.asarray(row["lead_delta"], dtype=np.float32),
        constraint_norm=np.asarray(constraint_norm, dtype=np.float32),
        constraint_radius=np.asarray(constraint_radius, dtype=np.float32),
        constraint_ratio=np.asarray(row["constraint_ratio"], dtype=np.float32),
        projected=np.asarray(projected),
        max_abs_clipped=np.asarray(max_abs_clipped),
        lat=np.asarray(dataset.source_payloads[case.source_idx]["lat"]),
        lon=np.asarray(dataset.source_payloads[case.source_idx]["lon"]),
    )
    csv_path = args.output_dir / "gradient_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    method_path = args.output_dir / "method.json"
    method_path.write_text(
        json.dumps(
            {
                "method": "zero_state_local_gradient",
                "description": "One local objective gradient at delta=0, then event-L2 scaling and max-abs clipping.",
                "objective_mode": args.objective_mode,
                "horizon": args.horizon,
                "perturbed_input": "final input month, normalized TOS and ZOS only",
                "perturb_grid": args.perturb_grid,
                "patch_grid": list(param_hw),
                "domain": args.domain,
                "constraint_file": args.constraint_file,
                "constraint_scale": args.constraint_scale,
                "constraint_radius": constraint_radius,
                "max_abs": args.max_abs,
                "checkpoint": args.checkpoint,
                "checkpoint_epoch": checkpoint.get("epoch"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        f"[gradient] {case.source_name} {case.target_year} domain={args.domain} "
        f"objective={row['objective']:.6f} norm/radius={row['constraint_ratio']:.6f} "
        f"wrote {npz_path} and {csv_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
