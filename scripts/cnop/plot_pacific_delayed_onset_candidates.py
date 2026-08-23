"""Plot paired Pacific CNOP top-3 response-evolution candidate panels."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import TwoSlopeNorm

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.cnop.compute_tos_zos_cnop import (  # noqa: E402
    compute_source_nino34_climatology,
    make_case_input,
    select_specific_case,
    target_month_tensor,
)
from src.dataset import WalkerDataset  # noqa: E402
from src.model import WalkerNet  # noqa: E402
from src.utils import load_config  # noqa: E402


def rollout_fields(model: WalkerNet, x0: torch.Tensor, dataset: WalkerDataset, case: object, horizon: int, trained_steps: int) -> torch.Tensor:
    values: list[torch.Tensor] = []
    window = x0
    with torch.inference_mode():
        for step in range(horizon):
            month = target_month_tensor(dataset, case, step, x0.device)
            rollout_step = torch.tensor([min(step, trained_steps - 1)], dtype=torch.long, device=x0.device)
            prediction = model(window, month, rollout_step=rollout_step)
            values.append(prediction[:, 0])
            window = torch.cat([window[:, 1:], prediction], dim=1)
    return torch.stack(values, dim=1)


def load_case_response(
    model: WalkerNet,
    dataset: WalkerDataset,
    case: object,
    artifact_path: Path,
    device: torch.device,
    std: torch.Tensor,
    trained_steps: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x0 = make_case_input(dataset, case, device)
    baseline = rollout_fields(model, x0, dataset, case, 12, trained_steps)
    with np.load(artifact_path) as artifact:
        deltas = np.asarray(artifact["top_delta_norm"], dtype=np.float32)
        starts = np.asarray(artifact["top_start_idx"], dtype=np.int64)
        lead_delta = np.asarray(artifact["top_lead_delta"], dtype=np.float32)
        baseline_nino = np.asarray(artifact["baseline_nino"], dtype=np.float32)
        lat = np.asarray(artifact["lat"], dtype=np.float32)
        lon = np.asarray(artifact["lon"], dtype=np.float32)
    responses: list[np.ndarray] = []
    for delta in deltas[:3]:
        perturbed_x0 = x0.clone()
        perturbed_x0[:, -1, :2] += torch.as_tensor(delta, dtype=x0.dtype, device=device).unsqueeze(0)
        perturbed = rollout_fields(model, perturbed_x0, dataset, case, 12, trained_steps)
        response = ((perturbed - baseline) * std).squeeze(0).detach().cpu().numpy().astype(np.float32)
        responses.append(response[:, :2])
    return np.stack(responses), starts[:3], lead_delta[:3], baseline_nino, lat, lon


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    config = load_config(args.config)
    dataset = WalkerDataset(config["data"]["path"], config, split="train")
    climatology = compute_source_nino34_climatology(dataset)
    model = WalkerNet(config).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    # WalkerDataset stores per-variable statistics as a 1-D vector.  Keep the
    # broadcast dimensions explicit so response fields are denormalised over
    # (batch, lead, variable, latitude, longitude) without assuming a spatial
    # statistics array.
    std = dataset._std[:4].to(device=device, dtype=torch.float32).view(1, 1, 4, 1, 1)  # noqa: SLF001
    trained_steps = int(config.get("training", {}).get("rollout_steps", 18))
    with (args.root / "metadata" / "formal_manifest_v1.csv").open(newline="", encoding="utf-8") as handle:
        manifest = list(csv.DictReader(handle))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    index: list[dict[str, object]] = []
    for case_row in manifest:
        source, year = case_row["source"], int(case_row["target_year"])
        key = f"{source}_{year}"
        case = select_specific_case(dataset, climatology, source, year, 12)
        branch_data = {}
        for branch in ("normal", "delayed"):
            artifact_path = args.root / branch / key / f"case_{key}.npz"
            branch_data[branch] = load_case_response(model, dataset, case, artifact_path, device, std, trained_steps)
        all_response = np.concatenate([branch_data["normal"][0], branch_data["delayed"][0]], axis=0)
        tos_vmax = max(0.6, float(np.nanpercentile(np.abs(all_response[:, :, 0]), 99.5)))
        zos_vmax = max(0.8, float(np.nanpercentile(np.abs(all_response[:, :, 1]), 99.5)))
        tos_vmax = min(tos_vmax, 1.2)
        zos_vmax = min(zos_vmax, 2.0)
        fig = plt.figure(figsize=(24, 10), constrained_layout=False)
        outer = fig.add_gridspec(2, 3, left=0.025, right=0.93, bottom=0.065, top=0.93, wspace=0.04, hspace=0.13)
        leads = [1, 3, 5, 7, 9, 11]
        for row, branch in enumerate(("normal", "delayed")):
            responses, starts, lead_delta, baseline_nino, lat, lon = branch_data[branch]
            for rank in range(3):
                inner = outer[row, rank].subgridspec(2, 6, wspace=0.025, hspace=0.06)
                for var_idx, var_name in enumerate(("TOS", "ZOS")):
                    for col, lead in enumerate(leads):
                        ax = fig.add_subplot(inner[var_idx, col])
                        field = responses[rank, lead, var_idx]
                        vmax = tos_vmax if var_idx == 0 else zos_vmax
                        ax.imshow(field, origin="lower", extent=[float(lon.min()), float(lon.max()), float(lat.min()), float(lat.max())], cmap="RdBu_r", norm=TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax), aspect="auto", interpolation="bilinear")
                        ax.set_xlim(100, 300)
                        ax.set_ylim(-35, 35)
                        ax.set_xticks([])
                        ax.set_yticks([])
                        if col == 0:
                            ax.set_ylabel(var_name, fontsize=7, labelpad=2)
                        if var_idx == 0:
                            ax.set_title(f"L{lead + 1}", fontsize=7, pad=2)
                early = float(np.mean(np.abs(responses[rank, :3, 0])))
                title = f"{branch} rank-{rank + 1} | start={int(starts[rank])} | lead12 ΔNiño={float(lead_delta[rank]):+.2f} | early={early:.2f}"
                fig.text(0.025 + rank * 0.302, 0.952 - row * 0.475, title, fontsize=8, ha="left", va="bottom")
        fig.suptitle(f"Pacific CNOP candidate response evolution — {source} {year} (TOS/ZOS response; 3% constraint)", fontsize=13, y=0.985)
        cax1 = fig.add_axes([0.945, 0.54, 0.012, 0.32])
        cax2 = fig.add_axes([0.945, 0.13, 0.012, 0.32])
        sm1 = plt.cm.ScalarMappable(norm=TwoSlopeNorm(vmin=-tos_vmax, vcenter=0, vmax=tos_vmax), cmap="RdBu_r")
        sm2 = plt.cm.ScalarMappable(norm=TwoSlopeNorm(vmin=-zos_vmax, vcenter=0, vmax=zos_vmax), cmap="RdBu_r")
        fig.colorbar(sm1, cax=cax1, label=f"TOS response (°C), ±{tos_vmax:.2f}")
        fig.colorbar(sm2, cax=cax2, label=f"ZOS response, ±{zos_vmax:.2f}")
        output = args.output_dir / f"pacific_delayed_candidate_panel_{key}.png"
        fig.savefig(output, dpi=220)
        plt.close(fig)
        index.append({"source": source, "target_year": year, "figure": output.name, "tos_vmax": tos_vmax, "zos_vmax": zos_vmax})
        print(output, flush=True)
    (args.output_dir / "candidate_panels_index.json").write_text(json.dumps(index, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
