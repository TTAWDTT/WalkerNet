"""Replay one saved delayed CNOP candidate and retain all four output fields."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.model import WalkerNet
from src.utils import load_config


def rollout(model: WalkerNet, window: torch.Tensor, months: torch.Tensor, horizon: int, trained_steps: int) -> torch.Tensor:
    frames: list[torch.Tensor] = []
    for step in range(horizon):
        month = months[step : step + 1].to(device=window.device)
        rollout_step = torch.tensor([min(step, trained_steps - 1)], dtype=torch.long, device=window.device)
        prediction = model(window, month, rollout_step=rollout_step)
        frames.append(prediction[:, 0])
        window = torch.cat([window[:, 1:], prediction], dim=1)
    return torch.stack(frames, dim=1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--portable", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = load_config(args.portable / "server_gpu006_historical_mixed5.yaml")
    model = WalkerNet(cfg).to(device)
    checkpoint = torch.load(args.portable / "historical_mixed5_best_skill.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    payload = torch.load(args.input, map_location=device, weights_only=False)
    x0 = payload["x0_norm"].to(device=device, dtype=torch.float32)
    months = payload["target_months"].to(device=device, dtype=torch.long)
    artifact = np.load(args.artifact)
    delta = torch.as_tensor(artifact["top_delta_norm"][0], device=device, dtype=x0.dtype).unsqueeze(0)
    perturbed_x0 = x0.clone()
    perturbed_x0[:, -1, :2] += delta
    stats = torch.load(args.portable / "mixed5_norm_stats_train.pt", map_location=device, weights_only=False)["stats"]
    std = stats["std"].to(device=device, dtype=torch.float32).view(1, 1, 4, 1, 1)
    trained_steps = int(cfg.get("training", {}).get("rollout_steps", 18))
    with torch.inference_mode():
        baseline = rollout(model, x0, months, 12, trained_steps)
        perturbed = rollout(model, perturbed_x0, months, 12, trained_steps)
    response = ((perturbed - baseline) * std).squeeze(0).cpu().numpy().astype(np.float32)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, response=response, perturbation=artifact["top_delta_phys"][0].astype(np.float32), lat=artifact["lat"], lon=artifact["lon"], labels=np.array([f"L{i}" for i in range(1, 13)]))
    print(f"saved {args.output} response={response.shape} device={device}")


if __name__ == "__main__":
    main()
