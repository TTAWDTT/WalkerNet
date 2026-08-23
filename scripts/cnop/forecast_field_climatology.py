"""Forecast-field climatology utilities for CNOP reporting plots.

Observed fields and model rollouts live in different climatological worlds.  An
observed anomaly is referenced to the source training-period monthly mean,
whereas a forecast anomaly must be referenced to the model's own mean at the
same source, forecast lead, and calendar month.  This module centralises that
distinction for all predicted variables, so figure scripts cannot accidentally
subtract an observed climatology from a model forecast.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from src.dataset import WalkerDataset


def monthly_observed_field_climatology(
    dataset: WalkerDataset,
    source_idx: int,
    target_months: np.ndarray,
) -> np.ndarray:
    """Return source-wise training-period field climatology for each month."""

    payload = dataset.source_payloads[source_idx]
    train_start, train_end = dataset.data_config["train_years"]
    years = np.asarray(payload["years"])
    months = np.asarray(payload["months"])
    train_mask = (years >= int(train_start)) & (years <= int(train_end))
    climatology: list[np.ndarray] = []
    for month in np.asarray(target_months, dtype=np.int64):
        mask = train_mask & (months == int(month))
        if not np.any(mask):
            source = dataset.source_names[source_idx]
            raise ValueError(f"No observed climatology samples for source={source}, month={month}")
        climatology.append(np.nanmean(np.asarray(payload["data"])[mask], axis=0, dtype=np.float64))
    return np.asarray(climatology, dtype=np.float32)


def valid_climatology_starts(
    dataset: WalkerDataset,
    source_idx: int,
    horizon: int,
    mode: str,
    split: str,
) -> list[int]:
    """Return complete rollout starts belonging to the requested year range."""

    payload = dataset.source_payloads[source_idx]
    years = np.asarray(payload["years"])
    starts = range(dataset.L, len(years) - horizon + 1)
    if mode == "all":
        return list(starts)
    year_key = "train_years" if mode == "train" else f"{split}_years"
    start_year, end_year = dataset.data_config[year_key]
    return [
        int(target_t)
        for target_t in starts
        if np.all((years[target_t : target_t + horizon] >= int(start_year)) & (years[target_t : target_t + horizon] <= int(end_year)))
    ]


def _make_batch_input(
    dataset: WalkerDataset,
    source_idx: int,
    target_indices: list[int],
    device: torch.device,
) -> torch.Tensor:
    payload = dataset.source_payloads[source_idx]
    raw = np.asarray(
        [payload["data"][target_t - dataset.L : target_t] for target_t in target_indices],
        dtype=np.float32,
    )
    x = torch.from_numpy(raw).to(device=device, dtype=torch.float32)
    x = dataset._normalize_tensor(x)  # noqa: SLF001 - mirrors WalkerDataset preprocessing
    return torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


def _rollout_field_batch(
    model: torch.nn.Module,
    dataset: WalkerDataset,
    source_idx: int,
    target_indices: list[int],
    x_norm: torch.Tensor,
    horizon: int,
    trained_rollout_steps: int,
) -> torch.Tensor:
    """Return denormalized fields with shape ``(batch, lead, variable, y, x)``."""

    months = np.asarray(dataset.source_payloads[source_idx]["months"])
    window = x_norm
    predictions: list[torch.Tensor] = []
    for step in range(horizon):
        target_month = torch.as_tensor(
            [int(months[target_t + step]) for target_t in target_indices],
            dtype=torch.long,
            device=x_norm.device,
        )
        rollout_step = torch.full(
            (x_norm.shape[0],),
            min(step, trained_rollout_steps - 1),
            dtype=torch.long,
            device=x_norm.device,
        )
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=x_norm.device.type == "cuda"):
            prediction_norm = model(window, target_month, rollout_step=rollout_step)
            predictions.append(dataset.denormalize(prediction_norm)[:, 0].detach().cpu())
        window = torch.cat([window[:, 1:], prediction_norm], dim=1)
    return torch.stack(predictions, dim=1)


def compute_forecast_field_climatology(
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
    """Estimate model climatology indexed by source, lead, month and variable."""

    climatology_by_source: dict[int, np.ndarray] = {}
    for source_idx in sorted(set(source_indices)):
        payload = dataset.source_payloads[source_idx]
        starts = valid_climatology_starts(dataset, source_idx, horizon, mode, split)
        source = dataset.source_names[source_idx]
        if not starts:
            raise ValueError(f"No forecast climatology starts for source={source}, mode={mode}")
        variable_count, height, width = np.asarray(payload["data"]).shape[1:]
        sums = np.zeros((horizon, 13, variable_count, height, width), dtype=np.float32)
        counts = np.zeros((horizon, 13), dtype=np.int32)
        months = np.asarray(payload["months"])
        print(f"[forecast-field-clim] {source}: {len(starts)} starts, mode={mode}", flush=True)
        for offset in range(0, len(starts), max(1, batch_size)):
            batch_starts = starts[offset : offset + max(1, batch_size)]
            x_batch = _make_batch_input(dataset, source_idx, batch_starts, device)
            predictions = _rollout_field_batch(
                model, dataset, source_idx, batch_starts, x_batch, horizon, trained_rollout_steps
            ).numpy()
            for batch_idx, target_t in enumerate(batch_starts):
                for lead_idx in range(horizon):
                    month = int(months[target_t + lead_idx])
                    sums[lead_idx, month] += predictions[batch_idx, lead_idx]
                    counts[lead_idx, month] += 1
        climatology = np.full_like(sums, np.nan)
        for lead_idx in range(horizon):
            for month in range(1, 13):
                if counts[lead_idx, month]:
                    climatology[lead_idx, month] = sums[lead_idx, month] / float(counts[lead_idx, month])
        climatology_by_source[source_idx] = climatology
    return climatology_by_source


def load_or_compute_forecast_field_climatology(
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
    """Load a compatible cache or compute the model-world field climatology."""

    source_indices = sorted(set(int(source_idx) for source_idx in source_indices))
    if cache_path.exists():
        with np.load(cache_path) as data:
            cached_sources = [int(item) for item in np.asarray(data["source_indices"]).tolist()]
            climatology = np.asarray(data["climatology"], dtype=np.float32)
            if (
                int(data["horizon"]) == horizon
                and str(data["mode"]) == mode
                and str(data["split"]) == split
                and climatology.ndim == 6
                and all(source_idx in cached_sources for source_idx in source_indices)
            ):
                print(f"[forecast-field-clim] using cache {cache_path}", flush=True)
                return {
                    source_idx: climatology[cached_sources.index(source_idx)] for source_idx in source_indices
                }

    climatology = compute_forecast_field_climatology(
        model,
        dataset,
        source_indices,
        horizon,
        trained_rollout_steps,
        device,
        mode,
        split,
        batch_size,
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        source_indices=np.asarray(source_indices, dtype=np.int64),
        climatology=np.stack([climatology[source_idx] for source_idx in source_indices]),
        horizon=np.asarray(horizon, dtype=np.int64),
        mode=np.asarray(mode),
        split=np.asarray(split),
    )
    print(f"[forecast-field-clim] wrote cache {cache_path}", flush=True)
    return climatology
