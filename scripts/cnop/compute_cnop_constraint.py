"""Compute an ENSO-event based CNOP perturbation constraint.

本脚本只负责计算约束半径，不做 CNOP 优化：

1. 按每个 source 自己的训练期月气候态计算 Niño3.4 anomaly；
2. 找出发生 ENSO 事件的年份；
3. 取事件年前一年 12 月的 TOS/ZOS；
4. 将 TOS/ZOS 分别转成无量纲且尺度相等的量；
5. 对每个样本计算联合二范数，并把均值作为 constraint。
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.cnop.basin_domains import BASIN_DOMAINS, numpy_basin_region  # noqa: E402
from scripts.cnop.compute_tos_zos_cnop import compute_nino34_numpy, parse_bounds, three_month_mean_np  # noqa: E402
from src.dataset import WalkerDataset  # noqa: E402
from src.utils import load_config  # noqa: E402


TOS_IDX = 0
ZOS_IDX = 1


@dataclass(frozen=True)
class EnsoEventSample:
    """一个用于计算 constraint 的 ENSO 事件样本。"""

    source_idx: int
    source_name: str
    event_year: int
    previous_december_index: int
    event_max_3m_abs_nino34: float
    event_peak_3m_nino34: float
    event_type: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute ENSO-event based CNOP constraint.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--split", type=str, default="train", choices=("train", "val", "test"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/cnop_constraint"))
    parser.add_argument("--event-threshold", type=float, default=0.5)
    parser.add_argument(
        "--event-year-range",
        type=str,
        default="train",
        choices=("train", "val", "test", "all"),
        help="在哪个年份范围内寻找 ENSO 事件。默认 train，避免用测试集信息定约束。",
    )
    parser.add_argument(
        "--normalization",
        type=str,
        default="december_anomaly_train_std_equal_rms",
        choices=("dataset_zscore_equal_rms", "december_anomaly_train_std_equal_rms"),
        help="TOS/ZOS 无量纲化方法。默认先做 12 月距平，再除训练期 12 月标准差，最后两变量 RMS 拉齐。",
    )
    parser.add_argument("--domain", choices=BASIN_DOMAINS, default="global")
    parser.add_argument(
        "--basin-lat-bounds",
        type=str,
        default="-60,60",
        help="三个海域 constraint 共同使用的纬度范围。",
    )
    return parser.parse_args()


def source_monthly_nino34_climatology(dataset: WalkerDataset) -> np.ndarray:
    """计算每个 source 的训练期 Niño3.4 月气候态。"""

    train_start, train_end = dataset.data_config["train_years"]
    climatology = np.full((len(dataset.source_payloads), 13), np.nan, dtype=np.float32)
    for source_idx, payload in enumerate(dataset.source_payloads):
        years = np.asarray(payload["years"])
        months = np.asarray(payload["months"])
        tos = np.asarray(payload["data"][:, TOS_IDX], dtype=np.float32)
        nino34 = compute_nino34_numpy(tos, np.asarray(payload["lat"]), np.asarray(payload["lon"]))
        train_mask = (years >= int(train_start)) & (years <= int(train_end))
        for month in range(1, 13):
            month_mask = train_mask & (months == month)
            climatology[source_idx, month] = float(np.nanmean(nino34[month_mask]))
    return climatology


def selected_year_bounds(dataset: WalkerDataset, mode: str) -> tuple[int, int] | None:
    """返回需要搜索事件年的年份范围；all 表示不限制。"""

    if mode == "all":
        return None
    start, end = dataset.data_config[f"{mode}_years"]
    return int(start), int(end)


def find_previous_december(payload: dict[str, Any], event_year: int) -> int | None:
    years = np.asarray(payload["years"])
    months = np.asarray(payload["months"])
    matches = np.where((years == event_year - 1) & (months == 12))[0]
    if matches.size == 0:
        return None
    return int(matches[-1])


def select_enso_event_samples(
    dataset: WalkerDataset,
    nino34_climatology: np.ndarray,
    threshold: float,
    year_range_mode: str,
) -> list[EnsoEventSample]:
    """找出 ENSO 事件年，并定位事件年前一年 12 月。"""

    bounds = selected_year_bounds(dataset, year_range_mode)
    samples: list[EnsoEventSample] = []
    for source_idx, payload in enumerate(dataset.source_payloads):
        years = np.asarray(payload["years"])
        months = np.asarray(payload["months"])
        tos = np.asarray(payload["data"][:, TOS_IDX], dtype=np.float32)
        nino34 = compute_nino34_numpy(tos, np.asarray(payload["lat"]), np.asarray(payload["lon"]))
        anomaly = nino34 - nino34_climatology[source_idx, months]
        source_name = dataset.source_names[source_idx]

        for event_year in sorted(set(int(year) for year in years)):
            if bounds is not None and not (bounds[0] <= event_year <= bounds[1]):
                continue
            year_indices = np.where((years == event_year) & (months >= 1) & (months <= 12))[0]
            if year_indices.size != 12:
                continue
            if not np.array_equal(months[year_indices], np.arange(1, 13)):
                continue
            rolling = three_month_mean_np(anomaly[year_indices])
            if rolling.size == 0 or not np.isfinite(rolling).any():
                continue
            peak = float(rolling[np.nanargmax(np.abs(rolling))])
            max_abs = abs(peak)
            if max_abs < threshold:
                continue
            previous_december_index = find_previous_december(payload, event_year)
            if previous_december_index is None:
                continue
            samples.append(
                EnsoEventSample(
                    source_idx=source_idx,
                    source_name=source_name,
                    event_year=event_year,
                    previous_december_index=previous_december_index,
                    event_max_3m_abs_nino34=max_abs,
                    event_peak_3m_nino34=peak,
                    event_type="ElNino" if peak > 0 else "LaNina",
                )
            )
    return samples


def december_anomaly_stats(
    dataset: WalkerDataset,
    source_indices: set[int],
    domain: str,
    basin_lat_bounds: tuple[float, float],
) -> dict[int, dict[str, np.ndarray]]:
    """为会被实际用到的 source 计算训练期 12 月 TOS/ZOS 统计量。"""

    train_start, train_end = dataset.data_config["train_years"]
    stats: dict[int, dict[str, np.ndarray]] = {}
    for source_idx, payload in enumerate(dataset.source_payloads):
        if source_idx not in source_indices:
            continue
        years = np.asarray(payload["years"])
        months = np.asarray(payload["months"])
        train_december = (years >= int(train_start)) & (years <= int(train_end)) & (months == 12)
        fields = np.asarray(payload["data"][train_december, :2], dtype=np.float32)
        if fields.size == 0:
            source = dataset.source_names[source_idx]
            raise ValueError(f"No training December fields for source={source}")
        mean_field = np.nanmean(fields, axis=0)
        region = numpy_basin_region(payload["lat"], payload["lon"], domain, basin_lat_bounds)
        source_valid = payload["valid_mask"][:2].cpu().numpy().astype(bool) & region[None]
        anomalies = np.where(source_valid[None], fields - mean_field[None], np.nan)
        std_scalar = np.nanstd(anomalies, axis=(0, 2, 3)).astype(np.float32)
        std_scalar = np.where(std_scalar > 1.0e-12, std_scalar, 1.0).astype(np.float32)
        stats[source_idx] = {
            "mean_field": mean_field,
            "std_scalar": std_scalar,
            "valid": source_valid,
        }
    return stats


def event_fields_to_dimensionless(
    dataset: WalkerDataset,
    samples: list[EnsoEventSample],
    normalization: str,
    domain: str,
    basin_lat_bounds: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[int, np.ndarray]]:
    """返回拉齐尺度后的事件场，以及 TOS/ZOS 的尺度因子。"""

    raw_fields: list[np.ndarray] = []
    source_stds: dict[int, np.ndarray] = {}
    if normalization == "dataset_zscore_equal_rms":
        mean = dataset.norm_stats["mean"].cpu().numpy()[:2].astype(np.float32)
        std = dataset.norm_stats["std"].cpu().numpy()[:2].astype(np.float32)
        for sample in samples:
            payload = dataset.source_payloads[sample.source_idx]
            field = np.asarray(
                payload["data"][sample.previous_december_index, :2],
                dtype=np.float32,
            )
            region = numpy_basin_region(payload["lat"], payload["lon"], domain, basin_lat_bounds)
            source_valid = payload["valid_mask"][:2].cpu().numpy().astype(bool) & region[None]
            raw_fields.append(np.where(source_valid, (field - mean[:, None, None]) / std[:, None, None], np.nan))
    else:
        stats = december_anomaly_stats(
            dataset,
            set(range(len(dataset.source_payloads))),
            domain,
            basin_lat_bounds,
        )
        source_stds = {source_idx: item["std_scalar"] for source_idx, item in stats.items()}
        for sample in samples:
            field = np.asarray(
                dataset.source_payloads[sample.source_idx]["data"][sample.previous_december_index, :2],
                dtype=np.float32,
            )
            source_stats = stats[sample.source_idx]
            raw_fields.append(
                np.where(
                    source_stats["valid"],
                    (field - source_stats["mean_field"]) / source_stats["std_scalar"][:, None, None],
                    np.nan,
                )
            )

    fields = np.stack(raw_fields, axis=0).astype(np.float32)
    valid = np.isfinite(fields)
    rms = np.sqrt(np.nanmean(np.where(valid, fields, np.nan) ** 2, axis=(0, 2, 3))).astype(np.float32)
    rms = np.where(rms > 1.0e-12, rms, 1.0).astype(np.float32)
    equalized = fields / rms[None, :, None, None]
    return equalized, rms, valid, source_stds


def compute_constraint(
    fields: np.ndarray,
    valid: np.ndarray,
) -> tuple[list[dict[str, float]], dict[str, float]]:
    """计算每个事件样本的联合二范数，并汇总 constraint。"""

    rows: list[dict[str, float]] = []
    for idx in range(fields.shape[0]):
        tos = np.where(valid[idx, 0], fields[idx, 0], 0.0)
        zos = np.where(valid[idx, 1], fields[idx, 1], 0.0)
        tos_l2 = float(np.sqrt(np.sum(tos * tos)))
        zos_l2 = float(np.sqrt(np.sum(zos * zos)))
        combined_l2 = float(np.sqrt(tos_l2 * tos_l2 + zos_l2 * zos_l2))
        valid_count = int(np.count_nonzero(valid[idx, :2]))
        rows.append(
            {
                "tos_l2": tos_l2,
                "zos_l2": zos_l2,
                "combined_l2": combined_l2,
                "combined_rms": combined_l2 / float(np.sqrt(max(valid_count, 1))),
                "valid_count": float(valid_count),
            }
        )

    summary = {
        "constraint_l2": float(np.mean([row["combined_l2"] for row in rows])),
        "constraint_l2_std": float(np.std([row["combined_l2"] for row in rows])),
        "constraint_rms": float(np.mean([row["combined_rms"] for row in rows])),
        "num_event_samples": float(len(rows)),
    }
    return rows, summary


def write_outputs(
    output_dir: Path,
    args: argparse.Namespace,
    samples: list[EnsoEventSample],
    scale_rms: np.ndarray,
    source_stds: dict[int, np.ndarray],
    source_names: list[str],
    event_rows: list[dict[str, float]],
    summary: dict[str, float],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "cnop_constraint_events.csv"
    fieldnames = [
        "source",
        "event_year",
        "previous_december_index",
        "event_type",
        "event_peak_3m_nino34",
        "event_max_3m_abs_nino34",
        "tos_l2",
        "zos_l2",
        "combined_l2",
        "combined_rms",
        "valid_count",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for sample, row in zip(samples, event_rows):
            writer.writerow(
                {
                    "source": sample.source_name,
                    "event_year": sample.event_year,
                    "previous_december_index": sample.previous_december_index,
                    "event_type": sample.event_type,
                    "event_peak_3m_nino34": sample.event_peak_3m_nino34,
                    "event_max_3m_abs_nino34": sample.event_max_3m_abs_nino34,
                    **row,
                }
            )

    payload = {
        "constraint_l2": summary["constraint_l2"],
        "constraint_l2_std": summary["constraint_l2_std"],
        "constraint_rms": summary["constraint_rms"],
        "num_event_samples": int(summary["num_event_samples"]),
        "event_threshold": args.event_threshold,
        "event_year_range": args.event_year_range,
        "normalization": args.normalization,
        "domain": args.domain,
        "basin_lat_bounds": list(parse_bounds(args.basin_lat_bounds)),
        "equalization_rms_before_rescale": {
            "tos": float(scale_rms[0]),
            "zos": float(scale_rms[1]),
        },
        "source_december_std": {
            dataset_name: [float(value) for value in source_stds[source_idx]]
            for source_idx, dataset_name in enumerate(source_names)
            if source_idx in source_stds
        },
        "definition": (
            "constraint_l2 is the mean joint Euclidean norm of previous-December "
            "TOS/ZOS fields for ENSO-event years after variable-wise "
            "dimensionless normalization and equal-RMS rescaling."
        ),
        "events_csv": str(csv_path),
    }
    with (output_dir / "cnop_constraint_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    dataset = WalkerDataset(config["data"]["path"], config, split=args.split)

    nino34_climatology = source_monthly_nino34_climatology(dataset)
    samples = select_enso_event_samples(
        dataset,
        nino34_climatology,
        threshold=float(args.event_threshold),
        year_range_mode=args.event_year_range,
    )
    if not samples:
        raise RuntimeError("No ENSO-event samples found; check event threshold or year range.")

    basin_lat_bounds = parse_bounds(args.basin_lat_bounds)
    fields, scale_rms, valid, source_stds = event_fields_to_dimensionless(
        dataset,
        samples,
        args.normalization,
        args.domain,
        basin_lat_bounds,
    )
    event_rows, summary = compute_constraint(fields, valid)
    write_outputs(
        args.output_dir,
        args,
        samples,
        scale_rms,
        source_stds,
        dataset.source_names,
        event_rows,
        summary,
    )

    print(f"[constraint] domain: {args.domain} lat={basin_lat_bounds}")
    print(f"[constraint] event samples: {len(samples)}")
    print(f"[constraint] constraint_l2: {summary['constraint_l2']:.6f}")
    print(f"[constraint] constraint_rms: {summary['constraint_rms']:.6f}")
    print(f"[constraint] output: {args.output_dir}")


if __name__ == "__main__":
    main()
