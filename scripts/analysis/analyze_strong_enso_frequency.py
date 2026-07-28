"""统计历史与四个 SSP 情景中的强 ENSO 年比例。

为了避免把外强迫导致的长期增暖误判成 El Niño，本脚本对每个
``情景 × 物理模式`` 独立执行：

1. 计算面积加权 Niño3.4 区域平均海温；
2. 去除该时段自身的逐月气候态；
3. 去除 Niño3.4 异常序列的线性趋势；
4. 计算居中的三个月滑动平均；
5. 若某日历年内的最大绝对三个月指数不低于阈值，则记为强 ENSO 年。

默认比较等长时段：历史 1929-2014 与未来 2015-2100，均为 86 年。
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr


SOURCES = ("CESM2", "EC-Earth3", "GFDL-ESM4", "IPSL-CM6A-LR", "MPI-ESM1-2-HR")
SCENARIOS = ("historical", "ssp126", "ssp245", "ssp370", "ssp585")


@dataclass(frozen=True)
class AnnualPeak:
    scenario: str
    source: str
    year: int
    peak_3m_nino34: float
    max_abs_3m_nino34: float
    event_type: str
    is_strong: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze strong ENSO frequency across CMIP6 scenarios")
    parser.add_argument("--historical-cesm-dir", type=Path, required=True)
    parser.add_argument("--historical-cmip-dir", type=Path, required=True)
    parser.add_argument("--future-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=1.5)
    parser.add_argument("--historical-years", type=int, nargs=2, default=(1929, 2014))
    parser.add_argument("--future-years", type=int, nargs=2, default=(2015, 2100))
    parser.add_argument("--no-detrend", action="store_true", help="仅去逐月气候态，不去线性趋势")
    return parser.parse_args()


def _tos_path(args: argparse.Namespace, scenario: str, source: str) -> Path:
    if scenario == "historical":
        root = args.historical_cesm_dir if source == "CESM2" else args.historical_cmip_dir / source
    else:
        root = args.future_root / scenario / source
    path = root / "tos_1x1.nc"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _nino34_series(path: Path, start_year: int, end_year: int) -> tuple[np.ndarray, ...]:
    """读取规则网格 TOS，并返回指定时段的年月和 Niño3.4 区域平均。"""

    with xr.open_dataset(path) as dataset:
        tos = dataset["tos"].squeeze(drop=True)
        years = np.asarray(tos["time"].dt.year.values, dtype=np.int32)
        months = np.asarray(tos["time"].dt.month.values, dtype=np.int8)
        period = (years >= start_year) & (years <= end_year)
        years = years[period]
        months = months[period]

        lat = np.asarray(tos["lat"].values, dtype=np.float64)
        lon = np.mod(np.asarray(tos["lon"].values, dtype=np.float64), 360.0)
        lat_mask = (lat >= -5.0) & (lat <= 5.0)
        lon_mask = (lon >= 190.0) & (lon <= 240.0)
        region = np.asarray(tos.isel(time=np.flatnonzero(period), lat=lat_mask, lon=lon_mask), dtype=np.float64)

    weights = np.cos(np.deg2rad(lat[lat_mask]))[:, None]
    valid = np.isfinite(region)
    numerator = np.sum(np.where(valid, region * weights[None], 0.0), axis=(1, 2))
    denominator = np.sum(np.where(valid, weights[None], 0.0), axis=(1, 2))
    raw = numerator / denominator
    if float(np.nanmean(raw)) > 100.0:
        raw = raw - 273.15

    expected_months = (end_year - start_year + 1) * 12
    if len(raw) != expected_months:
        raise ValueError(f"{path}: expected {expected_months} months, got {len(raw)}")
    return years, months, raw.astype(np.float64)


def _nino34_anomaly(months: np.ndarray, raw: np.ndarray, detrend: bool) -> np.ndarray:
    climatology = np.array([np.nanmean(raw[months == month]) for month in range(1, 13)])
    anomaly = raw - climatology[months - 1]
    if not detrend:
        return anomaly
    finite = np.isfinite(anomaly)
    time = np.arange(len(anomaly), dtype=np.float64)
    slope, intercept = np.polyfit(time[finite], anomaly[finite], deg=1)
    return anomaly - (slope * time + intercept)


def _annual_peaks(
    scenario: str,
    source: str,
    years: np.ndarray,
    anomaly: np.ndarray,
    threshold: float,
) -> list[AnnualPeak]:
    """按三个月滑动平均中心月所属年份，计算每年的最强冷暖峰值。"""

    rolling = np.convolve(anomaly, np.ones(3, dtype=np.float64) / 3.0, mode="valid")
    center_years = years[1:-1]
    rows: list[AnnualPeak] = []
    for year in range(int(years.min()), int(years.max()) + 1):
        values = rolling[center_years == year]
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue
        peak = float(values[np.argmax(np.abs(values))])
        rows.append(
            AnnualPeak(
                scenario=scenario,
                source=source,
                year=year,
                peak_3m_nino34=peak,
                max_abs_3m_nino34=abs(peak),
                event_type="ElNino" if peak >= 0.0 else "LaNina",
                is_strong=abs(peak) >= threshold,
            )
        )
    return rows


def _summarize(rows: list[AnnualPeak], raw_means: dict[tuple[str, str], float]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for scenario in SCENARIOS:
        selected = [row for row in rows if row.scenario == scenario]
        strong = [row for row in selected if row.is_strong]
        strong_el = [row for row in strong if row.event_type == "ElNino"]
        strong_la = [row for row in strong if row.event_type == "LaNina"]
        source_rates = []
        for source in SOURCES:
            source_rows = [row for row in selected if row.source == source]
            source_rates.append(sum(row.is_strong for row in source_rows) / len(source_rows))
        output.append(
            {
                "scenario": scenario,
                "model_years": len(selected),
                "strong_years": len(strong),
                "strong_year_rate": len(strong) / len(selected),
                "source_rate_std": float(np.std(source_rates, ddof=1)),
                "strong_el_nino_years": len(strong_el),
                "strong_la_nina_years": len(strong_la),
                "mean_abs_peak_strong": float(np.mean([row.max_abs_3m_nino34 for row in strong])),
                "mean_el_nino_peak": _optional_mean([row.peak_3m_nino34 for row in strong_el]),
                "mean_la_nina_peak": _optional_mean([row.peak_3m_nino34 for row in strong_la]),
                "mean_raw_nino34_sst_c": float(
                    np.mean([raw_means[(scenario, source)] for source in SOURCES])
                ),
            }
        )

    historical_rate = float(output[0]["strong_year_rate"])
    for item in output:
        rate = float(item["strong_year_rate"])
        item["rate_change_vs_historical_pp"] = 100.0 * (rate - historical_rate)
        item["relative_rate_change_vs_historical"] = (
            rate / historical_rate - 1.0 if historical_rate > 0.0 else float("nan")
        )
    return output


def _source_summary(rows: list[AnnualPeak], raw_means: dict[tuple[str, str], float]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for scenario in SCENARIOS:
        for source in SOURCES:
            selected = [row for row in rows if row.scenario == scenario and row.source == source]
            strong = [row for row in selected if row.is_strong]
            output.append(
                {
                    "scenario": scenario,
                    "source": source,
                    "years": len(selected),
                    "strong_years": len(strong),
                    "strong_year_rate": len(strong) / len(selected),
                    "mean_abs_peak_strong": _optional_mean(
                        [row.max_abs_3m_nino34 for row in strong]
                    ),
                    "mean_raw_nino34_sst_c": raw_means[(scenario, source)],
                }
            )
    return output


def _optional_mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if args.threshold <= 0.0:
        raise ValueError("threshold must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    annual_rows: list[AnnualPeak] = []
    raw_means: dict[tuple[str, str], float] = {}
    for scenario in SCENARIOS:
        bounds = args.historical_years if scenario == "historical" else args.future_years
        for source in SOURCES:
            path = _tos_path(args, scenario, source)
            years, months, raw = _nino34_series(path, int(bounds[0]), int(bounds[1]))
            anomaly = _nino34_anomaly(months, raw, detrend=not args.no_detrend)
            annual_rows.extend(_annual_peaks(scenario, source, years, anomaly, args.threshold))
            raw_means[(scenario, source)] = float(np.mean(raw))
            print(f"done scenario={scenario} source={source} years={bounds[0]}-{bounds[1]}", flush=True)

    scenario_rows = _summarize(annual_rows, raw_means)
    source_rows = _source_summary(annual_rows, raw_means)
    event_rows = [asdict(row) for row in annual_rows if row.is_strong]
    _write_csv(args.output_dir / "scenario_summary.csv", scenario_rows)
    _write_csv(args.output_dir / "source_summary.csv", source_rows)
    _write_csv(args.output_dir / "strong_event_years.csv", event_rows)

    metadata = {
        "threshold_c": args.threshold,
        "historical_years": list(args.historical_years),
        "future_years": list(args.future_years),
        "sources": list(SOURCES),
        "linear_detrend": not args.no_detrend,
        "definition": (
            "A strong ENSO year has max absolute centered 3-month, monthly-climatology-removed "
            "and linearly detrended Nino3.4 anomaly >= threshold within the calendar year."
        ),
        "scenario_summary": scenario_rows,
    }
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, ensure_ascii=False, indent=2)
    print(json.dumps(scenario_rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
