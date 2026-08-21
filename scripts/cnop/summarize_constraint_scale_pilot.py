"""Summarize a pre-registered CNOP constraint-scale pilot.

The runner writes one directory per (constraint scale, source, target year),
containing matched gradient, gradient-seeded CNOP, and random-control outputs.
This script makes the paired comparison explicit and refuses to silently omit a
missing method.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from statistics import mean, median


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize matched CNOP constraint-scale pilot outputs.")
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def read_single_row(path: Path) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise ValueError(f"Expected exactly one row in {path}, found {len(rows)}")
    return rows[0]


def read_random_objectives(path: Path) -> list[float]:
    with path.open(newline="", encoding="utf-8") as handle:
        values = [float(row["objective"]) for row in csv.DictReader(handle)]
    if not values:
        raise ValueError(f"No random controls in {path}")
    return values


def percentile(values: list[float], q: float) -> float:
    values = sorted(values)
    index = (len(values) - 1) * q
    lower = int(index)
    upper = min(lower + 1, len(values) - 1)
    fraction = index - lower
    return values[lower] * (1.0 - fraction) + values[upper] * fraction


def scale_from_directory(path: Path) -> float:
    value = path.name.removeprefix("scale_").replace("p", ".")
    return float(value)


def main() -> None:
    args = parse_args()
    experiment_dir = args.experiment_dir
    output_dir = args.output_dir or experiment_dir / "summary"
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, float | str]] = []
    for scale_dir in sorted(experiment_dir.glob("scale_*"), key=scale_from_directory):
        scale = scale_from_directory(scale_dir)
        for case_dir in sorted(path for path in scale_dir.iterdir() if path.is_dir()):
            cnop = read_single_row(case_dir / "cnop" / "cnop_summary.csv")
            gradient = read_single_row(case_dir / "gradient" / "gradient_summary.csv")
            random = read_random_objectives(case_dir / "random_controls.csv")
            cnop_objective = float(cnop["best_objective"])
            gradient_objective = float(gradient["objective"])
            rows.append(
                {
                    "constraint_scale": scale,
                    "case_id": case_dir.name,
                    "source": cnop["source"],
                    "target_year": cnop["target_year"],
                    "cnop_objective": cnop_objective,
                    "gradient_objective": gradient_objective,
                    "cnop_minus_gradient": cnop_objective - gradient_objective,
                    "cnop_constraint_ratio": float(cnop["constraint_ratio"]),
                    "gradient_constraint_ratio": float(gradient["constraint_ratio"]),
                    "random_mean_objective": mean(random),
                    "random_p95_objective": percentile(random, 0.95),
                    "cnop_random_percentile": 100.0
                    * sum(value <= cnop_objective for value in random)
                    / len(random),
                }
            )

    if not rows:
        raise ValueError(f"No scale_* case directories found in {experiment_dir}")

    fields = list(rows[0])
    with (output_dir / "constraint_scale_pilot_cases.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    grouped: dict[float, list[dict[str, float | str]]] = {}
    for row in rows:
        grouped.setdefault(float(row["constraint_scale"]), []).append(row)
    aggregate_rows: list[dict[str, float | int]] = []
    for scale, group in sorted(grouped.items()):
        deltas = [float(row["cnop_minus_gradient"]) for row in group]
        aggregate_rows.append(
            {
                "constraint_scale": scale,
                "n_cases": len(group),
                "mean_cnop_minus_gradient": mean(deltas),
                "median_cnop_minus_gradient": median(deltas),
                "cnop_wins": sum(delta > 0.0 for delta in deltas),
                "cnop_win_rate": mean(delta > 0.0 for delta in deltas),
                "mean_cnop_random_percentile": mean(float(row["cnop_random_percentile"]) for row in group),
                "max_constraint_ratio": max(
                    max(float(row["cnop_constraint_ratio"]), float(row["gradient_constraint_ratio"])) for row in group
                ),
            }
        )
    with (output_dir / "constraint_scale_pilot_by_scale.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(aggregate_rows[0]))
        writer.writeheader()
        writer.writerows(aggregate_rows)
    print(f"[summary] cases={len(rows)} scales={len(aggregate_rows)} wrote {output_dir}")


if __name__ == "__main__":
    main()
