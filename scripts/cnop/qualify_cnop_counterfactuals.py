"""Qualify CNOP outputs for counterfactual-ENSO plotting and aggregation.

The input case-selection CSV is produced by ``sample_cnop_cases_by_baseline``.
Each CNOP summary is joined to that frozen case set and retained only when the
observed truth and unperturbed forecast are neutral while the CNOP rollout
crosses the prescribed positive ENSO threshold.  The resulting table is the
only allowed source for workshop main-result figures.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


REQUIRED_SELECTION_COLUMNS = {
    "source",
    "target_year",
    "observed_max_3m_abs",
    "baseline_truth_nino_rmse",
    "baseline_lead12_abs_error",
    "baseline_eligible",
}
REQUIRED_CNOP_COLUMNS = {
    "source",
    "target_year",
    "observed_max_3m_abs",
    "baseline_max_3m",
    "cnop_max_3m",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter CNOP outputs to valid neutral-to-El Nino counterfactuals.")
    parser.add_argument("--selected-cases", type=Path, required=True)
    parser.add_argument("--cnop-summary", type=Path, required=True)
    parser.add_argument("--domain", required=True, choices=("pacific", "atlantic_indian", "global"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--neutral-threshold", type=float, default=0.5)
    parser.add_argument("--baseline-event-threshold", type=float, default=0.5)
    parser.add_argument("--event-threshold", type=float, default=0.5)
    return parser.parse_args()


def require_columns(frame: pd.DataFrame, required: set[str], name: str) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {', '.join(missing)}")


def main() -> None:
    args = parse_args()
    selected = pd.read_csv(args.selected_cases)
    summary = pd.read_csv(args.cnop_summary)
    require_columns(selected, REQUIRED_SELECTION_COLUMNS, str(args.selected_cases))
    require_columns(summary, REQUIRED_CNOP_COLUMNS, str(args.cnop_summary))

    selected = selected[
        selected["baseline_eligible"].astype(str).str.strip().str.lower().isin({"true", "1", "yes"})
    ].copy()
    merged = summary.merge(
        selected[
            [
                "source",
                "target_year",
                "baseline_truth_nino_rmse",
                "baseline_lead12_abs_error",
            ]
        ],
        on=["source", "target_year"],
        how="inner",
        validate="one_to_one",
    )
    merged.insert(0, "domain", args.domain)
    merged["truth_neutral"] = merged["observed_max_3m_abs"] <= args.neutral_threshold
    merged["baseline_neutral"] = merged["baseline_max_3m"] < args.baseline_event_threshold
    merged["cnop_induced_el_nino"] = merged["cnop_max_3m"] >= args.event_threshold
    merged["counterfactual_qualified"] = (
        merged["truth_neutral"] & merged["baseline_neutral"] & merged["cnop_induced_el_nino"]
    )
    merged = merged.sort_values(["counterfactual_qualified", "cnop_max_3m"], ascending=[False, False])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.output, index=False)

    qualified = int(merged["counterfactual_qualified"].sum())
    print(
        f"[qualify] domain={args.domain} selected={len(selected)} joined={len(merged)} qualified={qualified} "
        f"output={args.output}",
        flush=True,
    )
    if qualified == 0:
        raise RuntimeError("No counterfactual-qualified CNOP cases. Do not make a main-result figure from this run.")


if __name__ == "__main__":
    main()
