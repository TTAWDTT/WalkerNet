"""Plot the matched CNOP, zero-state-gradient, and random-control comparison.

This is a separate companion figure for the workshop claim.  It intentionally
does not change ``plot_basin_cnop_experiment.py`` or its CNOP--random figure:
that figure remains a description of CNOP impact, while this one is the
explicit three-method, equal-budget test of finite-amplitude nonlinearity.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


DOMAINS = ("pacific", "atlantic_indian", "global")
DOMAIN_LABELS = {
    "pacific": "Pacific only",
    "atlantic_indian": "Atlantic + Indian",
    "global": "Global ocean",
}
DOMAIN_COLORS = {
    "pacific": "#D55E00",
    "atlantic_indian": "#009E73",
    "global": "#0072B2",
}
CNOP_COLOR = "#111827"
GRADIENT_COLOR = "#7C3AED"


@dataclass(frozen=True)
class DomainComparison:
    domain: str
    source: str
    target_year: int
    cnop_value: float
    gradient_value: float
    random_values: np.ndarray
    cnop_norm: float
    cnop_radius: float
    cnop_ratio: float
    gradient_norm: float
    gradient_radius: float
    gradient_ratio: float
    gradient_projected: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot matched CNOP / gradient / random basin comparisons.")
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--metric",
        choices=("objective", "lead_delta"),
        default="objective",
        help="objective is the workshop late-3m delta; lead_delta is the lead-12 response.",
    )
    parser.add_argument("--dpi", type=int, default=320)
    parser.add_argument("--radius-atol", type=float, default=1.0e-5)
    parser.add_argument("--ratio-tol", type=float, default=1.0e-5)
    return parser.parse_args()


def read_one_csv(path: Path) -> dict[str, str]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise ValueError(f"Expected exactly one row in {path}, found {len(rows)}")
    return rows[0]


def as_bool(value: str | bool | np.bool_) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes"}


def scalar_from_npz(npz: Any, key: str) -> float:
    if key not in npz.files:
        raise KeyError(f"Required key {key!r} is absent from gradient case NPZ")
    return float(np.asarray(npz[key]).reshape(()).item())


def load_domain_comparison(
    experiment_dir: Path,
    domain: str,
    metric: str,
    radius_atol: float,
    ratio_tol: float,
) -> DomainComparison:
    """Read and cross-check all three matched controls for one basin."""

    cnop_row = read_one_csv(experiment_dir / "combined" / domain / "cnop_summary.csv")
    gradient_dir = experiment_dir / "gradient_baseline" / domain
    gradient_row = read_one_csv(gradient_dir / "gradient_summary.csv")
    random_path = experiment_dir / "random_controls" / f"{domain}.csv"
    with random_path.open("r", newline="", encoding="utf-8") as handle:
        random_rows = list(csv.DictReader(handle))
    if not random_rows:
        raise ValueError(f"No random controls in {random_path}")

    source = cnop_row["source"]
    year = int(cnop_row["target_year"])
    if gradient_row["source"] != source or int(gradient_row["target_year"]) != year:
        raise ValueError(
            f"Case mismatch for {domain}: CNOP={source} {year}, "
            f"gradient={gradient_row['source']} {gradient_row['target_year']}"
        )
    if gradient_row.get("objective_mode") != "late_3m_delta":
        raise ValueError(f"Gradient baseline for {domain} is not late_3m_delta")

    npz_path = gradient_dir / f"case_{source}_{year}.npz"
    with np.load(npz_path, allow_pickle=False) as npz:
        # The NPZ is a second, independent artifact: reject accidental CSV/NPZ
        # mixing before plotting a paper comparison.
        npz_value = scalar_from_npz(npz, metric)
        npz_norm = scalar_from_npz(npz, "constraint_norm")
        npz_radius = scalar_from_npz(npz, "constraint_radius")
        npz_ratio = scalar_from_npz(npz, "constraint_ratio")
        npz_projected = as_bool(np.asarray(npz["projected"]).reshape(()).item())
    gradient_value = float(gradient_row[metric])
    if not np.isclose(npz_value, gradient_value, rtol=1.0e-6, atol=1.0e-7):
        raise ValueError(f"Gradient CSV/NPZ {metric} mismatch for {domain}")
    for key, npz_value, csv_value in (
        ("constraint_norm", npz_norm, float(gradient_row["constraint_norm"])),
        ("constraint_radius", npz_radius, float(gradient_row["constraint_radius"])),
        ("constraint_ratio", npz_ratio, float(gradient_row["constraint_ratio"])),
    ):
        if not np.isclose(npz_value, csv_value, rtol=1.0e-6, atol=1.0e-7):
            raise ValueError(f"Gradient CSV/NPZ {key} mismatch for {domain}")

    cnop_radius = float(cnop_row["constraint_radius"])
    gradient_radius = float(gradient_row["constraint_radius"])
    if not np.isclose(cnop_radius, gradient_radius, rtol=0.0, atol=radius_atol):
        raise ValueError(
            f"Unequal event-L2 budgets for {domain}: CNOP={cnop_radius}, gradient={gradient_radius}"
        )
    cnop_ratio = float(cnop_row["constraint_ratio"])
    gradient_ratio = float(gradient_row["constraint_ratio"])
    if cnop_ratio > 1.0 + ratio_tol or gradient_ratio > 1.0 + ratio_tol:
        raise ValueError(f"Constraint violation for {domain}: CNOP={cnop_ratio}, gradient={gradient_ratio}")
    random_values = np.asarray([float(row[metric]) for row in random_rows], dtype=np.float64)
    if not np.isfinite(random_values).all():
        raise ValueError(f"Non-finite random-control {metric} values for {domain}")

    return DomainComparison(
        domain=domain,
        source=source,
        target_year=year,
        cnop_value=float(cnop_row["best_objective"] if metric == "objective" else cnop_row[metric]),
        gradient_value=gradient_value,
        random_values=random_values,
        cnop_norm=float(cnop_row["constraint_norm"]),
        cnop_radius=cnop_radius,
        cnop_ratio=cnop_ratio,
        gradient_norm=npz_norm,
        gradient_radius=gradient_radius,
        gradient_ratio=gradient_ratio,
        gradient_projected=npz_projected,
    )


def plot_comparison(products: list[DomainComparison], output_dir: Path, metric: str, dpi: int) -> Path:
    # Import lazily so the data-integrity loader can be unit tested without a
    # plotting stack on lightweight login nodes.
    import matplotlib.pyplot as plt

    ylabel = "Late-3-month Niño3.4 response (perturbed − baseline)" if metric == "objective" else "Lead-12 Niño3.4 response (perturbed − baseline)"
    fig, axes = plt.subplots(1, len(products), figsize=(13.4, 4.45), sharey=True)
    if len(products) == 1:
        axes = [axes]
    rng = np.random.default_rng(1042)
    for ax, item in zip(axes, products, strict=True):
        jitter = rng.uniform(-0.14, 0.14, size=item.random_values.size)
        violin = ax.violinplot(item.random_values, positions=[0.0], widths=0.62, showextrema=False)
        for body in violin["bodies"]:
            body.set_facecolor(DOMAIN_COLORS[item.domain])
            body.set_edgecolor("none")
            body.set_alpha(0.20)
        ax.scatter(jitter, item.random_values, s=10, color=DOMAIN_COLORS[item.domain], alpha=0.30, linewidths=0)
        ax.scatter([1.0], [item.cnop_value], marker="*", s=175, color=CNOP_COLOR, zorder=5, label="CNOP")
        ax.scatter([2.0], [item.gradient_value], marker="D", s=55, color=GRADIENT_COLOR, zorder=5, label="Zero-state gradient")
        ax.axhline(0.0, color="#64748B", linewidth=0.8, linestyle=":")
        ax.set_xlim(-0.55, 2.55)
        ax.set_xticks([0.0, 1.0, 2.0], ["Random\ncontrols", "CNOP", "Zero-state\ngradient"])
        ax.set_title(DOMAIN_LABELS[item.domain], fontweight="bold")
        ax.grid(axis="y", color="#94A3B8", alpha=0.22, linewidth=0.6)
        ax.spines[["top", "right"]].set_visible(False)
        p95 = float(np.percentile(item.random_values, 95))
        cnop_percentile = float(100.0 * np.mean(item.random_values <= item.cnop_value))
        gradient_percentile = float(100.0 * np.mean(item.random_values <= item.gradient_value))
        ax.text(
            0.02,
            0.98,
            f"R={item.cnop_radius:.3f}\n"
            f"CNOP ||δ||/R={item.cnop_ratio:.3f}\n"
            f"Grad ||δ||/R={item.gradient_ratio:.3f}\n"
            f"Random P95={p95:+.3f}\n"
            f"Percentile: CNOP {cnop_percentile:.1f}, grad {gradient_percentile:.1f}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=7.2,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.83, "pad": 1.8},
        )
    axes[0].set_ylabel(ylabel)
    fig.suptitle(
        f"Matched event-L2 comparison: CNOP vs zero-state local gradient vs random | "
        f"{products[0].source} {products[0].target_year}",
        fontsize=12,
        fontweight="bold",
        y=0.98,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.91))
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = "late3m" if metric == "objective" else "lead12"
    path = output_dir / f"fig09_cnop_gradient_random_{suffix}.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    args = parse_args()
    products = [
        load_domain_comparison(args.experiment_dir, domain, args.metric, args.radius_atol, args.ratio_tol)
        for domain in DOMAINS
    ]
    source_years = {(item.source, item.target_year) for item in products}
    if len(source_years) != 1:
        raise ValueError(f"All domains must use the same case, got {sorted(source_years)}")
    path = plot_comparison(products, args.output_dir, args.metric, args.dpi)
    print(f"[figure] {path}", flush=True)


if __name__ == "__main__":
    main()
