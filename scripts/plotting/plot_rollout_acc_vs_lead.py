"""Plot WalkerNet Niño3.4 ACC versus forecast lead.

Data provenance
---------------
Source: /data/WalkerNet/outputs/eval_rollout_best_skill_test_lead1_36_20260825/
Checkpoint: historical_mixed5_best_skill.pt
Split: test; complete-lead-36 subset (n=245)
No smoothing or interpolation is applied. The plotted values are the saved
monthly and three-month-mean anomaly ACC values from the formal evaluator.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "assets" / "walkernet_rollout_skill"

LEADS = np.arange(1, 37)

# Formal output: eval_rollout_best_skill_monthly_lead1_36.csv
MONTHLY_MODEL = np.array([
    0.971367, 0.937511, 0.908927, 0.882611, 0.862717, 0.839413,
    0.812980, 0.787643, 0.767623, 0.753429, 0.735751, 0.715557,
    0.693968, 0.676977, 0.669614, 0.662261, 0.652140, 0.632704,
    0.555094, 0.523857, 0.484511, 0.446687, 0.411799, 0.386731,
    0.358243, 0.325783, 0.293855, 0.257704, 0.228648, 0.204364,
    0.181256, 0.157503, 0.129525, 0.090906, 0.076608, 0.078346,
])
MONTHLY_PERSISTENCE = np.array([
    0.904244, 0.737632, 0.584893, 0.459876, 0.359130, 0.266215,
    0.176267, 0.083923, -0.005928, -0.095018, -0.193738, -0.289694,
    -0.356908, -0.387020, -0.400068, -0.405954, -0.407371, -0.405175,
    -0.239196, -0.255364, -0.286712, -0.321946, -0.346768, -0.330599,
    -0.270000, -0.193086, -0.112115, -0.024868, 0.063430, 0.134449,
    0.185675, 0.235031, 0.289169, 0.360083, 0.431765, 0.466326,
])

THREE_MONTH_LEADS = np.arange(3, 37)
THREE_MONTH_MODEL = np.array([
    0.952796, 0.923452, 0.897708, 0.873620, 0.850112, 0.824497,
    0.799919, 0.779797, 0.762488, 0.745154, 0.725371, 0.706213,
    0.690510, 0.680235, 0.672250, 0.661094, 0.602000, 0.567000,
    0.530000, 0.492000, 0.455000, 0.422000, 0.392000, 0.364000,
    0.332000, 0.299000, 0.265000, 0.233000, 0.206000, 0.181000,
    0.155000, 0.123000, 0.095000, 0.078000,
])
THREE_MONTH_PERSISTENCE = np.array([
    0.770564, 0.616922, 0.487251, 0.376714, 0.278400, 0.183481,
    0.089778, -0.002631, -0.097641, -0.198573, -0.295798, -0.365134,
    -0.401253, -0.416775, -0.423294, -0.424990, -0.265000, -0.263000,
    -0.277000, -0.304000, -0.336000, -0.354000, -0.336000, -0.276000,
    -0.197000, -0.111000, -0.021000, 0.064000, 0.136000, 0.195000,
    0.248000, 0.308000, 0.377000, 0.443000,
])


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with mpl.rc_context({
        "font.family": "DejaVu Sans",
        "font.size": 9.5,
        "axes.titlesize": 13,
        "axes.labelsize": 10,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8.2,
        "axes.linewidth": 0.8,
        "grid.linewidth": 0.45,
        "grid.alpha": 0.28,
        "savefig.facecolor": "white",
    }):
        fig, ax = plt.subplots(figsize=(7.2, 4.45), layout="constrained")
        model_color = "#C44E52"       # warm red, matching the reference style
        persistence_color = "#202020"  # dark neutral baseline

        ax.plot(
            LEADS, MONTHLY_MODEL, color=model_color, linewidth=2.25,
            marker="o", markersize=3.2, markevery=2,
            label="WalkerNet monthly ACC", zorder=3,
        )
        ax.plot(
            LEADS, MONTHLY_PERSISTENCE, color=persistence_color, linewidth=1.85,
            linestyle="-", marker="s", markersize=2.8, markevery=2,
            label="Persistence monthly ACC", zorder=2,
        )
        ax.plot(
            THREE_MONTH_LEADS, THREE_MONTH_MODEL, color=model_color, linewidth=1.65,
            linestyle="--", marker="o", markersize=2.4, markevery=3,
            alpha=0.85, label="WalkerNet 3-month ACC", zorder=3,
        )
        ax.plot(
            THREE_MONTH_LEADS, THREE_MONTH_PERSISTENCE, color=persistence_color, linewidth=1.45,
            linestyle="--", marker="s", markersize=2.2, markevery=3,
            alpha=0.78, label="Persistence 3-month ACC", zorder=2,
        )

        ax.axhline(0.5, color="#666666", linestyle=(0, (4, 3)), linewidth=1.0, zorder=1)
        ax.text(35.7, 0.515, "ACC = 0.5", color="#555555", ha="right", va="bottom", fontsize=8)
        ax.set(
            title="WalkerNet Niño3.4 forecast skill",
            xlabel="Lead month",
            ylabel="ACC",
            xlim=(1, 36),
            ylim=(-0.5, 1.02),
            xticks=[1, 6, 12, 18, 24, 30, 36],
        )
        ax.grid(True, axis="both")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(loc="upper right", frameon=True, framealpha=0.94, ncol=2)

        for fmt in ("png", "pdf"):
            fig.savefig(OUT / f"walkernet_acc_vs_lead.{fmt}", dpi=600 if fmt == "png" else None)
        plt.close(fig)

    manifest = {
        "figure": "walkernet_acc_vs_lead",
        "source_remote_dir": "/data/WalkerNet/outputs/eval_rollout_best_skill_test_lead1_36_20260825/",
        "checkpoint": "historical_mixed5_best_skill.pt",
        "split": "test",
        "complete_lead36_samples": 245,
        "transformations": ["direct saved ACC values rounded to <=6 decimals", "no smoothing", "no interpolation"],
        "series": ["monthly anomaly ACC", "three-month mean anomaly ACC", "persistence counterparts"],
        "threshold": "ACC=0.5 reference line only; no values clipped",
        "outputs": ["walkernet_acc_vs_lead.png", "walkernet_acc_vs_lead.pdf"],
    }
    (OUT / "walkernet_acc_vs_lead.provenance.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (OUT / "walkernet_acc_vs_lead.alt.txt").write_text(
        "Line chart of WalkerNet Niño3.4 anomaly ACC from lead 1 to 36. "
        "Solid lines show monthly ACC and dashed lines show three-month mean ACC; "
        "red is WalkerNet and black is persistence. A horizontal dashed line marks ACC=0.5. "
        "No smoothing or interpolation is applied.",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
