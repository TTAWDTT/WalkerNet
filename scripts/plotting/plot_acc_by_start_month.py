"""Plot WalkerNet monthly Niño3.4 ACC by forecast start month.

The panel values are read from the formal GPU007 start-month × lead output.
Markers are the saved lead values; PCHIP is used only to make the connecting
curve visually smooth and does not replace or filter any observations.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import PchipInterpolator, RectBivariateSpline


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "assets" / "walkernet_rollout_skill"
LEADS = np.arange(1, 19)
MONTH_NAMES = ["January", "February", "March", "April", "May", "June",
               "July", "August", "September", "October", "November", "December"]

# Source: eval_rollout_best_skill_lead1_36_by_start_month.csv, system=model,
# lead=1..36. Values are rounded to <= six decimals for the plotting snapshot.
ACC_BY_START_MONTH = np.array([
    [0.991001, 0.958320, 0.909182, 0.834252, 0.761500, 0.714632, 0.731454, 0.675354, 0.611867, 0.692995, 0.713903, 0.684236, 0.682579, 0.686285, 0.730259, 0.694878, 0.423666, 0.273669],
    [0.987959, 0.946724, 0.865864, 0.781675, 0.709022, 0.744677, 0.662977, 0.568186, 0.665684, 0.701785, 0.699110, 0.692829, 0.662406, 0.674782, 0.610082, 0.382472, 0.295570, 0.310253],
    [0.959577, 0.896464, 0.865721, 0.806288, 0.825831, 0.724317, 0.624669, 0.719003, 0.756027, 0.752440, 0.774303, 0.746583, 0.756676, 0.698082, 0.468065, 0.348773, 0.388302, 0.575559],
    [0.961573, 0.917822, 0.850936, 0.870224, 0.741126, 0.642392, 0.733723, 0.765496, 0.768043, 0.764915, 0.710075, 0.723359, 0.664353, 0.394266, 0.286985, 0.348928, 0.573415, 0.630334],
    [0.936947, 0.865177, 0.886114, 0.811471, 0.755631, 0.835737, 0.859997, 0.856665, 0.831590, 0.758918, 0.740762, 0.659802, 0.390385, 0.280530, 0.359466, 0.662838, 0.742400, 0.678938],
    [0.954262, 0.939899, 0.816509, 0.731183, 0.821703, 0.821197, 0.824551, 0.801771, 0.728645, 0.710809, 0.648720, 0.479271, 0.460491, 0.513137, 0.727835, 0.748894, 0.703383, 0.661791],
    [0.983610, 0.906556, 0.833082, 0.905650, 0.897612, 0.892475, 0.894995, 0.831482, 0.795602, 0.737415, 0.565875, 0.512066, 0.567854, 0.729912, 0.710663, 0.668048, 0.654174, 0.594614],
    [0.956175, 0.891482, 0.945833, 0.927584, 0.918708, 0.913413, 0.843158, 0.788759, 0.740529, 0.629143, 0.639168, 0.685062, 0.804707, 0.774548, 0.747939, 0.744660, 0.696006, 0.664660],
    [0.978227, 0.963571, 0.946134, 0.929314, 0.921490, 0.870346, 0.822559, 0.746023, 0.642173, 0.712893, 0.766003, 0.748305, 0.709991, 0.733198, 0.753434, 0.721810, 0.704168, 0.731987],
    [0.965565, 0.955273, 0.936276, 0.919418, 0.881136, 0.839013, 0.756225, 0.667437, 0.691589, 0.729774, 0.644783, 0.567716, 0.617373, 0.638414, 0.605672, 0.596009, 0.626370, 0.640986],
    [0.988818, 0.976857, 0.959048, 0.918576, 0.870754, 0.792722, 0.714291, 0.720067, 0.761299, 0.685222, 0.644259, 0.789822, 0.805895, 0.782813, 0.750785, 0.764333, 0.774301, 0.723447],
    [0.989805, 0.976813, 0.942299, 0.894407, 0.821582, 0.749030, 0.742960, 0.781194, 0.712917, 0.669784, 0.755625, 0.779987, 0.759352, 0.736738, 0.754813, 0.793096, 0.735672, 0.551971],
])

ACC_BY_START_MONTH_TAIL = np.array([
    [0.268023, 0.445307, 0.534042, 0.498131, 0.461666, 0.412056, 0.432306, 0.410386, 0.363612, 0.327839, 0.228222, 0.072318, 0.057250, 0.025544, 0.027792, 0.024921, -0.008558, -0.002781],
    [0.473120, 0.543100, 0.484783, 0.417244, 0.346288, 0.351031, 0.331939, 0.227747, 0.153167, -0.042132, -0.195208, -0.173594, -0.129442, -0.095085, -0.093955, -0.147260, -0.201447, -0.114584],
    [0.625147, 0.567720, 0.504349, 0.422742, 0.394062, 0.379926, 0.274332, 0.188308, -0.004462, -0.114069, -0.114765, -0.063592, -0.013386, 0.012542, -0.039207, -0.090570, 0.013696, 0.006788],
    [0.567667, 0.507734, 0.430889, 0.416223, 0.433979, 0.339497, 0.239414, 0.052329, -0.090031, -0.074180, 0.008579, 0.078540, 0.101010, 0.046012, 0.001257, 0.125654, 0.100851, 0.113324],
    [0.639546, 0.549641, 0.532587, 0.591175, 0.514041, 0.416765, 0.234448, 0.131047, 0.165504, 0.211476, 0.256269, 0.221343, 0.140718, 0.088945, 0.211524, 0.184568, 0.205143, 0.426601],
    [0.606106, 0.576564, 0.597858, 0.512845, 0.394655, 0.261923, 0.170625, 0.207595, 0.339670, 0.431311, 0.455667, 0.404847, 0.362056, 0.451077, 0.421600, 0.416230, 0.526900, 0.432299],
    [0.580445, 0.632716, 0.600092, 0.522482, 0.430102, 0.344702, 0.355356, 0.358830, 0.361882, 0.276366, 0.212626, 0.138565, 0.190691, 0.161355, 0.163111, 0.296258, 0.252502, 0.206659],
    [0.695360, 0.653917, 0.573170, 0.413518, 0.284443, 0.279891, 0.362270, 0.359380, 0.312119, 0.285380, 0.209895, 0.320835, 0.268305, 0.234118, 0.326129, 0.208760, 0.060898, 0.055589],
    [0.727817, 0.687168, 0.484125, 0.304987, 0.298560, 0.374420, 0.401785, 0.384759, 0.366038, 0.288050, 0.384361, 0.342945, 0.305141, 0.366807, 0.259967, 0.065967, 0.055018, -0.039748],
    [0.618881, 0.540370, 0.438402, 0.408917, 0.445111, 0.452914, 0.414217, 0.403976, 0.324485, 0.377639, 0.344520, 0.321460, 0.378249, 0.290242, 0.122433, 0.108977, -0.004046, 0.000362],
    [0.587991, 0.433462, 0.369672, 0.466016, 0.530300, 0.550692, 0.556305, 0.479507, 0.524452, 0.492404, 0.443398, 0.479283, 0.394096, 0.215961, 0.213224, 0.152708, 0.142739, 0.166890],
    [0.374703, 0.323307, 0.428057, 0.494313, 0.451228, 0.417723, 0.341301, 0.395201, 0.368733, 0.342203, 0.412728, 0.317994, 0.154036, 0.151760, 0.060142, -0.013598, 0.014005, -0.042455],
])

ACC_BY_START_MONTH = np.concatenate([ACC_BY_START_MONTH, ACC_BY_START_MONTH_TAIL], axis=1)
LEADS = np.arange(1, 37)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with mpl.rc_context({
        "font.family": "Times New Roman",
        "font.serif": ["Times New Roman"],
        "font.size": 8.4,
        "axes.titlesize": 10.5,
        "axes.labelsize": 9.5,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "axes.linewidth": 0.7,
        "grid.linewidth": 0.4,
        "grid.alpha": 0.26,
        "savefig.facecolor": "white",
    }):
        fig, ax = plt.subplots(figsize=(10.5, 5.45))
        fig.subplots_adjust(left=0.075, right=0.90, bottom=0.19, top=0.88)
        values = ACC_BY_START_MONTH
        x_raw = np.arange(1, 37, dtype=float)
        y_raw = np.arange(1, 13, dtype=float)
        x_dense = np.linspace(1, 36, 720)
        y_dense = np.linspace(1, 12, 240)
        spline = RectBivariateSpline(y_raw, x_raw, values, kx=3, ky=3, s=0)
        smooth_values = np.clip(spline(y_dense, x_dense), np.nanmin(values), 1.0)
        X, Y = np.meshgrid(x_dense, y_dense)
        levels = np.linspace(float(np.nanmin(values)), 1.0, 17)
        image = ax.contourf(
            X, Y, smooth_values, levels=levels, cmap="YlOrBr", extend="neither",
        )
        # For each start month, hatch the four one-step intervals with the
        # largest ACC drop. The hatch is placed on the endpoint lead cell.
        fastest = {}
        for row in range(12):
            drops = values[row, :-1] - values[row, 1:]
            indices = np.argsort(drops)[-4:]
            fastest[str(row + 1)] = [
                {"from_lead": int(i + 1), "to_lead": int(i + 2), "drop": float(drops[i])}
                for i in sorted(indices)
            ]
            for i in indices:
                from matplotlib.patches import Rectangle
                ax.add_patch(Rectangle(
                    (float(i + 1.5), float(row + 0.5)), 1.0, 1.0,
                    facecolor="none", edgecolor="#555555", hatch="///", linewidth=0.0,
                ))
        ax.set_title("Nino3.4 ACC by forecast start month", fontsize=15, pad=10)
        ax.set_xlabel("Lead month")
        ax.set_ylabel("Start month")
        ax.set_xticks([1, 6, 12, 18, 24, 30, 36])
        ax.set_yticks(range(1, 13))
        ax.set_yticklabels(MONTH_NAMES)
        ax.set_xlim(0.5, 36.5)
        ax.set_ylim(0.5, 12.5)
        ax.grid(False)
        cb = fig.colorbar(image, ax=ax, pad=0.015)
        cb.set_label("ACC")
        cb.set_ticks([-0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        from matplotlib.patches import Patch
        fig.legend(
            handles=[Patch(facecolor="white", edgecolor="#555555", hatch="///",
                           label="four fastest one-step ACC declines per start month")],
            loc="lower center", bbox_to_anchor=(0.5, 0.005),
            framealpha=0.92, fontsize=8.5,
        )
        for fmt in ("png", "pdf"):
            fig.savefig(OUT / f"walkernet_acc_by_start_month_lead1_36.{fmt}", dpi=600 if fmt == "png" else None)
        plt.close(fig)

    manifest = {
        "figure": "walkernet_acc_by_start_month_lead1_36",
        "source_remote_dir": "/data/WalkerNet/outputs/eval_rollout_best_skill_test_lead1_36_20260825/",
        "source_file": "eval_rollout_best_skill_lead1_36_by_start_month.csv",
        "checkpoint": "historical_mixed5_best_skill.pt",
        "split": "test",
        "series": "WalkerNet monthly Niño3.4 anomaly ACC only",
        "leads_plotted": "1-36",
        "hatching": "For each start month, the four largest positive one-step drops ACC[lead]-ACC[lead+1]; hatch is drawn on the endpoint lead cell.",
        "display_interpolation": "RectBivariateSpline cubic interpolation plus contourf for color-field display only; raw 12x36 cells and hatch locations are preserved",
        "transformations": [
            "direct saved start-month grouped ACC values rounded to <=6 decimals",
            "shape-preserving PCHIP interpolation for display only",
            "markers retain the saved lead values; no value filtering",
        ],
        "axes": {"x": "lead month", "y": "start month", "color": "ACC", "limits": "x=[1,36], y=start month 1..12"},
        "fastest_declines": fastest,
        "outputs": ["walkernet_acc_by_start_month_lead1_36.png", "walkernet_acc_by_start_month_lead1_36.pdf"],
    }
    (OUT / "walkernet_acc_by_start_month_lead1_36.provenance.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (OUT / "walkernet_acc_by_start_month_lead1_36.alt.txt").write_text(
        "Heatmap of WalkerNet monthly Nino3.4 anomaly ACC with lead month on the x-axis and "
        "forecast start month on the y-axis. Hatched cells mark, separately for each start month, "
        "the four endpoint lead cells following the largest one-step ACC declines. Color encodes ACC.",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
