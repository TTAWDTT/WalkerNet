"""Cluster CNOP perturbations and create a representative 10-case subset."""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cluster saved CNOP perturbations.")
    parser.add_argument("--cnop-dir", type=Path, required=True)
    parser.add_argument("--summary-name", type=str, default="cnop_summary_forecast_clim.csv")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--representative-count", type=int, default=10)
    parser.add_argument("--cluster-count", type=int, default=4)
    return parser.parse_args()


def short_name(name: str) -> str:
    return (
        name.replace("EC-Earth3_", "EC ")
        .replace("GFDL-ESM4_", "GFDL ")
        .replace("MPI-ESM1-2-HR_", "MPI ")
        .replace("IPSL-CM6A-LR_", "IPSL ")
        .replace("CESM2_", "CESM ")
    )


def kmeans(X: np.ndarray, k: int, seeds: int = 80) -> tuple[np.ndarray, float]:
    best_labels: np.ndarray | None = None
    best_inertia = float("inf")
    for seed in range(seeds):
        rng = np.random.default_rng(seed)
        centers = X[rng.choice(len(X), k, replace=False)].copy()
        labels = np.zeros(len(X), dtype=np.int64)
        for _ in range(200):
            dist = 1.0 - X @ centers.T
            new_labels = dist.argmin(axis=1)
            new_centers = []
            for cluster_idx in range(k):
                if np.any(new_labels == cluster_idx):
                    center = X[new_labels == cluster_idx].mean(axis=0)
                    center = center / (np.linalg.norm(center) + 1e-12)
                else:
                    center = centers[cluster_idx]
                new_centers.append(center)
            new_centers = np.vstack(new_centers)
            if np.all(new_labels == labels):
                break
            labels = new_labels
            centers = new_centers
        inertia = float(sum(1.0 - X[i] @ centers[labels[i]] for i in range(len(X))))
        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels.copy()
    if best_labels is None:
        raise RuntimeError("kmeans failed")
    return best_labels, best_inertia


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or args.cnop_dir / "cluster"
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = pd.read_csv(args.cnop_dir / args.summary_name)
    names: list[str] = []
    fields: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    lat = lon = None
    for row in rows.itertuples(index=False):
        source = str(row.source)
        year = int(row.target_year)
        path = args.cnop_dir / f"case_{source}_{year}.npz"
        with np.load(path) as payload:
            delta = np.asarray(payload["delta_norm"], dtype=np.float64)
            fields.append(delta)
            masks.append(np.isfinite(delta).all(axis=0) & (np.abs(delta).sum(axis=0) > 1e-10))
            lat = np.asarray(payload["lat"], dtype=np.float64)
            lon = np.asarray(payload["lon"], dtype=np.float64)
        names.append(f"{source}_{year}")
    if lat is None or lon is None:
        raise RuntimeError("No CNOP npz files found.")

    common_mask = np.logical_and.reduce(masks)
    weights = np.sqrt(np.clip(np.cos(np.deg2rad(lat)), 0.0, 1.0))[:, None]
    X = []
    for delta in fields:
        vec = np.concatenate([(delta[0] * weights)[common_mask], (delta[1] * weights)[common_mask]])
        X.append(vec / (np.linalg.norm(vec) + 1e-12))
    X = np.vstack(X)
    similarity = X @ X.T
    centered = X - X.mean(axis=0, keepdims=True)
    U, s, _ = np.linalg.svd(centered, full_matrices=False)
    pc = U[:, :2] * s[:2]
    explained = (s * s) / np.sum(s * s)
    labels, inertia = kmeans(X, min(args.cluster_count, len(X)))

    rows = rows.copy()
    rows["case"] = names
    rows["cluster"] = labels + 1
    rows["pc1"] = pc[:, 0]
    rows["pc2"] = pc[:, 1]
    rows.to_csv(output_dir / "cnop_cluster_summary.csv", index=False)
    np.savez_compressed(
        output_dir / "cnop_cluster_arrays.npz",
        names=np.asarray(names),
        similarity=similarity,
        pc=pc,
        explained=explained[:8],
        labels=labels + 1,
        common_mask=common_mask,
    )

    # 代表样本：每个簇先取离簇中心最近的，再用 lead_delta 较强的样本补足。
    representatives: list[int] = []
    for cluster_idx in sorted(set(labels.tolist())):
        idx = np.where(labels == cluster_idx)[0]
        center = X[idx].mean(axis=0)
        center = center / (np.linalg.norm(center) + 1e-12)
        representatives.append(int(idx[np.argmax(X[idx] @ center)]))
    remaining = [int(i) for i in np.argsort(-rows["lead_delta"].to_numpy()) if int(i) not in representatives]
    representatives.extend(remaining[: max(0, args.representative_count - len(representatives))])
    representatives = representatives[: args.representative_count]
    rep_rows = rows.iloc[representatives].copy()
    rep_rows.to_csv(output_dir / "representative_cases.csv", index=False)

    rep_dir = args.cnop_dir / "representative_10"
    rep_dir.mkdir(parents=True, exist_ok=True)
    rep_rows.to_csv(rep_dir / "cnop_summary_forecast_clim.csv", index=False)
    rep_rows.to_csv(rep_dir / "cnop_summary.csv", index=False)
    for row in rep_rows.itertuples(index=False):
        source = str(row.source)
        year = int(row.target_year)
        for suffix in (".npz", "_history.json", "_candidates.json"):
            src = args.cnop_dir / f"case_{source}_{year}{suffix}"
            if src.exists():
                shutil.copy2(src, rep_dir / src.name)
    method = args.cnop_dir / "method.json"
    if method.exists():
        shutil.copy2(method, rep_dir / method.name)

    fig = plt.figure(figsize=(13.5, 5.8), dpi=190)
    ax = fig.add_subplot(1, 2, 1)
    palette = np.asarray(["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#F0E442", "#56B4E9"])
    for i, name in enumerate(names):
        ax.scatter(pc[i, 0], pc[i, 1], s=56, c=palette[labels[i] % len(palette)], edgecolor="black", linewidth=0.45)
        if i in representatives:
            ax.text(pc[i, 0], pc[i, 1], "  " + short_name(name), fontsize=6.4, va="center")
    ax.axhline(0, color="0.86", lw=0.8)
    ax.axvline(0, color="0.86", lw=0.8)
    ax.set_xlabel(f"PC1 ({explained[0] * 100:.1f}%)")
    ax.set_ylabel(f"PC2 ({explained[1] * 100:.1f}%)")
    ax.set_title(f"CNOP perturbation PCA, k={min(args.cluster_count, len(X))}")
    ax.grid(True, alpha=0.22)
    ax2 = fig.add_subplot(1, 2, 2)
    order = np.lexsort((rows["pc1"].to_numpy(), labels))
    im = ax2.imshow(similarity[order][:, order], vmin=-1, vmax=1, cmap="RdBu_r")
    ax2.set_title("Cosine similarity, ordered by cluster")
    ax2.set_xticks([])
    ax2.set_yticks([])
    fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_dir / "cnop_cluster_pca_similarity.png", bbox_inches="tight")

    with (output_dir / "cluster_report.md").open("w", encoding="utf-8") as handle:
        handle.write(f"# CNOP cluster report\n\n")
        handle.write(f"- cases: {len(rows)}\n")
        handle.write(f"- common ocean cells: {int(common_mask.sum())}\n")
        handle.write(f"- kmeans inertia: {inertia:.4f}\n")
        handle.write(f"- explained: PC1={explained[0] * 100:.1f}%, PC2={explained[1] * 100:.1f}%\n")
        handle.write(f"- representative subset: `{rep_dir}`\n")

    print(f"[cluster] rows={len(rows)} representatives={len(rep_rows)}")
    print(f"[cluster] output={output_dir}")
    print(f"[cluster] representative_dir={rep_dir}")


if __name__ == "__main__":
    main()
