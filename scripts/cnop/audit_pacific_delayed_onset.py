"""Audit the paired Pacific delayed-onset CNOP pilot without rerunning the model."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def finite_float(row: dict[str, str], key: str) -> float:
    value = float(row[key])
    if not np.isfinite(value):
        raise ValueError(f"non-finite {key}")
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    with (root / "metadata" / "formal_manifest_v1.csv").open(newline="", encoding="utf-8") as handle:
        manifest = list(csv.DictReader(handle))
    rows: list[dict[str, object]] = []
    failures: list[str] = []
    for item in manifest:
        source, year = item["source"], int(item["target_year"])
        key = f"{source}_{year}"
        branches: dict[str, dict[str, object]] = {}
        for branch in ("normal", "delayed"):
            directory = root / branch / key
            summary_path = directory / "cnop_summary.csv"
            candidate_path = directory / "cnop_candidate_summary.csv"
            npz_path = directory / f"case_{key}.npz"
            json_path = directory / f"case_{key}_candidates.json"
            missing = [str(path.relative_to(root)) for path in (summary_path, candidate_path, npz_path, json_path) if not path.is_file()]
            if missing:
                failures.append(f"{branch}/{key}: missing {missing}")
                continue
            with summary_path.open(newline="", encoding="utf-8") as handle:
                summary_rows = list(csv.DictReader(handle))
            with candidate_path.open(newline="", encoding="utf-8") as handle:
                candidate_rows = list(csv.DictReader(handle))
            candidates = json.loads(json_path.read_text(encoding="utf-8"))
            with np.load(npz_path) as artifact:
                baseline_nino = np.asarray(artifact["baseline_nino"], dtype=np.float64)
                top_nino = np.asarray(artifact["top_cnop_nino"], dtype=np.float64)
                top_ratios = np.asarray(artifact["top_constraint_ratio"], dtype=np.float64)
                top_starts = np.asarray(artifact["top_start_idx"], dtype=np.int64)
            if len(summary_rows) != 1 or len(candidate_rows) != 3 or len(candidates) != 3 or top_nino.shape[0] != 3:
                failures.append(f"{branch}/{key}: expected 1 summary and 3 candidates")
                continue
            summary = summary_rows[0]
            early_abs = np.mean(np.abs(top_nino[:, :3] - baseline_nino[None, :3]), axis=1)
            lead12_delta = top_nino[:, 11] - baseline_nino[11]
            ratios = np.asarray([finite_float(row, "constraint_ratio") for row in candidate_rows], dtype=np.float64)
            if np.max(np.abs(ratios - 1.0)) > 5.0e-5 or np.max(np.abs(top_ratios - 1.0)) > 5.0e-5:
                failures.append(f"{branch}/{key}: constraint ratio outside tolerance")
            branches[branch] = {
                "summary": {
                    "constraint_ratio": finite_float(summary, "constraint_ratio"),
                    "lead_delta": finite_float(summary, "lead_delta"),
                    "best_early_penalty": finite_float(summary, "best_early_penalty"),
                    "best_start_idx": int(summary["best_start_idx"]),
                },
                "candidate_count": 3,
                "start_indices": [int(value) for value in top_starts.tolist()],
                "constraint_ratios": [float(value) for value in top_ratios.tolist()],
                "early_abs_delta": [float(value) for value in early_abs.tolist()],
                "lead12_delta": [float(value) for value in lead12_delta.tolist()],
            }
        if set(branches) == {"normal", "delayed"}:
            normal = branches["normal"]["early_abs_delta"]
            delayed = branches["delayed"]["early_abs_delta"]
            rows.append({
                "source": source,
                "target_year": year,
                "normal": branches["normal"],
                "delayed": branches["delayed"],
                "delayed_better_early_rank1": bool(delayed[0] <= normal[0]),
                "delayed_best_lead12_delta": float(max(branches["delayed"]["lead12_delta"])),
                "normal_best_lead12_delta": float(max(branches["normal"]["lead12_delta"])),
            })
    report = {
        "manifest_cases": len(manifest),
        "paired_cases": len(rows),
        "expected_jobs": 20,
        "summary_files": sum(1 for branch in ("normal", "delayed") for item in manifest if (root / branch / f"{item['source']}_{item['target_year']}" / "cnop_summary.csv").is_file()),
        "failures": failures,
        "cases": rows,
        "constraint_ratio_tolerance": 5.0e-5,
    }
    (root / "audit_pacific_delayed_onset.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    with (root / "audit_pacific_delayed_onset.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["source", "target_year", "normal_rank1_early_abs", "delayed_rank1_early_abs", "normal_rank1_lead12_delta", "delayed_rank1_lead12_delta", "delayed_better_early_rank1"])
        for row in rows:
            writer.writerow([row["source"], row["target_year"], row["normal"]["early_abs_delta"][0], row["delayed"]["early_abs_delta"][0], row["normal"]["lead12_delta"][0], row["delayed"]["lead12_delta"][0], row["delayed_better_early_rank1"]])
    print(json.dumps({key: report[key] for key in ("manifest_cases", "paired_cases", "summary_files", "failures")}, indent=2, ensure_ascii=False))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
