"""Merge parallel CNOP start shards into one result per perturbation domain."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.cnop.compute_tos_zos_cnop import cosine_similarity


DOMAINS = ("pacific", "atlantic_indian", "global")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-cosine-similarity", type=float, default=0.98)
    return parser.parse_args()


def load_shard(path: Path) -> tuple[dict[str, np.ndarray], list[dict[str, object]]]:
    case_path = next(path.glob("case_*.npz"))
    candidate_path = next(path.glob("case_*_candidates.json"))
    with np.load(case_path) as payload:
        arrays = {key: np.asarray(payload[key]) for key in payload.files}
    candidates = json.loads(candidate_path.read_text(encoding="utf-8"))
    return arrays, candidates


def candidate_records(path: Path) -> tuple[dict[str, np.ndarray], list[dict[str, object]]]:
    arrays, metadata = load_shard(path)
    records: list[dict[str, object]] = []
    for idx, item in enumerate(metadata):
        record = dict(item)
        record["delta_norm"] = arrays["top_delta_norm"][idx]
        record["delta_phys"] = arrays["top_delta_phys"][idx]
        record["cnop_nino"] = arrays["top_cnop_nino"][idx]
        record["cnop_3m"] = arrays["top_cnop_3m"][idx]
        records.append(record)
    return arrays, records


def select_distinct(records: list[dict[str, object]], top_k: int, threshold: float) -> list[dict[str, object]]:
    ordered = sorted(records, key=lambda item: float(item["objective"]), reverse=True)
    selected: list[dict[str, object]] = []
    for item in ordered:
        if all(
            cosine_similarity(np.asarray(item["delta_norm"]), np.asarray(previous["delta_norm"])) <= threshold
            for previous in selected
        ):
            selected.append(item)
        if len(selected) >= top_k:
            break
    return selected or ordered[:1]


def write_domain_result(
    output_dir: Path,
    domain: str,
    base: dict[str, np.ndarray],
    selected: list[dict[str, object]],
) -> None:
    domain_dir = output_dir / domain
    domain_dir.mkdir(parents=True, exist_ok=True)
    source = "GFDL-ESM4"
    year = 1995
    np.savez_compressed(
        domain_dir / f"case_{source}_{year}.npz",
        delta_norm=np.asarray(selected[0]["delta_norm"]),
        delta_phys=np.asarray(selected[0]["delta_phys"]),
        top_delta_norm=np.stack([np.asarray(item["delta_norm"]) for item in selected]),
        top_delta_phys=np.stack([np.asarray(item["delta_phys"]) for item in selected]),
        top_cnop_nino=np.stack([np.asarray(item["cnop_nino"]) for item in selected]),
        top_cnop_3m=np.stack([np.asarray(item["cnop_3m"]) for item in selected]),
        top_objective=np.asarray([item["objective"] for item in selected], dtype=np.float32),
        top_lead_nino=np.asarray([item["lead_nino"] for item in selected], dtype=np.float32),
        top_lead_delta=np.asarray([item["lead_delta"] for item in selected], dtype=np.float32),
        top_cnop_max_3m=np.asarray([item["cnop_max_3m"] for item in selected], dtype=np.float32),
        top_gain_max_3m=np.asarray([item["gain_max_3m"] for item in selected], dtype=np.float32),
        top_start_idx=np.asarray([item["start_idx"] for item in selected], dtype=np.int32),
        top_constraint_norm=np.asarray([item["constraint_norm"] for item in selected], dtype=np.float32),
        top_constraint_radius=np.asarray([item["constraint_radius"] for item in selected], dtype=np.float32),
        top_constraint_ratio=np.asarray([item["constraint_ratio"] for item in selected], dtype=np.float32),
        baseline_nino=base["baseline_nino"],
        cnop_nino=np.asarray(selected[0]["cnop_nino"]),
        baseline_3m=base["baseline_3m"],
        cnop_3m=np.asarray(selected[0]["cnop_3m"]),
        lat=base["lat"],
        lon=base["lon"],
    )
    serializable = [{key: value for key, value in item.items() if not isinstance(value, np.ndarray)} for item in selected]
    (domain_dir / f"case_{source}_{year}_candidates.json").write_text(
        json.dumps(serializable, indent=2), encoding="utf-8"
    )

    summary_path = domain_dir / "cnop_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "source", "target_year", "target_t", "observed_max_3m_abs", "baseline_max_3m",
                "baseline_lead_nino", "cnop_lead_nino", "lead_delta", "cnop_max_3m", "gain_max_3m",
                "best_start_idx", "best_objective", "constraint_norm", "constraint_radius", "constraint_ratio",
                "mask_count",
            ],
        )
        writer.writeheader()
        best = selected[0]
        writer.writerow(
            {
                "source": source,
                "target_year": year,
                "target_t": 1740,
                "observed_max_3m_abs": 0.22883670032024384,
                "baseline_max_3m": float(np.max(base["baseline_3m"])),
                "baseline_lead_nino": float(base["baseline_nino"][-1]),
                "cnop_lead_nino": best["lead_nino"],
                "lead_delta": best["lead_delta"],
                "cnop_max_3m": best["cnop_max_3m"],
                "gain_max_3m": best["gain_max_3m"],
                "best_start_idx": best["start_idx"],
                "best_objective": best["objective"],
                "constraint_norm": best["constraint_norm"],
                "constraint_radius": best["constraint_radius"],
                "constraint_ratio": best["constraint_ratio"],
                "mask_count": int(np.count_nonzero(np.asarray(best["delta_norm"]))),
            }
        )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    comparison_rows: list[dict[str, object]] = []
    for domain in DOMAINS:
        records: list[dict[str, object]] = []
        base: dict[str, np.ndarray] | None = None
        for shard in sorted((args.input_dir / domain).glob("shard_*")):
            arrays, shard_records = candidate_records(shard)
            base = arrays if base is None else base
            records.extend(shard_records)
        if base is None or not records:
            raise FileNotFoundError(f"No completed shards for {domain}")
        selected = select_distinct(records, args.top_k, args.max_cosine_similarity)
        write_domain_result(args.output_dir, domain, base, selected)
        best = selected[0]
        random_path = args.input_dir.parent / "random_controls" / f"{domain}.csv"
        with random_path.open("r", encoding="utf-8") as handle:
            random_objectives = np.asarray(
                [float(row["objective"]) for row in csv.DictReader(handle)], dtype=np.float64
            )
        objective = float(best["objective"])
        comparison_rows.append(
            {
                "domain": domain,
                "objective": objective,
                "lead_delta": best["lead_delta"],
                "cnop_max_3m": best["cnop_max_3m"],
                "gain_max_3m": best["gain_max_3m"],
                "constraint_ratio": best["constraint_ratio"],
                "best_start_idx": best["start_idx"],
                "distinct_candidates": len(selected),
                "random_mean_objective": float(random_objectives.mean()),
                "random_p95_objective": float(np.percentile(random_objectives, 95)),
                "random_max_objective": float(random_objectives.max()),
                "cnop_random_percentile": float(100.0 * np.mean(random_objectives <= objective)),
            }
        )

    with (args.output_dir / "basin_comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(comparison_rows[0]))
        writer.writeheader()
        writer.writerows(comparison_rows)
    print(args.output_dir / "basin_comparison.csv")


if __name__ == "__main__":
    main()
