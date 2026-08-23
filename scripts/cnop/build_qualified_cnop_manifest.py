"""Freeze the qualified CNOP run records used by maps and numerical audits."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


SCALE_PATTERN = re.compile(r"^scale_(\d+)p(\d+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a manifest of qualified CNOP run records.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-name", default="cnop_summary.csv")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--require-runs", type=int, default=0, help="Fail unless exactly this many run records qualify.")
    return parser.parse_args()


def scale_from_path(path: Path, root: Path) -> str:
    for part in path.relative_to(root).parts:
        match = SCALE_PATTERN.match(part)
        if match:
            return f"{int(match.group(1))}.{match.group(2)}"
    raise ValueError(f"Could not infer scale directory from {path}")


def main() -> None:
    args = parse_args()
    selected: list[dict[str, object]] = []
    for summary_path in sorted(args.root.rglob(args.summary_name)):
        with summary_path.open("r", newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                baseline = float(row["baseline_max_3m"])
                perturbed = float(row["cnop_max_3m"])
                if baseline < args.threshold and perturbed >= args.threshold:
                    selected.append(
                        {
                            "scale": scale_from_path(summary_path, args.root),
                            "source": row["source"],
                            "target_year": int(row["target_year"]),
                            "cnop_dir": str(summary_path.parent),
                            "summary_name": args.summary_name,
                        }
                    )
    selected.sort(key=lambda row: (float(str(row["scale"])), str(row["source"]), int(row["target_year"])))
    if args.require_runs and len(selected) != args.require_runs:
        raise ValueError(f"Expected {args.require_runs} qualified run records, found {len(selected)}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["scale", "source", "target_year", "cnop_dir", "summary_name"])
        writer.writeheader()
        writer.writerows(selected)
    print(args.output)
    print(f"qualified_runs={len(selected)}")


if __name__ == "__main__":
    main()
