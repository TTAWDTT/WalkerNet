"""用 CDO 将 CMIP6 SSP 数据重网格到 WalkerNet 1 度网格。

输入目录：``<input_root>/<scenario>/<source>/<variable-resolution>/*.nc``。
输出目录：``<output_root>/<scenario>/<source>/<variable>_1x1.nc``。

脚本会在调用 CDO 前检查每个模式/变量是否完整覆盖 2015-01 到
2100-12，发现缺月或重叠时直接停止，不提供插值或其它 fallback。
"""

from __future__ import annotations

import argparse
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path


VARIABLES = ("tos", "zos", "tauu", "tauv")
EXPECTED_START = 201501
EXPECTED_END = 210012
EXPECTED_MONTHS = 1032
FILENAME_RE = re.compile(
    r"^(?P<var>[^_]+)_(?P<table>[^_]+)_(?P<source>.+?)_"
    r"(?P<scenario>ssp\d+)_(?P<member>[^_]+)_(?P<grid>[^_]+)_"
    r"(?P<start>\d{6})-(?P<end>\d{6})\.nc$"
)


@dataclass(frozen=True)
class VariableJob:
    scenario: str
    source_id: str
    variable: str
    files: tuple[Path, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Remap CMIP6 SSP data to 180x360 with CDO")
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--cdo-bin", default="cdo")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--scenarios", nargs="*", default=None)
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--variables", nargs="*", choices=VARIABLES, default=list(VARIABLES))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    jobs = discover_jobs(args.input_root, args.scenarios, args.models, tuple(args.variables))
    if not jobs:
        raise SystemExit(f"No SSP NetCDF jobs found under {args.input_root}")

    for job in jobs:
        validate_input_timeline(job)

    args.output_root.mkdir(parents=True, exist_ok=True)
    grid_path = write_target_grid(args.output_root)
    print(f"target grid: {grid_path}", flush=True)
    print(f"jobs: {len(jobs)}", flush=True)

    for job in jobs:
        output_dir = args.output_root / job.scenario / job.source_id
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{job.variable}_1x1.nc"
        if output_path.exists() and not args.overwrite:
            print(f"skip existing {output_path}", flush=True)
            continue

        print(
            f"remap {job.scenario}/{job.source_id}/{job.variable}: "
            f"{len(job.files)} files -> {output_path}",
            flush=True,
        )
        if args.dry_run:
            continue
        run_cdo_remap(args.cdo_bin, args.threads, grid_path, job, output_path)
        print(f"done {output_path}", flush=True)


def discover_jobs(
    input_root: Path,
    selected_scenarios: list[str] | None,
    selected_models: list[str] | None,
    variables: tuple[str, ...],
) -> list[VariableJob]:
    scenario_dirs = [path for path in sorted(input_root.iterdir()) if path.is_dir()]
    if selected_scenarios:
        allowed = set(selected_scenarios)
        scenario_dirs = [path for path in scenario_dirs if path.name in allowed]

    jobs: list[VariableJob] = []
    for scenario_dir in scenario_dirs:
        model_dirs = [path for path in sorted(scenario_dir.iterdir()) if path.is_dir()]
        if selected_models:
            allowed = set(selected_models)
            model_dirs = [path for path in model_dirs if path.name in allowed]

        for model_dir in model_dirs:
            by_variable: dict[str, list[Path]] = {variable: [] for variable in variables}
            source_ids: dict[str, str] = {}
            for path in sorted(model_dir.rglob("*.nc")):
                match = FILENAME_RE.match(path.name)
                if not match:
                    print(f"warn: ignore unmatched file name: {path}", flush=True)
                    continue
                variable = match.group("var")
                if variable not in by_variable or match.group("scenario") != scenario_dir.name:
                    continue
                by_variable[variable].append(path)
                source_ids[variable] = match.group("source")

            missing = [variable for variable, files in by_variable.items() if not files]
            if missing:
                raise SystemExit(f"Missing variables under {model_dir}: {missing}")

            source_id = source_ids.get("tos") or next(iter(source_ids.values()))
            if any(value != source_id for value in source_ids.values()):
                raise SystemExit(f"Inconsistent source ids under {model_dir}: {source_ids}")
            for variable in variables:
                files = tuple(sorted(by_variable[variable], key=file_start_month))
                jobs.append(VariableJob(scenario_dir.name, source_id, variable, files))
    return jobs


def file_start_month(path: Path) -> int:
    match = FILENAME_RE.match(path.name)
    return int(match.group("start")) if match else 0


def next_month(value: int) -> int:
    year, month = divmod(value, 100)
    return (year + 1) * 100 + 1 if month == 12 else year * 100 + month + 1


def month_count(start: int, end: int) -> int:
    start_year, start_month = divmod(start, 100)
    end_year, end_month = divmod(end, 100)
    return (end_year - start_year) * 12 + end_month - start_month + 1


def validate_input_timeline(job: VariableJob) -> None:
    ranges: list[tuple[int, int, Path]] = []
    for path in job.files:
        match = FILENAME_RE.match(path.name)
        if match is None:
            raise ValueError(f"Unexpected file name: {path}")
        start = int(match.group("start"))
        end = int(match.group("end"))
        if month_count(start, end) <= 0:
            raise ValueError(f"Invalid time range: {path}")
        ranges.append((start, end, path))

    if ranges[0][0] > EXPECTED_START or ranges[-1][1] < EXPECTED_END:
        raise ValueError(
            f"{job.scenario}/{job.source_id}/{job.variable}: expected coverage "
            f"of {EXPECTED_START}-{EXPECTED_END}, got {ranges[0][0]}-{ranges[-1][1]}"
        )
    for previous, current in zip(ranges, ranges[1:]):
        expected = next_month(previous[1])
        if current[0] != expected:
            raise ValueError(
                f"{job.scenario}/{job.source_id}/{job.variable}: gap or overlap after "
                f"{previous[2].name}; expected {expected}, got {current[0]}"
            )
    selected_months = sum(
        month_count(max(start, EXPECTED_START), min(end, EXPECTED_END))
        for start, end, _path in ranges
        if start <= EXPECTED_END and end >= EXPECTED_START
    )
    if selected_months != EXPECTED_MONTHS:
        raise ValueError(
            f"{job.scenario}/{job.source_id}/{job.variable}: "
            f"expected {EXPECTED_MONTHS} selected months, got {selected_months}"
        )
    print(
        f"timeline OK {job.scenario}/{job.source_id}/{job.variable}: "
        f"{len(job.files)} files, {selected_months} selected months "
        f"from coverage {ranges[0][0]}-{ranges[-1][1]}",
        flush=True,
    )


def write_target_grid(output_root: Path) -> Path:
    grid_path = output_root / "grid_1x1_180x360.txt"
    grid_path.write_text(
        "\n".join(
            [
                "gridtype = lonlat",
                "xsize = 360",
                "ysize = 180",
                "xfirst = 0.5",
                "xinc = 1.0",
                "yfirst = -89.5",
                "yinc = 1.0",
                "",
            ]
        ),
        encoding="ascii",
    )
    return grid_path


def run_cdo_remap(
    cdo_bin: str,
    threads: int,
    grid_path: Path,
    job: VariableJob,
    output_path: Path,
) -> None:
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()
    command = [
        cdo_bin,
        "-O",
        "-P",
        str(int(threads)),
        "-f",
        "nc4",
        "-z",
        "zip_4",
        f"remapbil,{grid_path}",
        "-selyear,2015/2100",
        f"-selname,{job.variable}",
        "-mergetime",
        *[str(path) for path in job.files],
        str(tmp_path),
    ]
    print("  command: " + shell_join(command), flush=True)
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    if completed.stdout:
        print(completed.stdout, end="", flush=True)
    if completed.stderr:
        print(completed.stderr, end="", flush=True)
    if completed.returncode != 0:
        raise SystemExit(f"CDO failed with exit code {completed.returncode}: {output_path}")
    validate_output(cdo_bin, tmp_path)
    tmp_path.replace(output_path)


def validate_output(cdo_bin: str, path: Path) -> None:
    completed = subprocess.run(
        [cdo_bin, "-s", "sinfo", str(path)],
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise SystemExit(f"CDO sinfo failed for {path}:\n{completed.stderr}")
    text = completed.stdout + completed.stderr
    if "points=64800 (360x180)" not in text:
        raise SystemExit(f"Unexpected output grid for {path}:\n{text}")
    if f"time : {EXPECTED_MONTHS} steps" not in text:
        raise SystemExit(f"Unexpected time length for {path}:\n{text}")


def shell_join(parts: list[str]) -> str:
    quoted = []
    for item in parts:
        if re.search(r"[\s'\"$`]", item):
            quoted.append("'" + item.replace("'", "'\\''") + "'")
        else:
            quoted.append(item)
    return " ".join(quoted)


if __name__ == "__main__":
    main()
