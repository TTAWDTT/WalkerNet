"""Remap CMIP6 SSP scenario data to WalkerNet's 1-degree grid with CDO.

输入目录约定：

    <input_root>/<scenario>/<source>/<variable or variable-resolution>/*.nc

例如 ``ssp126/GFDL-ESM4/tos-50/*.nc`` 或 ``ssp126/IPSL-CM6A-LR/tauu-250/*.nc``。
脚本只认 NetCDF 文件名里的变量、模式和 scenario，不依赖变量目录是否带 ``-50``、
``-250`` 或其它后缀。

输出目录：

    <output_root>/<scenario>/<source>/<variable>_1x1.nc

目标网格固定为 WalkerNet 训练使用的 180x360：
lat=-89.5..89.5, lon=0.5..359.5。
"""

from __future__ import annotations

import argparse
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path


VARIABLES = ("tos", "zos", "tauu", "tauv")
DEFAULT_EXPECTED_MONTHS = 1032  # 2015-01 到 2100-12
FILENAME_RE = re.compile(
    r"^(?P<var>[^_]+)_(?P<table>[^_]+)_(?P<source>.+?)_"
    r"(?P<scenario>ssp\d+)_(?P<member>[^_]+)_(?P<grid>[^_]+)_"
    r"(?P<start>\d{6})-(?P<end>\d{6})\.nc$"
)


@dataclass(frozen=True)
class VariableJob:
    """一个 CDO remap 任务。"""

    scenario: str
    source_id: str
    variable: str
    files: tuple[Path, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Remap CMIP6 SSP files to WalkerNet 180x360 grid with CDO.")
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--cdo-bin", default="cdo")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--scenarios", nargs="*", default=None, help="例如 ssp126 ssp245；默认处理全部。")
    parser.add_argument("--models", nargs="*", default=None, help="可选：只处理这些模式。")
    parser.add_argument("--variables", nargs="*", choices=VARIABLES, default=list(VARIABLES))
    parser.add_argument("--expected-months", type=int, default=DEFAULT_EXPECTED_MONTHS)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    jobs = discover_jobs(args.input_root, args.scenarios, args.models, tuple(args.variables))
    if not jobs:
        raise SystemExit(f"No SSP NetCDF jobs found under {args.input_root}")

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

        run_cdo_remap(
            cdo_bin=args.cdo_bin,
            threads=args.threads,
            grid_path=grid_path,
            variable=job.variable,
            input_files=job.files,
            output_path=output_path,
            expected_months=args.expected_months,
        )
        print(f"done {output_path}", flush=True)


def discover_jobs(
    input_root: Path,
    selected_scenarios: list[str] | None,
    selected_models: list[str] | None,
    variables: tuple[str, ...],
) -> list[VariableJob]:
    scenario_dirs = [path for path in sorted(input_root.iterdir()) if path.is_dir()]
    if selected_scenarios:
        allowed_scenarios = set(selected_scenarios)
        scenario_dirs = [path for path in scenario_dirs if path.name in allowed_scenarios]

    jobs: list[VariableJob] = []
    for scenario_dir in scenario_dirs:
        model_dirs = [path for path in sorted(scenario_dir.iterdir()) if path.is_dir()]
        if selected_models:
            allowed_models = set(selected_models)
            model_dirs = [path for path in model_dirs if path.name in allowed_models]

        for model_dir in model_dirs:
            by_var: dict[str, list[Path]] = {var: [] for var in variables}
            source_ids: dict[str, str] = {}
            scenario_ids: set[str] = set()
            for path in sorted(model_dir.rglob("*.nc")):
                match = FILENAME_RE.match(path.name)
                if not match:
                    print(f"warn: ignore unmatched file name: {path}", flush=True)
                    continue
                variable = match.group("var")
                scenario = match.group("scenario")
                if variable not in by_var or scenario != scenario_dir.name:
                    continue
                by_var[variable].append(path)
                source_ids[variable] = match.group("source")
                scenario_ids.add(scenario)

            missing = [var for var, files in by_var.items() if not files]
            if missing:
                raise SystemExit(f"Missing variables under {model_dir}: {missing}")
            if len(scenario_ids) != 1:
                raise SystemExit(f"Unexpected scenarios under {model_dir}: {sorted(scenario_ids)}")

            source_id = source_ids.get("tos") or next(iter(source_ids.values()))
            for variable in variables:
                jobs.append(
                    VariableJob(
                        scenario=scenario_dir.name,
                        source_id=source_id,
                        variable=variable,
                        files=tuple(sorted(by_var[variable], key=file_start_time)),
                    )
                )
    return jobs


def file_start_time(path: Path) -> str:
    match = FILENAME_RE.match(path.name)
    return match.group("start") if match else path.name


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
    variable: str,
    input_files: tuple[Path, ...],
    output_path: Path,
    expected_months: int,
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
        f"-selname,{variable}",
        "-mergetime",
        *[str(path) for path in input_files],
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

    validate_output(cdo_bin, tmp_path, expected_months)
    tmp_path.replace(output_path)


def validate_output(cdo_bin: str, path: Path, expected_months: int) -> None:
    sinfo = subprocess.run([cdo_bin, "-s", "sinfo", str(path)], text=True, capture_output=True, check=False)
    if sinfo.returncode != 0:
        raise SystemExit(f"CDO sinfo failed for {path}:\n{sinfo.stderr}")
    text = sinfo.stdout + sinfo.stderr
    if "points=64800 (360x180)" not in text:
        raise SystemExit(f"Unexpected output grid for {path}:\n{text}")
    if expected_months > 0 and f"time : {expected_months} steps" not in text:
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
