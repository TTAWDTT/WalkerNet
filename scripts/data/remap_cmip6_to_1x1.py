"""用 CDO 将 CMIP6 historical 数据重网格到 WalkerNet 的 1 度网格。

这个脚本只负责把每个模式、每个变量分别整理成统一的 180x360 NetCDF：

    <output_root>/<source_id>/<variable>_1x1.nc

重要约定：
    1. 只使用 CDO 做 remap，不提供 scipy/xESMF fallback。
    2. 目标网格固定为 lat=-89.5..89.5, lon=0.5..359.5。
    3. 每个输出文件内部时间长度应为 1850-01 到 2014-12 共 1980 个月。
    4. 这里只做“按模式独立重网格”，不做多模式混合训练数据拼接。

服务器示例：
    python scripts/data/remap_cmip6_to_1x1.py \
        --input-root /path/to/CMIP6-historical-data \
        --output-root /path/to/cmip6_1x1 \
        --cdo-bin cdo \
        --threads 16 \
        --overwrite
"""

from __future__ import annotations

import argparse
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path


VARIABLES = ("tos", "zos", "tauu", "tauv")
EXPECTED_MONTHS = 1980
FILENAME_RE = re.compile(
    r"^(?P<var>[^_]+)_(?P<table>[^_]+)_(?P<source>.+?)_historical_"
    r"(?P<member>[^_]+)_(?P<grid>[^_]+)_(?P<start>\d{6})-(?P<end>\d{6})\.nc$"
)


@dataclass(frozen=True)
class VariableJob:
    """一个 CDO 任务：同一个模式、同一个变量的全部历史文件。"""

    source_id: str
    variable: str
    files: tuple[Path, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Remap CMIP6 files to WalkerNet 180x360 grid with CDO.")
    parser.add_argument("--input-root", type=Path, required=True, help="CMIP6 原始数据根目录。")
    parser.add_argument("--output-root", type=Path, required=True, help="重网格后的输出根目录。")
    parser.add_argument("--cdo-bin", default="cdo", help="CDO 可执行文件路径。")
    parser.add_argument("--threads", type=int, default=8, help="传给 CDO -P 的 OpenMP 线程数。")
    parser.add_argument("--models", nargs="*", default=None, help="可选：只处理这些顶层模式目录。")
    parser.add_argument("--variables", nargs="*", choices=VARIABLES, default=list(VARIABLES), help="可选：只处理部分变量。")
    parser.add_argument("--overwrite", action="store_true", help="覆盖已有输出。")
    parser.add_argument("--dry-run", action="store_true", help="只打印任务，不真正执行。")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    variables = tuple(args.variables)
    jobs = discover_jobs(args.input_root, args.models, variables)
    if not jobs:
        raise SystemExit(f"No CMIP6 NetCDF jobs found under {args.input_root}")

    args.output_root.mkdir(parents=True, exist_ok=True)
    grid_path = write_target_grid(args.output_root)
    print(f"target grid: {grid_path}", flush=True)
    print(f"jobs: {len(jobs)}", flush=True)

    for job in jobs:
        output_dir = args.output_root / job.source_id
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{job.variable}_1x1.nc"

        if output_path.exists() and not args.overwrite:
            print(f"skip existing {output_path}", flush=True)
            continue

        print(
            f"remap {job.source_id}/{job.variable}: "
            f"{len(job.files)} files -> {output_path}",
            flush=True,
        )
        if args.dry_run:
            continue

        run_cdo_remap(
            cdo_bin=args.cdo_bin,
            threads=args.threads,
            grid_path=grid_path,
            input_files=job.files,
            output_path=output_path,
        )
        print(f"done {output_path}", flush=True)


def discover_jobs(input_root: Path, selected_models: list[str] | None, variables: tuple[str, ...]) -> list[VariableJob]:
    """从目录树中找出每个模式、每个变量的文件列表。"""

    model_dirs = [path for path in sorted(input_root.iterdir()) if path.is_dir()]
    if selected_models:
        selected = set(selected_models)
        model_dirs = [path for path in model_dirs if path.name in selected]

    jobs: list[VariableJob] = []
    for model_dir in model_dirs:
        by_var: dict[str, list[Path]] = {var: [] for var in variables}
        source_ids: dict[str, str] = {}

        for path in sorted(model_dir.rglob("*.nc")):
            match = FILENAME_RE.match(path.name)
            if not match:
                print(f"warn: ignore unmatched file name: {path}", flush=True)
                continue

            variable = match.group("var")
            if variable not in by_var:
                continue

            by_var[variable].append(path)
            source_ids[variable] = match.group("source")

        missing = [var for var, files in by_var.items() if not files]
        if missing:
            raise SystemExit(f"Missing variables under {model_dir}: {missing}")

        # 文件名里的 source_id 比顶层目录更精确，例如 EC-Earth -> EC-Earth3。
        source_id = source_ids.get("tos") or next(iter(source_ids.values()))
        for variable in variables:
            files = tuple(sorted(by_var[variable], key=file_start_time))
            jobs.append(VariableJob(source_id=source_id, variable=variable, files=files))

    return jobs


def file_start_time(path: Path) -> str:
    match = FILENAME_RE.match(path.name)
    if not match:
        return path.name
    return match.group("start")


def write_target_grid(output_root: Path) -> Path:
    """写出 CDO 目标网格描述文件。"""

    grid_path = output_root / "grid_1x1_180x360.txt"
    grid_text = "\n".join(
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
    )
    grid_path.write_text(grid_text, encoding="ascii")
    return grid_path


def run_cdo_remap(
    cdo_bin: str,
    threads: int,
    grid_path: Path,
    input_files: tuple[Path, ...],
    output_path: Path,
) -> None:
    """调用 CDO：先按时间合并，再双线性重网格。失败时直接抛错退出。"""

    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()

    # CDO 链式表达式：remapbil 作用在 mergetime 的结果上。
    command = [
        cdo_bin,
        "-O",
        "-P",
        str(int(threads)),
        f"remapbil,{grid_path}",
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

    validate_output(cdo_bin, tmp_path)
    tmp_path.replace(output_path)


def validate_output(cdo_bin: str, path: Path) -> None:
    """用 CDO 自己检查输出网格和时间长度。"""

    sinfo = subprocess.run([cdo_bin, "-s", "sinfo", str(path)], text=True, capture_output=True, check=False)
    if sinfo.returncode != 0:
        raise SystemExit(f"CDO sinfo failed for {path}:\n{sinfo.stderr}")

    text = sinfo.stdout + sinfo.stderr
    if "points=64800 (360x180)" not in text:
        raise SystemExit(f"Unexpected output grid for {path}:\n{text}")
    if f"time : {EXPECTED_MONTHS} steps" not in text:
        raise SystemExit(f"Unexpected time length for {path}:\n{text}")


def shell_join(parts: list[str]) -> str:
    """只用于日志展示，不参与实际执行。"""

    quoted = []
    for part in parts:
        if re.search(r"[\s'\"$`]", part):
            quoted.append("'" + part.replace("'", "'\\''") + "'")
        else:
            quoted.append(part)
    return " ".join(quoted)


if __name__ == "__main__":
    main()
