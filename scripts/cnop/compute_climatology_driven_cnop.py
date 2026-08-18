"""Compute CNOP on source-wise climatology-driven WalkerNet rollouts.

这个脚本不再从真实年份里挑 Jan-Dec case，而是为每个 source 构造一个
``1-12 月训练期月气候态`` 输入窗口。这样得到的是一个“平均背景态”实验：

1. 用每个 source 自己的训练期四变量月气候态作为 WalkerNet 输入；
2. 从这个气候态输入自回归 rollout 12 个月；
3. 只在输入第 12 个月的 TOS/ZOS 上优化 CNOP；
4. 目标函数默认仍为 lead-12 的 ``perturbed - baseline`` Niño3.4 响应。

它复用 ``compute_tos_zos_cnop.py`` 里的优化、约束和输出逻辑，只替换 case
选择、输入窗口构造和目标月份读取方式。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.cnop.compute_tos_zos_cnop as cnop  # noqa: E402
from src.dataset import WalkerDataset  # noqa: E402
from src.utils import load_config, set_seed  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="WalkerNet climatology-driven TOS/ZOS CNOP optimization")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", type=str, default="test", choices=("train", "val", "test"))
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output-dir", type=str, default="outputs/cnop_climatology_driven")
    parser.add_argument("--source-name", type=str, default="", help="Only run one source; empty means all sources.")
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.08)
    parser.add_argument("--num-starts", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--random-init-scale", type=float, default=0.02)
    parser.add_argument("--lbfgs-steps", type=int, default=0)
    parser.add_argument("--lbfgs-lr", type=float, default=0.5)
    parser.add_argument("--epsilon-tos", type=float, default=0.1)
    parser.add_argument("--epsilon-zos", type=float, default=0.1)
    parser.add_argument(
        "--constraint-mode",
        type=str,
        default="event_l2",
        choices=("normalized_rms", "relative_initial_l2", "event_l2"),
    )
    parser.add_argument("--relative-l2-fraction", type=float, default=0.1)
    parser.add_argument("--constraint-file", type=str, default="outputs/cnop_constraint_0705/cnop_constraint_summary.json")
    parser.add_argument("--constraint-scale", type=float, default=0.4)
    parser.add_argument("--max-abs", type=float, default=2.0)
    parser.add_argument("--domain", type=str, default="tropical_pacific", choices=("tropical_pacific", "global"))
    parser.add_argument("--perturb-grid", type=str, default="patch", choices=("patch", "full"))
    parser.add_argument("--perturb-patch-size", type=int, default=4)
    parser.add_argument("--lat-bounds", type=str, default="-20,20")
    parser.add_argument("--lon-bounds", type=str, default="120,290")
    parser.add_argument("--objective-mode", type=str, default="lead_delta", choices=("softmax_3m", "lead_delta"))
    parser.add_argument("--objective-lead", type=int, default=12)
    parser.add_argument("--objective-temperature", type=float, default=0.25)
    parser.add_argument("--smoothness-weight", type=float, default=0.001)
    parser.set_defaults(amp=True, checkpoint_rollout=True)
    parser.add_argument("--amp", dest="amp", action="store_true")
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    parser.add_argument("--checkpoint-rollout", dest="checkpoint_rollout", action="store_true")
    parser.add_argument("--no-checkpoint-rollout", dest="checkpoint_rollout", action="store_false")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def source_monthly_field_climatology(dataset: WalkerDataset, source_idx: int) -> np.ndarray:
    """返回一个 source 的训练期 1-12 月四变量场气候态，shape 为 ``(12, 4, H, W)``。"""

    payload = dataset.source_payloads[source_idx]
    years = np.asarray(payload["years"])
    months = np.asarray(payload["months"])
    train_start, train_end = dataset.data_config["train_years"]
    fields: list[np.ndarray] = []
    for month in range(1, 13):
        mask = (years >= int(train_start)) & (years <= int(train_end)) & (months == month)
        if not np.any(mask):
            source = dataset.source_names[source_idx]
            raise ValueError(f"No training samples for source={source} month={month}")
        fields.append(np.nanmean(np.asarray(payload["data"][mask], dtype=np.float32), axis=0))
    return np.stack(fields, axis=0).astype(np.float32)


def build_climatology_inputs(dataset: WalkerDataset, source_indices: list[int], device: torch.device) -> dict[int, torch.Tensor]:
    """构造并归一化每个 source 的气候态输入窗口。"""

    inputs: dict[int, torch.Tensor] = {}
    for source_idx in source_indices:
        raw = source_monthly_field_climatology(dataset, source_idx)
        x = torch.from_numpy(raw).to(device=device, dtype=torch.float32).unsqueeze(0)
        x = dataset._normalize_tensor(x)  # noqa: SLF001 - script intentionally mirrors Dataset preprocessing
        inputs[source_idx] = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return inputs


def make_climatology_cases(dataset: WalkerDataset, source_name: str = "") -> list[cnop.NeutralCase]:
    """为每个 source 创建一个 synthetic case。target_year=0 表示气候态背景。"""

    if source_name:
        if source_name not in dataset.source_names:
            raise ValueError(f"Unknown source {source_name!r}; available={list(dataset.source_names)}")
        source_indices = [list(dataset.source_names).index(source_name)]
    else:
        source_indices = list(range(len(dataset.source_names)))
    return [
        cnop.NeutralCase(
            source_idx=source_idx,
            source_name=dataset.source_names[source_idx],
            target_t=0,
            target_year=0,
            neutral_score=0.0,
            observed_max_3m_abs=0.0,
        )
        for source_idx in source_indices
    ]


def write_climatology_method_json(
    output_dir: Path,
    args: argparse.Namespace,
    checkpoint: dict[str, Any],
    cases: list[cnop.NeutralCase],
) -> None:
    payload = {
        "experiment": "climatology_driven_cnop",
        "definition": "Each source input is its own training-period monthly climatology for Jan-Dec.",
        "config": args.config,
        "checkpoint": args.checkpoint,
        "checkpoint_epoch": checkpoint.get("epoch"),
        "split": args.split,
        "horizon": args.horizon,
        "steps": args.steps,
        "lr": args.lr,
        "num_starts": args.num_starts,
        "top_k": args.top_k,
        "random_init_scale": args.random_init_scale,
        "constraint_mode": args.constraint_mode,
        "constraint_file": args.constraint_file or None,
        "constraint_scale": args.constraint_scale,
        "event_constraint_l2": getattr(args, "event_constraint_l2", None),
        "domain": args.domain,
        "lat_bounds": args.lat_bounds,
        "lon_bounds": args.lon_bounds,
        "objective_mode": args.objective_mode,
        "objective_lead": args.objective_lead,
        "selected_cases": [
            {
                "source": case.source_name,
                "target_year": "climatology",
                "target_t": case.target_t,
                "observed_max_3m_abs": case.observed_max_3m_abs,
            }
            for case in cases
        ],
    }
    (output_dir / "method.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(args.config)
    dataset = WalkerDataset(config["data"]["path"], config, split=args.split)
    cnop.prepare_event_l2_constraint(dataset, args)
    model, checkpoint = cnop.load_model(config, args.checkpoint, device)
    trained_rollout_steps = int(config.get("training", {}).get("rollout_steps", args.horizon))
    climatology_np = cnop.compute_source_nino34_climatology(dataset)
    climatology = torch.from_numpy(climatology_np).to(device=device, dtype=torch.float32)

    cases = make_climatology_cases(dataset, args.source_name)
    climate_inputs = build_climatology_inputs(dataset, [case.source_idx for case in cases], device)

    def _make_case_input(dataset_: WalkerDataset, case: cnop.NeutralCase, device_: torch.device) -> torch.Tensor:
        del dataset_, device_
        return climate_inputs[case.source_idx].clone()

    def _target_month_tensor(dataset_: WalkerDataset, case: cnop.NeutralCase, step: int, device_: torch.device) -> torch.Tensor:
        del dataset_, case
        month = step % 12 + 1
        return torch.tensor([month], dtype=torch.long, device=device_)

    # 复用原优化器，但把真实年份输入替换为 source-wise climatology 输入。
    cnop.make_case_input = _make_case_input
    cnop.target_month_tensor = _target_month_tensor

    write_climatology_method_json(output_dir, args, checkpoint, cases)
    print(
        f"checkpoint_epoch={checkpoint.get('epoch')} climatology_cases={len(cases)} "
        f"horizon={args.horizon} steps={args.steps} device={device}",
        flush=True,
    )
    for case in cases:
        print(f"selected climatology case source={case.source_name}", flush=True)

    results: list[dict[str, Any]] = []
    for idx, case in enumerate(cases, start=1):
        print(f"optimize climatology case {idx}/{len(cases)} {case.source_name}", flush=True)
        result = cnop.optimize_case(model, dataset, case, climatology, args, device, trained_rollout_steps)
        cnop.write_case_npz(output_dir, result, dataset)
        (output_dir / f"case_{case.source_name}_climatology_history.json").write_text(
            json.dumps(result["history"], indent=2),
            encoding="utf-8",
        )
        (output_dir / f"case_{case.source_name}_climatology_candidates.json").write_text(
            json.dumps([cnop.serializable_candidate(item) for item in result["top_candidates"]], indent=2),
            encoding="utf-8",
        )
        print(
            f"case {case.source_name} climatology: "
            f"baseline_max_3m={result['baseline_max_3m']:.4f} "
            f"cnop_max_3m={result['cnop_max_3m']:.4f} "
            f"gain={result['gain_max_3m']:.4f} "
            f"best_start={result['best_start_idx']}",
            flush=True,
        )
        results.append(result)

    summary_path = cnop.write_summary_csv(output_dir, results)
    candidate_summary_path = cnop.write_candidate_summary_csv(output_dir, results)
    cnop.maybe_plot_results(output_dir, results, dataset)
    print(f"wrote {summary_path}", flush=True)
    print(f"wrote {candidate_summary_path}", flush=True)
    print(f"wrote figures to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
