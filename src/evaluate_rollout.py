"""WalkerNet 自回归场预报评测。

该脚本用于对齐 ENSO 文献中的 lead skill 评测：
1. 先用模型逐月 rollout 预测未来场；
2. 再从预测 tos 场计算 Niño3.4 anomaly；
3. 输出 lead-1/3/6/9/12/18 的 ACC/RMSE，并与 persistence 对比。
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from .dataset import WalkerDataset
from .evaluate import (
    _build_comparison,
    _compute_nino34_numpy,
    _empty_stats,
    _finalize_stats,
    _maybe_plot_nino,
    _nino_summary,
    _update_stats,
)
from .interfaces import VARIABLES
from .metrics import compute_nino34
from .model import WalkerNet
from .utils import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="WalkerNet Rollout Evaluation")
    parser.add_argument("--config", type=str, default="configs/server_3090.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", type=str, default="test", choices=("train", "val", "test"))
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-lead", type=int, default=18)
    parser.add_argument("--leads", type=str, default="1,3,6,9,12,18")
    parser.add_argument("--output-dir", type=str, default="outputs/eval_rollout")
    parser.add_argument(
        "--source-names",
        type=str,
        default="",
        help="可选，逗号分隔的 source 名称；为空时评测全部 source。",
    )
    return parser.parse_args()


def _parse_leads(value: str, max_lead: int) -> list[int]:
    leads = sorted({int(x.strip()) for x in value.split(",") if x.strip()})
    if not leads:
        raise ValueError("At least one lead is required")
    if min(leads) < 1 or max(leads) > max_lead:
        raise ValueError(f"leads must be within [1, {max_lead}], got {leads}")
    return leads


def _parse_source_names(value: str) -> set[str] | None:
    names = {item.strip() for item in value.split(",") if item.strip()}
    return names or None


def _valid_subset_positions(
    dataset: WalkerDataset,
    max_lead: int,
    source_names: set[str] | None = None,
) -> list[int]:
    """只保留可以完整 rollout 到 max_lead 的 test 样本。"""
    if source_names is not None:
        unknown = source_names.difference(dataset.source_names)
        if unknown:
            raise ValueError(f"Unknown source_names={sorted(unknown)}, available={dataset.source_names}")

    positions: list[int] = []
    for pos, sample_index in enumerate(dataset.sample_indices):
        if np.ndim(sample_index) == 0:
            source_idx = 0
            target_t = int(sample_index)
        else:
            source_idx = int(sample_index[0])
            target_t = int(sample_index[1])
        if source_names is not None and dataset.source_names[source_idx] not in source_names:
            continue
        last_available_t = len(dataset.source_payloads[source_idx]["years"]) - 1
        if int(target_t) + max_lead - 1 <= last_available_t:
            positions.append(pos)
    return positions


def _target_months(dataset: WalkerDataset, source_indices: torch.Tensor, target_indices: torch.Tensor) -> torch.Tensor:
    """按 source 读取每个样本 target_t 对应的月份。"""
    source_np = source_indices.detach().cpu().numpy()
    target_np = target_indices.detach().cpu().numpy()
    months = [
        int(dataset.source_payloads[int(source_idx)]["months"][int(target_t)])
        for source_idx, target_t in zip(source_np, target_np)
    ]
    return torch.as_tensor(months, dtype=torch.long, device=target_indices.device)


def _target_tensors(
    dataset: WalkerDataset,
    source_indices: torch.Tensor,
    target_indices: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """读取未来目标场，返回 normalized 和 physical 两个版本。"""
    source_np = source_indices.detach().cpu().numpy()
    target_np = target_indices.detach().cpu().numpy()
    raw = np.stack(
        [
            np.asarray(dataset.source_payloads[int(source_idx)]["data"][int(target_t)], dtype=np.float32)
            for source_idx, target_t in zip(source_np, target_np)
        ],
        axis=0,
    )
    target_phys = torch.from_numpy(raw).float().to(device)[:, None]
    target_norm = dataset._normalize_tensor(  # noqa: SLF001 - 评测内部复用 Dataset 归一化
        target_phys,
        source_indices,
    )
    target_norm = torch.nan_to_num(target_norm, nan=0.0, posinf=0.0, neginf=0.0)
    return target_norm, target_phys


def _compute_source_nino34_climatology(dataset: WalkerDataset) -> torch.Tensor:
    """为每个 source 分别计算训练年份 Niño3.4 月气候态，shape=(S, 13)。"""
    data_config = dataset.data_config
    train_start, train_end = data_config["train_years"]
    climatology = np.zeros((len(dataset.source_payloads), 13), dtype=np.float32)

    for source_idx, payload in enumerate(dataset.source_payloads):
        years = payload["years"]
        months = payload["months"]
        train_mask = (years >= int(train_start)) & (years <= int(train_end))
        tos = np.asarray(payload["data"][:, 0])
        nino = _compute_nino34_numpy(tos, np.asarray(payload["lat"]), np.asarray(payload["lon"]))
        for month in range(1, 13):
            month_mask = train_mask & (months == month)
            climatology[source_idx, month] = float(np.nanmean(nino[month_mask]))
    return torch.from_numpy(climatology)


def _init_lead_stats(leads: list[int]) -> dict[int, dict[str, dict[str, torch.Tensor]]]:
    """为每个 lead 创建 model/persistence 的统计容器。"""
    return {
        lead: {
            "model_norm": _empty_stats(),
            "model_phys": _empty_stats(),
            "persistence_norm": _empty_stats(),
            "persistence_phys": _empty_stats(),
        }
        for lead in leads
    }


def _effective_lead(metrics: dict[int, dict[str, float]], threshold: float) -> int:
    """返回 ACC 不低于阈值的最大 lead。"""
    valid = [lead for lead, row in metrics.items() if row["corr"] >= threshold]
    return max(valid) if valid else 0


def _series_summary(
    model: dict[int, torch.Tensor],
    persistence: dict[int, torch.Tensor],
    target: dict[int, torch.Tensor],
    leads: list[int],
) -> dict[str, Any]:
    """计算 monthly 与 3-month mean Niño3.4 指标。"""
    monthly = {
        "model": {lead: _nino_summary(model[lead], target[lead]) for lead in leads},
        "persistence": {lead: _nino_summary(persistence[lead], target[lead]) for lead in leads},
    }

    leads_3m = [lead for lead in leads if lead >= 3 and (lead - 1) in model and (lead - 2) in model]
    three_month: dict[str, dict[int, dict[str, float]]] = {"model": {}, "persistence": {}}
    for lead in leads_3m:
        model_mean = (model[lead - 2] + model[lead - 1] + model[lead]) / 3.0
        persistence_mean = (persistence[lead - 2] + persistence[lead - 1] + persistence[lead]) / 3.0
        target_mean = (target[lead - 2] + target[lead - 1] + target[lead]) / 3.0
        three_month["model"][lead] = _nino_summary(model_mean, target_mean)
        three_month["persistence"][lead] = _nino_summary(persistence_mean, target_mean)

    return {
        "monthly": monthly,
        "three_month_mean": three_month,
        "effective_lead": {
            "monthly_acc_ge_0.5": _effective_lead(monthly["model"], 0.5),
            "monthly_acc_ge_0.6": _effective_lead(monthly["model"], 0.6),
            "three_month_acc_ge_0.5": _effective_lead(three_month["model"], 0.5),
            "three_month_acc_ge_0.6": _effective_lead(three_month["model"], 0.6),
        },
    }


def _jsonable_lead_dict(values: dict[int, Any]) -> dict[str, Any]:
    return {str(key): value for key, value in values.items()}


def _write_field_csv(
    path: Path,
    field_metrics: dict[int, dict[str, dict[str, Any]]],
    leads: list[int],
) -> None:
    """写出每个 lead 的场预测指标。"""
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["lead", "system", "variable", "space", "rmse", "mae", "corr", "count"])
        for lead in leads:
            for system in ("model", "persistence"):
                for space in ("normalized", "physical"):
                    for variable in VARIABLES:
                        row = field_metrics[lead][system][space][variable]
                        writer.writerow([
                            lead,
                            system,
                            variable,
                            space,
                            row["rmse"],
                            row["mae"],
                            row["corr"],
                            row["count"],
                        ])


def _write_nino_lead_csv(path: Path, nino_metrics: dict[str, Any], leads: list[int]) -> None:
    """写出 Niño3.4 lead skill 表。"""
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["mode", "lead", "system", "rmse", "mae", "corr", "num_samples"])
        for mode in ("monthly", "three_month_mean"):
            for system in ("model", "persistence"):
                for lead in leads:
                    row = nino_metrics[mode][system].get(lead)
                    if row is None:
                        continue
                    writer.writerow([mode, lead, system, row["rmse"], row["mae"], row["corr"], row["num_samples"]])


def _maybe_plot_lead_skill(path: Path, nino_metrics: dict[str, Any]) -> bool:
    """如果 matplotlib 可用，则画 lead ACC。"""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False

    plt.figure(figsize=(8, 4))
    for mode, linestyle in (("monthly", "-"), ("three_month_mean", "--")):
        for system, marker in (("model", "o"), ("persistence", "s")):
            rows = nino_metrics[mode][system]
            if not rows:
                continue
            leads = sorted(rows)
            acc = [rows[lead]["corr"] for lead in leads]
            plt.plot(leads, acc, linestyle=linestyle, marker=marker, label=f"{system} {mode}")
    plt.axhline(0.5, color="gray", linewidth=0.8, alpha=0.6)
    plt.axhline(0.6, color="gray", linewidth=0.8, alpha=0.6)
    plt.xlabel("lead month")
    plt.ylabel("Niño3.4 anomaly ACC")
    plt.ylim(-1.0, 1.0)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    return True


@torch.no_grad()
def evaluate_rollout(
    model: torch.nn.Module,
    loader: DataLoader,
    dataset: WalkerDataset,
    device: torch.device,
    max_lead: int,
    leads: list[int],
    trained_rollout_steps: int,
) -> dict[str, Any]:
    """执行自回归 rollout 评测。"""
    model.eval()
    trained_rollout_steps = max(1, int(trained_rollout_steps))
    lead_stats = _init_lead_stats(leads)

    # 为了计算 3-month mean，需要保留所有 1..max_lead 的 Niño3.4 anomaly。
    model_nino: dict[int, list[torch.Tensor]] = {lead: [] for lead in range(1, max_lead + 1)}
    persistence_nino: dict[int, list[torch.Tensor]] = {lead: [] for lead in range(1, max_lead + 1)}
    target_nino: dict[int, list[torch.Tensor]] = {lead: [] for lead in range(1, max_lead + 1)}

    lat = torch.as_tensor(dataset.lat, dtype=torch.float32, device=device)
    lon = torch.as_tensor(dataset.lon, dtype=torch.float32, device=device)
    climatology = _compute_source_nino34_climatology(dataset).to(device=device, dtype=torch.float32)

    for batch_idx, batch in enumerate(loader, start=1):
        window = batch["x"].to(device)
        valid_mask = batch["valid_mask"].to(device)
        source_index = batch.get("source_index")
        if source_index is None:
            source_index = torch.zeros(window.shape[0], dtype=torch.long)
        source_index = source_index.to(device=device, dtype=torch.long)
        persistence_norm = window[:, -1:].contiguous()
        persistence_phys = dataset.denormalize(persistence_norm, source_index)
        base_target_t = batch["time_index"].to(device=device, dtype=torch.long)

        for step in range(1, max_lead + 1):
            target_t = base_target_t + step - 1
            target_month = _target_months(dataset, source_index, target_t)
            # 训练时 rollout_step 从 0 开始计数；超过训练步数的远期评测复用最后一个已训练 step。
            rollout_step = torch.full(
                (window.shape[0],),
                min(step - 1, trained_rollout_steps - 1),
                dtype=torch.long,
                device=device,
            )
            pred_norm = model(window, target_month, rollout_step=rollout_step)
            pred_phys = dataset.denormalize(pred_norm, source_index)
            target_norm, target_phys = _target_tensors(dataset, source_index, target_t, device)

            if step in lead_stats:
                _update_stats(lead_stats[step]["model_norm"], pred_norm, target_norm, valid_mask[:, None])
                _update_stats(lead_stats[step]["model_phys"], pred_phys, target_phys, valid_mask[:, None])
                _update_stats(
                    lead_stats[step]["persistence_norm"],
                    persistence_norm,
                    target_norm,
                    valid_mask[:, None],
                )
                _update_stats(
                    lead_stats[step]["persistence_phys"],
                    persistence_phys,
                    target_phys,
                    valid_mask[:, None],
                )

            clim = climatology[source_index, target_month].detach().cpu()
            model_raw = compute_nino34(pred_phys[:, 0, 0], lat, lon).detach().cpu()
            persistence_raw = compute_nino34(persistence_phys[:, 0, 0], lat, lon).detach().cpu()
            target_raw = compute_nino34(target_phys[:, 0, 0], lat, lon).detach().cpu()
            model_nino[step].append(model_raw - clim)
            persistence_nino[step].append(persistence_raw - clim)
            target_nino[step].append(target_raw - clim)

            window = torch.cat([window[:, 1:], pred_norm], dim=1)

        if batch_idx % 5 == 0:
            print(f"rollout evaluated batches={batch_idx}/{len(loader)}", flush=True)

    nino_model = {lead: torch.cat(values) for lead, values in model_nino.items()}
    nino_persistence = {lead: torch.cat(values) for lead, values in persistence_nino.items()}
    nino_target = {lead: torch.cat(values) for lead, values in target_nino.items()}

    field_metrics: dict[int, dict[str, dict[str, Any]]] = {}
    for lead in leads:
        model_metrics = {
            "normalized": _finalize_stats(lead_stats[lead]["model_norm"]),
            "physical": _finalize_stats(lead_stats[lead]["model_phys"]),
        }
        persistence_metrics = {
            "normalized": _finalize_stats(lead_stats[lead]["persistence_norm"]),
            "physical": _finalize_stats(lead_stats[lead]["persistence_phys"]),
        }
        field_metrics[lead] = {
            "model": model_metrics,
            "persistence": persistence_metrics,
            "comparison": _build_comparison(
                {
                    **model_metrics,
                    "nino34_anomaly": _nino_summary(nino_model[lead], nino_target[lead]),
                },
                {
                    **persistence_metrics,
                    "nino34_anomaly": _nino_summary(nino_persistence[lead], nino_target[lead]),
                },
            ),
        }

    nino_metrics = _series_summary(nino_model, nino_persistence, nino_target, leads)
    return {
        "field": field_metrics,
        "nino34_anomaly": nino_metrics,
    }


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    leads = _parse_leads(args.leads, args.max_lead)
    source_names = _parse_source_names(args.source_names)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = WalkerDataset(config["data"]["path"], config, split=args.split)
    positions = _valid_subset_positions(dataset, args.max_lead, source_names=source_names)
    subset = Subset(dataset, positions)
    loader = DataLoader(
        subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model = WalkerNet(config).to(device)
    model.load_state_dict(checkpoint["model"])
    trained_rollout_steps = int(config.get("training", {}).get("rollout_steps", 1))

    print(f"checkpoint={args.checkpoint}")
    print(f"checkpoint_epoch={checkpoint.get('epoch')}")
    print(
        f"split={args.split} usable_samples={len(subset)} original_samples={len(dataset)} "
        f"source_names={sorted(source_names) if source_names else 'ALL'} "
        f"max_lead={args.max_lead} leads={leads} batches={len(loader)} "
        f"trained_rollout_steps={trained_rollout_steps} device={device}"
    )

    result = evaluate_rollout(model, loader, dataset, device, args.max_lead, leads, trained_rollout_steps)

    json_payload = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "split": args.split,
        "source_names": sorted(source_names) if source_names else None,
        "usable_samples": len(subset),
        "original_samples": len(dataset),
        "max_lead": args.max_lead,
        "leads": leads,
        "trained_rollout_steps": trained_rollout_steps,
        "rollout_step_policy": "min(step - 1, trained_rollout_steps - 1)",
        "field": _jsonable_lead_dict(result["field"]),
        "nino34_anomaly": {
            "monthly": {
                "model": _jsonable_lead_dict(result["nino34_anomaly"]["monthly"]["model"]),
                "persistence": _jsonable_lead_dict(result["nino34_anomaly"]["monthly"]["persistence"]),
            },
            "three_month_mean": {
                "model": _jsonable_lead_dict(result["nino34_anomaly"]["three_month_mean"]["model"]),
                "persistence": _jsonable_lead_dict(result["nino34_anomaly"]["three_month_mean"]["persistence"]),
            },
            "effective_lead": result["nino34_anomaly"]["effective_lead"],
        },
    }

    metrics_json = output_dir / f"{args.split}_rollout_metrics.json"
    field_csv = output_dir / f"{args.split}_rollout_field_metrics.csv"
    nino_csv = output_dir / f"{args.split}_rollout_nino34_lead_metrics.csv"
    lead_png = output_dir / f"{args.split}_rollout_nino34_lead_acc.png"

    metrics_json.write_text(json.dumps(json_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_field_csv(field_csv, result["field"], leads)
    _write_nino_lead_csv(nino_csv, result["nino34_anomaly"], leads)
    plotted = _maybe_plot_lead_skill(lead_png, result["nino34_anomaly"])

    print(json.dumps(json_payload["nino34_anomaly"], indent=2, ensure_ascii=False))
    print(f"wrote {metrics_json}")
    print(f"wrote {field_csv}")
    print(f"wrote {nino_csv}")
    if plotted:
        print(f"wrote {lead_png}")
    else:
        print("matplotlib unavailable; skipped lead skill png")


if __name__ == "__main__":
    main()
