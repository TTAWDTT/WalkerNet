"""根据 rollout 评测结果选择 checkpoint。

这个脚本不重新跑模型，只读取 ``evaluate_rollout.py`` 写出的
``*_rollout_metrics.json``，用固定权重计算一个综合分数。它的目标是把
checkpoint selection 从单纯 ``val_loss`` 推向真正关心的自由滚动 skill。
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any


DEFAULT_LEAD_WEIGHTS = {
    1: 1.0,
    3: 1.0,
    6: 1.2,
    9: 1.4,
    12: 1.6,
    18: 1.2,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select WalkerNet checkpoint by rollout skill")
    parser.add_argument(
        "metrics",
        nargs="+",
        help="一个或多个 evaluate_rollout.py 输出的 *_rollout_metrics.json",
    )
    parser.add_argument("--output", type=str, default="outputs/rollout_checkpoint_selection.json")
    parser.add_argument("--copy-best-to", type=str, default=None, help="可选：把最佳 checkpoint 复制到该路径")
    parser.add_argument("--rmse-weight", type=float, default=1.0)
    parser.add_argument("--acc-weight", type=float, default=0.5)
    parser.add_argument("--three-month-weight", type=float, default=0.5)
    return parser.parse_args()


def _metric_row(payload: dict[str, Any], mode: str, system: str, lead: int) -> dict[str, float]:
    return payload["nino34_anomaly"][mode][system][str(lead)]


def _score_payload(
    payload: dict[str, Any],
    rmse_weight: float,
    acc_weight: float,
    three_month_weight: float,
) -> dict[str, Any]:
    """计算越小越好的 rollout selection score。

    RMSE 用相对 persistence 的 skill 表示，ACC 用 model-persistence 差值表示。
    这样不同 lead 的量纲可比较，也能明确奖励“真的赢 persistence”的 checkpoint。
    """
    leads = [int(lead) for lead in payload["leads"]]
    monthly_score = 0.0
    three_month_score = 0.0
    weight_sum = 0.0
    three_weight_sum = 0.0
    lead_details: dict[str, Any] = {}

    for lead in leads:
        weight = float(DEFAULT_LEAD_WEIGHTS.get(lead, 1.0))
        model = _metric_row(payload, "monthly", "model", lead)
        persistence = _metric_row(payload, "monthly", "persistence", lead)
        rmse_skill = (persistence["rmse"] - model["rmse"]) / max(persistence["rmse"], 1e-12)
        acc_gain = model["corr"] - persistence["corr"]
        lead_score = -rmse_weight * rmse_skill - acc_weight * acc_gain
        monthly_score += weight * lead_score
        weight_sum += weight
        lead_details[str(lead)] = {
            "monthly_rmse_skill": rmse_skill,
            "monthly_acc_gain": acc_gain,
            "monthly_score": lead_score,
        }

        three_month_model = payload["nino34_anomaly"]["three_month_mean"]["model"].get(str(lead))
        three_month_persistence = payload["nino34_anomaly"]["three_month_mean"]["persistence"].get(str(lead))
        if three_month_model is not None and three_month_persistence is not None:
            three_rmse_skill = (
                three_month_persistence["rmse"] - three_month_model["rmse"]
            ) / max(three_month_persistence["rmse"], 1e-12)
            three_acc_gain = three_month_model["corr"] - three_month_persistence["corr"]
            three_score = -rmse_weight * three_rmse_skill - acc_weight * three_acc_gain
            three_month_score += weight * three_score
            three_weight_sum += weight
            lead_details[str(lead)].update(
                {
                    "three_month_rmse_skill": three_rmse_skill,
                    "three_month_acc_gain": three_acc_gain,
                    "three_month_score": three_score,
                }
            )

    monthly_score = monthly_score / max(weight_sum, 1e-12)
    if three_weight_sum > 0:
        three_month_score = three_month_score / three_weight_sum
    total_score = monthly_score + three_month_weight * three_month_score
    return {
        "score": total_score,
        "monthly_score": monthly_score,
        "three_month_score": three_month_score,
        "lead_details": lead_details,
    }


def main() -> None:
    args = parse_args()
    rows = []
    for metrics_path in args.metrics:
        path = Path(metrics_path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        score = _score_payload(
            payload,
            rmse_weight=args.rmse_weight,
            acc_weight=args.acc_weight,
            three_month_weight=args.three_month_weight,
        )
        rows.append(
            {
                "metrics_path": str(path),
                "checkpoint": payload.get("checkpoint"),
                "checkpoint_epoch": payload.get("checkpoint_epoch"),
                **score,
            }
        )

    rows.sort(key=lambda row: row["score"])
    result = {
        "best": rows[0] if rows else None,
        "candidates": rows,
        "score_note": "score 越小越好；相对 persistence 的 RMSE skill 和 ACC gain 越高越好",
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"wrote {output_path}")

    if args.copy_best_to and result["best"] is not None:
        checkpoint = result["best"].get("checkpoint")
        if not checkpoint:
            raise ValueError("Best metrics file does not contain checkpoint path")
        target = Path(args.copy_best_to)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(checkpoint, target)
        print(f"copied best checkpoint to {target}")


if __name__ == "__main__":
    main()
