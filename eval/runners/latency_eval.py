"""Latency + rough cost evaluation.

Offline by default: aggregates latency/cost from a recorded predictions.json
(uses the latency captured during the live run that produced it). With --live
(+ --confirm-cloud-cost) it re-runs the questions to measure fresh wall-clock.

Cost is approximate — see eval/metrics/ops.py.

    python -m eval.runners.latency_eval --predictions eval/datasets/smoke_predictions.json
    python -m eval.runners.latency_eval --gold eval/datasets/manual_questions.yaml \
        --live --confirm-cloud-cost
"""

from __future__ import annotations

import argparse
from pathlib import Path

from config import Settings
from eval._common import (
    EVAL_ROOT,
    PredictionFile,
    load_config,
    load_gold,
    load_predictions,
    new_report_dir,
    require_cloud_confirmation,
    utc_stamp,
    write_json,
)
from eval.metrics import ops


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gold", type=Path, default=None, help="Required for --live.")
    p.add_argument("--predictions", type=Path, default=None, help="Recorded predictions to aggregate (offline).")
    p.add_argument("--config", type=Path, default=EVAL_ROOT / "configs" / "default.yaml")
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--live", action="store_true")
    p.add_argument("--confirm-cloud-cost", action="store_true")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = load_config(args.config)

    if args.live:
        if not args.gold:
            raise SystemExit("--live needs --gold to know which questions to time.")
        gold = load_gold(args.gold)
        require_cloud_confirmation(
            live=True,
            confirm_cloud_cost=args.confirm_cloud_cost,
            n_calls=len(gold.questions),
            what="latency_eval --live",
        )
        from eval.runners._generate import generate_predictions

        preds_list = generate_predictions(gold.questions, settings=Settings(), top_k=cfg.top_k)
        pred_file = PredictionFile(run={"timestamp": utc_stamp(), "top_k": cfg.top_k}, predictions=preds_list)
    else:
        if not args.predictions:
            raise SystemExit("Offline mode needs --predictions <file> (or pass --live to re-time).")
        pred_file = load_predictions(args.predictions)

    report = ops.aggregate(pred_file.predictions, usd_per_1k_tokens=cfg.usd_per_1k_tokens)
    out_dir = args.out_dir or new_report_dir("latency")
    write_json(out_dir / "ops_metrics.json", report)
    print(f"[latency_eval] {'live' if args.live else 'offline'} → {out_dir}")
    print("  ops:", report)


if __name__ == "__main__":
    main()
