"""Answer + citation-support evaluation.

Offline by default: deterministic must-include / must-not-include / no-answer
checks against recorded predictions. With --live (+ --confirm-cloud-cost) it
regenerates predictions and additionally runs the LLM-as-judge
(correctness / faithfulness / completeness / suggestion actionability).

    python -m eval.runners.answer_eval --gold eval/datasets/smoke.yaml \
        --predictions eval/datasets/smoke_predictions.json
    python -m eval.runners.answer_eval --gold eval/datasets/manual_questions.yaml \
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
from eval.metrics import answer as answer_metrics


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gold", type=Path, required=True)
    p.add_argument("--predictions", type=Path, default=None)
    p.add_argument("--config", type=Path, default=EVAL_ROOT / "configs" / "default.yaml")
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--live", action="store_true", help="Regenerate predictions + run the LLM judge (calls OpenAI).")
    p.add_argument("--confirm-cloud-cost", action="store_true")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = load_config(args.config)
    gold = load_gold(args.gold)

    if args.live:
        require_cloud_confirmation(
            live=True,
            confirm_cloud_cost=args.confirm_cloud_cost,
            # generation + judge ≈ 2 model touchpoints per question
            n_calls=len(gold.questions) * 2,
            what="answer_eval --live (judge)",
        )
        from eval.runners._generate import generate_predictions

        settings = Settings()
        preds_list = generate_predictions(gold.questions, settings=settings, top_k=cfg.top_k)
        pred_file = PredictionFile(run={"timestamp": utc_stamp(), "top_k": cfg.top_k}, predictions=preds_list)
    else:
        if not args.predictions:
            raise SystemExit("Offline mode needs --predictions <file> (or pass --live to regenerate).")
        pred_file = load_predictions(args.predictions)

    by_id = {p.id: p for p in pred_file.predictions}
    det_report = answer_metrics.aggregate(gold.questions, by_id)

    out_dir = args.out_dir or new_report_dir("answer")
    write_json(out_dir / "predictions.json", pred_file.model_dump())
    write_json(out_dir / "answer_metrics.json", det_report)
    print(f"[answer_eval] {'live' if args.live else 'offline'} → {out_dir}")
    print("  deterministic:", det_report["aggregate"])

    if args.live:
        from eval.metrics import judge as judge_metrics

        rows = []
        for q in gold.questions:
            if q.id not in by_id:
                continue
            out = judge_metrics.judge_one(q, by_id[q.id], settings=settings, judge_model=cfg.judge_model)
            rows.append((q, out))
        judge_report = judge_metrics.aggregate(rows)
        write_json(out_dir / "llm_judge.json", judge_report)
        print(f"  judge ({cfg.judge_model}):", judge_report["aggregate"])


if __name__ == "__main__":
    main()
