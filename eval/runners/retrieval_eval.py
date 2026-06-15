"""Retrieval + citation evaluation.

Offline by default: scores a recorded predictions.json against the gold set.
With --live (+ --confirm-cloud-cost) it regenerates predictions first.

    python -m eval.runners.retrieval_eval --gold eval/datasets/smoke.yaml \
        --predictions eval/datasets/smoke_predictions.json
    python -m eval.runners.retrieval_eval --gold eval/datasets/manual_questions.yaml \
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
from eval.metrics import citation, retrieval


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gold", type=Path, required=True)
    p.add_argument("--predictions", type=Path, default=None, help="Recorded predictions to score (offline).")
    p.add_argument("--config", type=Path, default=EVAL_ROOT / "configs" / "default.yaml")
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--live", action="store_true", help="Regenerate predictions via the QA pipeline (calls OpenAI).")
    p.add_argument("--confirm-cloud-cost", action="store_true", help="Acknowledge OpenAI spend for --live.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = load_config(args.config)
    gold = load_gold(args.gold)

    if args.live:
        require_cloud_confirmation(
            live=True,
            confirm_cloud_cost=args.confirm_cloud_cost,
            n_calls=len(gold.questions),
            what="retrieval_eval --live",
        )
        from eval.runners._generate import generate_predictions

        preds_list = generate_predictions(gold.questions, settings=Settings(), top_k=cfg.top_k)
        pred_file = PredictionFile(run={"timestamp": utc_stamp(), "top_k": cfg.top_k}, predictions=preds_list)
    else:
        if not args.predictions:
            raise SystemExit("Offline mode needs --predictions <file> (or pass --live to regenerate).")
        pred_file = load_predictions(args.predictions)

    by_id = {p.id: p for p in pred_file.predictions}

    retrieval_report = retrieval.aggregate(gold.questions, by_id, ks=cfg.ks, page_window=cfg.page_window)
    citation_report = citation.aggregate(gold.questions, by_id, top_n=cfg.citation_top_n, page_window=cfg.page_window)

    out_dir = args.out_dir or new_report_dir("retrieval")
    write_json(out_dir / "predictions.json", pred_file.model_dump())
    write_json(out_dir / "retrieval_metrics.json", retrieval_report)
    write_json(out_dir / "citation_metrics.json", citation_report)

    print(f"[retrieval_eval] {'live' if args.live else 'offline'} → {out_dir}")
    print("  retrieval:", retrieval_report["aggregate"])
    print("  citation: ", citation_report["aggregate"])


if __name__ == "__main__":
    main()
