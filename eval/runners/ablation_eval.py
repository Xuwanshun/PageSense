"""Feature-flag ablation: measure what each retrieval/answer flag contributes.

Reads named variants from eval/configs/ablation.yaml, each setting a few
Settings flags. For every variant it regenerates predictions and scores
retrieval, so you can see which flags actually move the numbers (and which you
could drop to lighten the system).

Always calls OpenAI → requires --live --confirm-cloud-cost. Without --live it
does a dry-run that just prints the variants and the flags they set.

    python -m eval.runners.ablation_eval --gold eval/datasets/manual_questions.yaml   # dry-run
    python -m eval.runners.ablation_eval --gold eval/datasets/manual_questions.yaml \
        --live --confirm-cloud-cost
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from config import Settings
from eval._common import (
    EVAL_ROOT,
    load_config,
    load_gold,
    new_report_dir,
    require_cloud_confirmation,
    write_json,
)
from eval.metrics import retrieval


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gold", type=Path, required=True)
    p.add_argument("--config", type=Path, default=EVAL_ROOT / "configs" / "default.yaml")
    p.add_argument("--ablation", type=Path, default=EVAL_ROOT / "configs" / "ablation.yaml")
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--live", action="store_true")
    p.add_argument("--confirm-cloud-cost", action="store_true")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = load_config(args.config)
    gold = load_gold(args.gold)
    variants: dict[str, dict] = yaml.safe_load(args.ablation.read_text(encoding="utf-8"))["variants"]

    if not args.live:
        print("[ablation_eval] dry-run — variants and the flags they set:")
        for name, flags in variants.items():
            print(f"  {name}: {flags}")
        print("\nRe-run with --live --confirm-cloud-cost to score them.")
        return

    require_cloud_confirmation(
        live=True,
        confirm_cloud_cost=args.confirm_cloud_cost,
        n_calls=len(gold.questions) * len(variants),
        what="ablation_eval --live",
    )
    from eval.runners._generate import generate_predictions

    base = Settings()
    out_dir = args.out_dir or new_report_dir("ablation")
    comparison: dict[str, dict] = {}
    for name, flags in variants.items():
        settings = base.model_copy(update=flags)
        preds = generate_predictions(gold.questions, settings=settings, top_k=cfg.top_k)
        by_id = {p.id: p for p in preds}
        report = retrieval.aggregate(gold.questions, by_id, ks=cfg.ks, page_window=cfg.page_window)
        comparison[name] = report["aggregate"]
        print(f"  {name}: {report['aggregate']}")

    write_json(out_dir / "ablation_comparison.json", {"variants": variants, "results": comparison})
    print(f"[ablation_eval] → {out_dir}")


if __name__ == "__main__":
    main()
