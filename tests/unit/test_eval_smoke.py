"""CI-safe smoke test for the eval framework.

Fully offline: loads the tiny smoke gold set + recorded predictions and checks
the pure-Python metrics produce the expected values. No network, no Paddle.
"""

from __future__ import annotations

from eval._common import EVAL_ROOT, load_gold, load_predictions
from eval.metrics import answer as answer_metrics
from eval.metrics import citation, ops, retrieval

GOLD = EVAL_ROOT / "datasets" / "smoke.yaml"
PREDS = EVAL_ROOT / "datasets" / "smoke_predictions.json"


def _load():
    gold = load_gold(GOLD)
    preds = load_predictions(PREDS)
    by_id = {p.id: p for p in preds.predictions}
    return gold, preds, by_id


def test_retrieval_metrics():
    gold, _, by_id = _load()
    report = retrieval.aggregate(gold.questions, by_id, ks=[1, 3, 5], page_window=1)
    agg = report["aggregate"]
    assert agg["mrr"] == 1.0
    assert agg["recall_at_k"]["1"] == 1.0
    assert agg["ndcg_at_k"]["1"] == 1.0
    assert agg["scored_questions"] == 2  # s3 (no-answer) excluded


def test_citation_metrics():
    gold, _, by_id = _load()
    report = citation.aggregate(gold.questions, by_id, top_n=2, page_window=1)
    agg = report["aggregate"]
    # s1 cites one matching + one non-matching source (0.5), s2 cites one match (1.0)
    assert agg["precision_tolerant"] == 0.75
    assert agg["recall_tolerant"] == 1.0


def test_answer_metrics():
    gold, _, by_id = _load()
    report = answer_metrics.aggregate(gold.questions, by_id)
    agg = report["aggregate"]
    assert agg["must_include_pass_rate"] == 1.0
    assert agg["must_not_violation_rate"] == 0.0
    assert agg["no_answer_accuracy"] == 1.0
    assert agg["false_refusal_rate"] == 0.0
    assert agg["suggestion_support_rate"] == 1.0


def test_ops_metrics():
    _, preds, _ = _load()
    report = ops.aggregate(preds.predictions, usd_per_1k_tokens=0.005)
    assert report["latency_ms"]["mean"] == 100.0
    assert report["latency_ms"]["max"] == 120.0
    assert report["estimated_cost_usd"] >= 0.0
