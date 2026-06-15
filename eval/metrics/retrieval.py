"""Retrieval-quality metrics: Recall@k, MRR, nDCG@k (graded).

All offline. Operates on recorded predictions vs the gold set. A predicted
source is "relevant" when it matches any gold source (file name + page window);
nDCG uses the matched gold's graded relevance (0-3).
"""

from __future__ import annotations

import math
from statistics import mean

from eval._common import GoldQuestion, Prediction, source_matches


def _dcg(relevances: list[int]) -> float:
    return sum(rel / math.log2(rank + 2) for rank, rel in enumerate(relevances))


def _ndcg_at_k(pred_relevances: list[int], ideal_relevances: list[int], k: int) -> float:
    idcg = _dcg(sorted(ideal_relevances, reverse=True)[:k])
    if idcg == 0:
        return 0.0
    return _dcg(pred_relevances[:k]) / idcg


def score_question(
    question: GoldQuestion,
    prediction: Prediction,
    *,
    ks: list[int],
    page_window: int,
) -> dict:
    golds = question.expected.sources
    ranked = prediction.sources  # already ordered best-first by the pipeline

    # Graded relevance of each predicted source, in rank order. Each gold source
    # is credited only ONCE: when several retrieved chunks map to the same gold
    # page, only the first (highest-ranked) earns the gain — otherwise DCG can
    # exceed the ideal and nDCG rises above 1.0.
    consumed: set[int] = set()
    pred_rels: list[int] = []
    for p in ranked:
        best_rel, best_idx = 0, None
        for gi, g in enumerate(golds):
            if gi in consumed:
                continue
            if source_matches(p, g, page_window=page_window) and g.relevance > best_rel:
                best_rel, best_idx = g.relevance, gi
        if best_idx is not None:
            consumed.add(best_idx)
        pred_rels.append(best_rel)

    # Recall@k: distinct gold sources hit within the top-k predictions.
    recall_at_k: dict[str, float] = {}
    for k in ks:
        topk = ranked[:k]
        hit = sum(1 for g in golds if any(source_matches(p, g, page_window=page_window) for p in topk))
        recall_at_k[str(k)] = (hit / len(golds)) if golds else 0.0

    # MRR: reciprocal rank of the first relevant prediction.
    rr = 0.0
    for rank, rel in enumerate(pred_rels, start=1):
        if rel > 0:
            rr = 1.0 / rank
            break

    ideal = [g.relevance for g in golds]
    ndcg_at_k = {str(k): _ndcg_at_k(pred_rels, ideal, k) for k in ks}

    return {
        "id": question.id,
        "reciprocal_rank": rr,
        "recall_at_k": recall_at_k,
        "ndcg_at_k": ndcg_at_k,
        "num_gold": len(golds),
    }


def aggregate(
    questions: list[GoldQuestion],
    predictions: dict[str, Prediction],
    *,
    ks: list[int],
    page_window: int,
) -> dict:
    # Only score questions that actually have gold sources (no-answer questions
    # have none and would otherwise drag recall to zero).
    per_question = [
        score_question(q, predictions[q.id], ks=ks, page_window=page_window)
        for q in questions
        if q.expected.sources and q.id in predictions
    ]
    if not per_question:
        return {"aggregate": {}, "per_question": []}

    agg = {
        "mrr": mean(r["reciprocal_rank"] for r in per_question),
        "recall_at_k": {str(k): mean(r["recall_at_k"][str(k)] for r in per_question) for k in ks},
        "ndcg_at_k": {str(k): mean(r["ndcg_at_k"][str(k)] for r in per_question) for k in ks},
        "scored_questions": len(per_question),
        "page_window": page_window,
    }
    return {"aggregate": agg, "per_question": per_question}
