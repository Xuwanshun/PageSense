"""Citation-quality metrics: precision/recall of the cited source set.

Distinct from retrieval.py: retrieval scores the full ranked list; citation
scores the top-N sources the answer actually surfaces, under both page-exact
(window 0) and page-tolerant (window N) matching. Section matching is applied
when both prediction and gold carry section labels, else it is ignored.
"""

from __future__ import annotations

from statistics import mean

from eval._common import GoldQuestion, Prediction, PredSource, source_matches


def _section_ok(pred: PredSource, gold_sections: list[str]) -> bool:
    if not gold_sections or not pred.sections:
        return True  # nothing to compare against — do not penalize
    gold_norm = {s.strip().lower() for s in gold_sections}
    return any(s.strip().lower() in gold_norm for s in pred.sections)


def _matched(pred: PredSource, question: GoldQuestion, *, page_window: int) -> bool:
    for g in question.expected.sources:
        if source_matches(pred, g, page_window=page_window) and _section_ok(pred, g.sections):
            return True
    return False


def score_question(
    question: GoldQuestion,
    prediction: Prediction,
    *,
    top_n: int,
    page_window: int,
) -> dict:
    cited = prediction.sources[:top_n]
    golds = question.expected.sources

    matched_preds = sum(1 for p in cited if _matched(p, question, page_window=page_window))
    matched_golds = sum(
        1
        for g in question.expected.sources
        if any(source_matches(p, g, page_window=page_window) and _section_ok(p, g.sections) for p in cited)
    )
    precision = (matched_preds / len(cited)) if cited else 0.0
    recall = (matched_golds / len(golds)) if golds else 0.0
    return {
        "id": question.id,
        "cited": len(cited),
        "precision": precision,
        "recall": recall,
    }


def aggregate(
    questions: list[GoldQuestion],
    predictions: dict[str, Prediction],
    *,
    top_n: int,
    page_window: int,
) -> dict:
    scored = [
        score_question(q, predictions[q.id], top_n=top_n, page_window=page_window)
        for q in questions
        if q.expected.sources and q.id in predictions
    ]
    if not scored:
        return {"aggregate": {}, "per_question": []}

    # Exact (page_window=0) variant for the aggregate, reusing the same helper.
    exact = [
        score_question(q, predictions[q.id], top_n=top_n, page_window=0)
        for q in questions
        if q.expected.sources and q.id in predictions
    ]
    agg = {
        "top_n": top_n,
        "page_window": page_window,
        "precision_tolerant": mean(r["precision"] for r in scored),
        "recall_tolerant": mean(r["recall"] for r in scored),
        "precision_exact": mean(r["precision"] for r in exact),
        "recall_exact": mean(r["recall"] for r in exact),
    }
    return {"aggregate": agg, "per_question": scored}
