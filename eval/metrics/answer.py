"""Deterministic, offline answer checks.

These need no model: substring must-include / must-not-include, and no-answer
(refusal) handling. The subjective dimensions — correctness, faithfulness,
completeness, suggestion actionability — live in judge.py and run only --live.
"""

from __future__ import annotations

from statistics import mean

from eval._common import GoldQuestion, Prediction

# Phrases the pipeline uses when it declines to answer. Kept in sync with the
# refusal strings in rag/qa.py and rag/retrieve.py.
_REFUSAL_MARKERS = (
    "cannot answer",
    "can't answer",
    "no relevant",
    "not in the provided context",
    "not in the context",
    "insufficient",
    "no relevant context",
    "i don't have",
    "i do not have",
    # Phrasings the synthesis agent actually uses when evidence is missing.
    "do not provide",
    "does not provide",
    "do not contain",
    "does not contain",
    "not mentioned",
    "not stated",
    "not provided",
    "not specified",
    "no information",
)


def looks_like_refusal(answer: str) -> bool:
    lowered = answer.lower()
    return any(marker in lowered for marker in _REFUSAL_MARKERS)


def score_question(question: GoldQuestion, prediction: Prediction) -> dict:
    answer = prediction.answer or ""
    lowered = answer.lower()
    exp = question.expected

    must_include_hits = [m for m in exp.must_include if m.lower() in lowered]
    must_include_ok = len(must_include_hits) == len(exp.must_include)
    must_not_violations = [m for m in exp.must_not_include if m.lower() in lowered]
    must_not_ok = not must_not_violations

    refused = looks_like_refusal(answer)
    if exp.no_answer:
        # Should have refused.
        no_answer_ok = refused
        false_refusal = False
    else:
        no_answer_ok = True  # not a no-answer question
        false_refusal = refused  # refused something it should have answered

    # Suggestion evidence support (offline proxy): the recommendation cites at
    # least one source AND satisfies its must_include evidence anchors.
    suggestion_supported: bool | None = None
    if exp.requires_citation:
        suggestion_supported = bool(prediction.sources) and must_include_ok

    return {
        "id": question.id,
        "type": question.type,
        "must_include_ok": must_include_ok,
        "must_include_hits": len(must_include_hits),
        "must_include_total": len(exp.must_include),
        "must_not_ok": must_not_ok,
        "must_not_violations": must_not_violations,
        "no_answer_ok": no_answer_ok,
        "false_refusal": false_refusal,
        "suggestion_supported": suggestion_supported,
    }


def aggregate(questions: list[GoldQuestion], predictions: dict[str, Prediction]) -> dict:
    scored = [score_question(q, predictions[q.id]) for q in questions if q.id in predictions]
    if not scored:
        return {"aggregate": {}, "per_question": []}

    no_answer_rows = [r for r in scored if any(q.id == r["id"] and q.expected.no_answer for q in questions)]
    answerable_rows = [r for r in scored if r not in no_answer_rows]
    suggestion_rows = [r for r in scored if r["suggestion_supported"] is not None]

    agg = {
        "must_include_pass_rate": mean(1.0 if r["must_include_ok"] else 0.0 for r in scored),
        "must_not_violation_rate": mean(0.0 if r["must_not_ok"] else 1.0 for r in scored),
        "no_answer_accuracy": (
            mean(1.0 if r["no_answer_ok"] else 0.0 for r in no_answer_rows) if no_answer_rows else None
        ),
        "false_refusal_rate": (
            mean(1.0 if r["false_refusal"] else 0.0 for r in answerable_rows) if answerable_rows else None
        ),
        "suggestion_support_rate": (
            mean(1.0 if r["suggestion_supported"] else 0.0 for r in suggestion_rows) if suggestion_rows else None
        ),
    }
    return {"aggregate": agg, "per_question": scored}
