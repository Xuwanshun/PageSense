"""LLM-as-judge answer scoring. LIVE ONLY — this module makes OpenAI calls.

Never imported by the offline default path. Runners call it only after
``require_cloud_confirmation`` passes. The judge defaults to a stronger model
than the generator (configured in eval/configs/default.yaml) to reduce
self-preference bias.
"""

from __future__ import annotations

from statistics import mean

from pydantic import BaseModel, Field

from config import Settings
from document_process.clients import build_openai_client
from eval._common import GoldQuestion, Prediction


class JudgeOutput(BaseModel):
    answer_correctness: int = Field(default=3, ge=1, le=5)
    faithfulness: int = Field(default=3, ge=1, le=5)
    completeness: int = Field(default=3, ge=1, le=5)
    hallucination_detected: bool = False
    # Only meaningful for suggestion-type questions; the judge returns null for
    # other types, so these are optional and excluded from aggregates when None.
    suggestion_actionable: int | None = None
    suggestion_supported_by_evidence: bool | None = None
    rationale: str = ""


_SYSTEM = (
    "You are a strict evaluator for a document-grounded QA system over technical "
    "operation manuals. Score only from the provided evidence and reference answer. "
    "Return a JSON object with integer scores 1-5 for answer_correctness, "
    "faithfulness, completeness; booleans hallucination_detected and "
    "suggestion_supported_by_evidence; integer 1-5 suggestion_actionable; and a "
    "short rationale. For non-suggestion questions, leave the suggestion_* fields "
    "at neutral values."
)


def judge_one(
    question: GoldQuestion,
    prediction: Prediction,
    *,
    settings: Settings,
    judge_model: str,
) -> JudgeOutput:
    judge_settings = settings.model_copy(update={"openai_model": judge_model})
    client = build_openai_client(judge_settings)
    evidence = "\n\n".join(prediction.evidence) if prediction.evidence else "(no evidence recorded)"
    user = (
        f"Question type: {question.type}\n"
        f"Question: {question.question}\n\n"
        f"Reference answer: {question.expected.reference_answer or '(none provided)'}\n\n"
        f"Predicted answer: {prediction.answer}\n\n"
        f"Retrieved evidence:\n{evidence}"
    )
    return client.generate_structured(
        system_prompt=_SYSTEM,
        user_prompt=user,
        response_model=JudgeOutput,
    )


def aggregate(rows: list[tuple[GoldQuestion, JudgeOutput]]) -> dict:
    if not rows:
        return {"aggregate": {}, "per_question": []}
    outs = [o for _, o in rows]
    suggestions = [o for q, o in rows if q.type == "suggestion"]
    actionable = [o.suggestion_actionable for o in suggestions if o.suggestion_actionable is not None]
    supported = [o.suggestion_supported_by_evidence for o in suggestions if o.suggestion_supported_by_evidence is not None]
    agg = {
        "answer_correctness_mean": mean(o.answer_correctness for o in outs),
        "faithfulness_mean": mean(o.faithfulness for o in outs),
        "completeness_mean": mean(o.completeness for o in outs),
        "hallucination_rate": mean(1.0 if o.hallucination_detected else 0.0 for o in outs),
        "suggestion_actionable_mean": (mean(actionable) if actionable else None),
        "suggestion_supported_rate": (mean(1.0 if s else 0.0 for s in supported) if supported else None),
    }
    per_question = [{"id": q.id, **o.model_dump()} for q, o in rows]
    return {"aggregate": agg, "per_question": per_question}
