"""
RAG evaluation metrics — no external eval frameworks needed.

All scoring uses LLM-as-judge via the existing OpenAI client, plus
exact-match retrieval metrics when ground-truth chunk IDs are provided.

Metric summary
--------------
Generation (no labels required):
  faithfulness        -- fraction of answer claims supported by retrieved context
  answer_relevance    -- does the answer address the question? (0-1)
  context_relevance   -- are retrieved chunks relevant to the question? (0-1)

Generation (requires ground_truth answer):
  context_recall      -- fraction of ground-truth statements covered by context
  answer_correctness  -- semantic overlap between generated and ground-truth answer

Retrieval (requires relevant_chunk_ids labels):
  hit_rate            -- was any labeled chunk retrieved? (binary per query)
  mrr                 -- reciprocal rank of first labeled chunk in result list
  precision_at_k      -- fraction of retrieved chunks that are labeled relevant
  recall_at_k         -- fraction of labeled chunks that were retrieved
"""

from __future__ import annotations

import json
import logging
from typing import Any

from document_process.clients import OpenAIJSONModelClient

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Retrieval metrics (exact match, no LLM needed)
# ---------------------------------------------------------------------------


def hit_rate(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    return 1.0 if any(cid in relevant_ids for cid in retrieved_ids) else 0.0


def mrr(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    for rank, cid in enumerate(retrieved_ids, start=1):
        if cid in relevant_ids:
            return 1.0 / rank
    return 0.0


def precision_at_k(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    if not retrieved_ids:
        return 0.0
    return sum(1 for cid in retrieved_ids if cid in relevant_ids) / len(retrieved_ids)


def recall_at_k(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    if not relevant_ids:
        return 0.0
    return sum(1 for cid in retrieved_ids if cid in relevant_ids) / len(relevant_ids)


# ---------------------------------------------------------------------------
# Generation metrics (LLM-as-judge)
# ---------------------------------------------------------------------------


def _score(client: OpenAIJSONModelClient, system: str, user: str, key: str) -> float:
    """Call the LLM and return a single float from the JSON response."""
    try:
        raw = client.client.chat.completions.create(
            model=client.model,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        payload = json.loads(raw.choices[0].message.content or "{}")
        return float(payload.get(key, 0.0))
    except Exception as exc:
        logger.warning("LLM scoring failed: %s", exc)
        return 0.0


def faithfulness(
    client: OpenAIJSONModelClient,
    answer: str,
    contexts: list[str],
) -> float:
    """Fraction of answer claims that are supported by the retrieved context."""
    context_block = "\n\n".join(f"[{i + 1}] {c}" for i, c in enumerate(contexts))
    return _score(
        client,
        system=(
            "You are an evaluation judge. Given an answer and source passages, "
            "identify every factual claim in the answer. "
            "Count how many claims are directly supported by the passages vs unsupported. "
            'Return JSON: {"supported": <int>, "total": <int>, "score": <float 0-1>}'
        ),
        user=f"Answer:\n{answer}\n\nSource passages:\n{context_block}",
        key="score",
    )


def answer_relevance(
    client: OpenAIJSONModelClient,
    question: str,
    answer: str,
) -> float:
    """How well does the answer address the question? (0 = off-topic, 1 = fully answers it)"""
    return _score(
        client,
        system=(
            "You are an evaluation judge. Score how well the answer addresses the question. "
            "0 = completely irrelevant or empty, 1 = fully and directly answers the question. "
            'Return JSON: {"score": <float 0-1>, "reason": "<one sentence>"}'
        ),
        user=f"Question: {question}\n\nAnswer: {answer}",
        key="score",
    )


def context_relevance(
    client: OpenAIJSONModelClient,
    question: str,
    contexts: list[str],
) -> float:
    """Are the retrieved passages relevant to the question? (0-1)"""
    context_block = "\n\n".join(f"[{i + 1}] {c[:400]}" for i, c in enumerate(contexts))
    return _score(
        client,
        system=(
            "You are an evaluation judge. Given a question and retrieved passages, "
            "score how relevant the passages are as a group for answering the question. "
            "0 = completely off-topic, 1 = all passages are highly relevant. "
            'Return JSON: {"score": <float 0-1>, "reason": "<one sentence>"}'
        ),
        user=f"Question: {question}\n\nRetrieved passages:\n{context_block}",
        key="score",
    )


def context_recall(
    client: OpenAIJSONModelClient,
    ground_truth: str,
    contexts: list[str],
) -> float:
    """Fraction of ground-truth statements that are covered by the retrieved context."""
    context_block = "\n\n".join(f"[{i + 1}] {c[:400]}" for i, c in enumerate(contexts))
    return _score(
        client,
        system=(
            "You are an evaluation judge. Break the ground-truth answer into individual statements. "
            "For each statement, check if the retrieved passages contain the information needed to derive it. "
            'Return JSON: {"covered": <int>, "total": <int>, "score": <float 0-1>}'
        ),
        user=f"Ground truth answer:\n{ground_truth}\n\nRetrieved passages:\n{context_block}",
        key="score",
    )


def answer_correctness(
    client: OpenAIJSONModelClient,
    ground_truth: str,
    answer: str,
) -> float:
    """Semantic overlap between generated answer and ground truth. (0-1)"""
    return _score(
        client,
        system=(
            "You are an evaluation judge. Compare the generated answer to the ground truth. "
            "Score semantic correctness: 1.0 = same meaning, 0.5 = partially correct, 0.0 = wrong or contradictory. "
            'Return JSON: {"score": <float 0-1>, "reason": "<one sentence>"}'
        ),
        user=f"Ground truth: {ground_truth}\n\nGenerated answer: {answer}",
        key="score",
    )


# ---------------------------------------------------------------------------
# Batch scorer
# ---------------------------------------------------------------------------


def score_sample(
    client: OpenAIJSONModelClient,
    *,
    question: str,
    answer: str,
    contexts: list[str],
    retrieved_ids: list[str],
    ground_truth: str | None = None,
    relevant_chunk_ids: list[str] | None = None,
) -> dict[str, Any]:
    """Compute all applicable metrics for one QA sample."""
    results: dict[str, Any] = {}

    results["faithfulness"] = faithfulness(client, answer, contexts)
    results["answer_relevance"] = answer_relevance(client, question, answer)
    results["context_relevance"] = context_relevance(client, question, contexts)

    if ground_truth:
        results["context_recall"] = context_recall(client, ground_truth, contexts)
        results["answer_correctness"] = answer_correctness(client, ground_truth, answer)

    if relevant_chunk_ids:
        rel = set(relevant_chunk_ids)
        results["hit_rate"] = hit_rate(retrieved_ids, rel)
        results["mrr"] = mrr(retrieved_ids, rel)
        results["precision_at_k"] = precision_at_k(retrieved_ids, rel)
        results["recall_at_k"] = recall_at_k(retrieved_ids, rel)

    return results
