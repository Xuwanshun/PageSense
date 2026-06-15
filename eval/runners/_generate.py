"""Live prediction generation (calls OpenAI). Shared by the live runners.

Only ever invoked after a cloud-cost guardrail has passed. Produces a list of
``Prediction`` objects: answer, ordered sources, captured evidence text (pulled
from the local vector store by chunk id — no extra embedding calls), and
wall-clock latency.
"""

from __future__ import annotations

import json

from config import Settings
from eval._common import GoldQuestion, Prediction, PredSource, timed
from rag.qa import answer_question_from_frozen_artifacts


def _chunk_text_lookup(settings: Settings) -> dict[str, str]:
    """Map chunk_id -> text from the default JSON vector store, if present.
    Lets the judge see real evidence without re-embedding. Best effort."""
    store_path = settings.vectorstore_dir / "store.json"
    if not store_path.exists():
        return {}
    try:
        payload = json.loads(store_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return {row["chunk_id"]: row.get("text", "") for row in payload.get("rows", [])}


def generate_predictions(
    questions: list[GoldQuestion],
    *,
    settings: Settings,
    top_k: int,
) -> list[Prediction]:
    text_lookup = _chunk_text_lookup(settings)
    predictions: list[Prediction] = []
    for q in questions:
        response, elapsed_ms = timed(
            answer_question_from_frozen_artifacts,
            q.question,
            settings=settings,
            top_k=top_k,
        )
        sources = [
            PredSource(
                chunk_id=s.get("chunk_id"),
                source_filename=s.get("source_filename"),
                page_number=s.get("page_number"),
                document_id=s.get("document_id"),
                score=s.get("score", 0.0),
            )
            for s in response.sources
        ]
        evidence = [text_lookup[s.chunk_id] for s in sources if s.chunk_id and s.chunk_id in text_lookup]
        predictions.append(
            Prediction(
                id=q.id,
                question=q.question,
                answer=response.answer,
                sources=sources,
                evidence=evidence,
                latency_ms=round(elapsed_ms, 2),
            )
        )
    return predictions
