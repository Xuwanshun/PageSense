"""
Query endpoint — ask a question against the indexed corpus.
"""

from __future__ import annotations

import logging
import time

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from rag.qa import answer_question_from_frozen_artifacts

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/query", tags=["query"])


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, description="The question to ask.")
    top_k: int = Field(default=4, ge=1, le=20, description="Number of chunks to retrieve (1-20).")
    doc_filter: list[str] | None = Field(
        default=None, description="Limit search to these document IDs. Null searches all."
    )


@router.post("")
async def query(request: Request, body: QueryRequest) -> JSONResponse:
    """
    Ask a question against the indexed PDF corpus.

    The pipeline:
      1. Embed the question with OpenAI
      2. Retrieve the top-k most similar chunks from the vector store
      3. Rerank by semantic similarity + term overlap
      4. Route table/figure questions to specialist agents
      5. Synthesise a final grounded answer

    Request body (JSON):
        {"question": "What is the annual revenue?", "top_k": 4}

    Response:
        {
          "question": "...",
          "answer": "...",
          "sources": [{"chunk_id": "...", "page_number": 1, "score": 0.92, ...}],
          "router": {...},
          "specialists": [...],
          "latency_ms": 1234,
          "top_k": 4
        }
    """
    settings = request.app.state.settings

    if not settings.openai_api_key:
        raise HTTPException(
            status_code=422,
            detail="OPENAI_API_KEY is required for queries. Set it in your environment.",
        )

    logger.info("Query received: %r (top_k=%d)", body.question, body.top_k)
    started = time.time()
    try:
        response = answer_question_from_frozen_artifacts(
            body.question,
            settings=settings,
            top_k=body.top_k,
            doc_filter=body.doc_filter,
        )
    except Exception as exc:
        logger.exception("Query failed: %r", body.question)
        raise HTTPException(status_code=500, detail=f"Query failed: {exc}") from exc

    return JSONResponse(
        {
            "question": response.question,
            "answer": response.answer,
            "sources": response.sources,
            "router": response.router,
            "specialists": [
                {"agent_name": s.agent_name, "output": s.output, "region_ids": s.region_ids}
                for s in response.specialists
            ],
            "latency_ms": int((time.time() - started) * 1000),
            "top_k": body.top_k,
        }
    )
