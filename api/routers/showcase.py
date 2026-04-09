"""
Showcase endpoints — power the main web UI.

GET  /api/showcase-data  → corpus stats, eval metrics, example questions
POST /api/ask            → run QA pipeline and return answer + sources
"""

from __future__ import annotations

import json
import logging
import time
from collections import Counter
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from rag.qa import answer_question_from_frozen_artifacts

logger = logging.getLogger(__name__)
router = APIRouter(tags=["showcase"])

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES_PATH = PROJECT_ROOT / "api" / "static" / "examples.json"

DEFAULT_EXAMPLE_QUESTIONS = [
    "What are the four AI RMF Core functions?",
    "What goal does the AI RMF have according to the executive summary?",
    "What does the MAP function cover in the AI RMF core?",
    "What are the GOVERN categories and subcategories?",
    "How are stakeholder expectations connected to technical requirements in the NASA Systems Engineering Handbook?",
    "How does the NASA systems engineering handbook distinguish technical and programmatic risk?",
]


def _document_blurb(source_filename: str) -> tuple[str, str]:
    filename = (source_filename or "").lower()
    if "nist" in filename and "ai_rmf" in filename:
        return (
            "NIST AI RMF 1.0",
            "A voluntary framework for managing AI risk with the core functions: GOVERN, MAP, MEASURE, MANAGE.",
        )
    if "nasa" in filename and "systems_engineering" in filename:
        return (
            "NASA Systems Engineering Handbook (Rev. 2)",
            "Systems engineering processes, requirements flows, and risk concepts used in NASA programs.",
        )
    stem = Path(source_filename).stem if source_filename else "PDF document"
    return (stem.replace("_", " ").strip() or "PDF document", "A document in the current corpus.")


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _latest_eval_run() -> Path | None:
    results_dir = PROJECT_ROOT / "evaluation" / "results"
    if not results_dir.exists():
        return None
    candidates = [p for p in results_dir.iterdir() if p.is_dir() and (p / "summary.json").exists()]
    return sorted(candidates, key=lambda p: p.name)[-1] if candidates else None


def _corpus_stats(settings) -> tuple[list[dict], int, dict[str, int]]:
    docs: list[dict] = []
    total_chunks = 0
    region_counter: Counter[str] = Counter()

    processed_dir = settings.processed_documents_dir
    if not processed_dir.exists():
        return docs, total_chunks, {}

    for doc_dir in sorted(p for p in processed_dir.iterdir() if p.is_dir()):
        document = _load_json(doc_dir / "document.json") if (doc_dir / "document.json").exists() else {}
        chunks = _load_json(doc_dir / "chunks.json") if (doc_dir / "chunks.json").exists() else []
        visuals = _load_json(doc_dir / "visual_summaries.json") if (doc_dir / "visual_summaries.json").exists() else []

        source_filename = str(document.get("source_filename") or doc_dir.name)
        docs.append({"document_id": doc_dir.name, "source_filename": source_filename, "chunk_count": len(chunks)})
        total_chunks += len(chunks)
        for item in visuals:
            if isinstance(item, dict):
                region_counter[str(item.get("region_type", "unknown"))] += 1

    return docs, total_chunks, dict(sorted(region_counter.items()))


def _corpus_previews(settings) -> list[dict]:
    previews: list[dict] = []
    processed_dir = settings.processed_documents_dir

    doc_dirs = sorted(p for p in processed_dir.iterdir() if p.is_dir()) if processed_dir.exists() else []
    if doc_dirs:
        for doc_dir in doc_dirs:
            document = _load_json(doc_dir / "document.json") if (doc_dir / "document.json").exists() else {}
            chunks = _load_json(doc_dir / "chunks.json") if (doc_dir / "chunks.json").exists() else []
            source_filename = str(document.get("source_filename") or doc_dir.name)
            title, description = _document_blurb(source_filename)

            page_count = int(document.get("page_count") or 0)
            chunk_count = len(chunks) if isinstance(chunks, list) else 0
            facts = f"{page_count} pages · {chunk_count} chunks" if page_count or chunk_count else ""

            previews.append(
                {
                    "document_id": doc_dir.name,
                    "source_filename": source_filename,
                    "title": title,
                    "description": description,
                    "facts": facts,
                }
            )
        return previews

    # Fallback: show raw PDFs if preprocessing hasn't run yet
    raw_dir = PROJECT_ROOT / "Data" / "Raw"
    for pdf_path in sorted(raw_dir.glob("*.pdf")) if raw_dir.exists() else []:
        title, description = _document_blurb(pdf_path.name)
        previews.append(
            {
                "document_id": None,
                "source_filename": pdf_path.name,
                "title": title,
                "description": description,
                "facts": "",
            }
        )
    return previews


def _collect_example_questions(predictions: list, ocr_examples: list) -> list[str]:
    seen: set[str] = set()
    questions: list[str] = []

    def add(candidate) -> None:
        q = str(candidate or "").strip()
        if not q or q.lower() in seen:
            return
        seen.add(q.lower())
        questions.append(q)

    for item in predictions:
        if isinstance(item, dict):
            add(item.get("question"))
        if len(questions) >= 6:
            break

    if len(questions) < 6:
        for item in ocr_examples:
            if isinstance(item, dict):
                add(item.get("query"))
            if len(questions) >= 6:
                break

    for q in DEFAULT_EXAMPLE_QUESTIONS:
        add(q)
        if len(questions) >= 6:
            break

    return questions


@router.get("/api/showcase-data")
async def showcase_data(request: Request) -> JSONResponse:
    settings = request.app.state.settings
    latest_run = _latest_eval_run()
    summary = _load_json(latest_run / "summary.json") if latest_run else None
    llm_judge = (
        _load_json(latest_run / "llm_judge.json") if latest_run and (latest_run / "llm_judge.json").exists() else None
    )
    predictions = _load_json(latest_run / "predictions.json") if latest_run else []

    docs, chunk_count, region_counts = _corpus_stats(settings)
    corpus_previews = _corpus_previews(settings)
    examples = _load_json(EXAMPLES_PATH) if EXAMPLES_PATH.exists() else {"ocr_impact": []}
    example_questions = _collect_example_questions(
        predictions=predictions or [],
        ocr_examples=examples.get("ocr_impact", []),
    )

    return JSONResponse(
        {
            "latest_run": latest_run.name if latest_run else None,
            "summary": summary,
            "judge": llm_judge,
            "prediction_examples": (predictions or [])[:3],
            "ocr_impact": examples.get("ocr_impact", []),
            "example_questions": example_questions,
            "corpus_previews": corpus_previews,
            "stats": {
                "documents": docs,
                "chunks": chunk_count,
                "region_counts": region_counts,
            },
        }
    )


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: int = Field(default=4, ge=1, le=10)


@router.post("/api/ask")
async def ask(request: Request, body: AskRequest) -> JSONResponse:
    settings = request.app.state.settings

    if not settings.openai_api_key:
        raise HTTPException(status_code=422, detail="OPENAI_API_KEY is not set.")

    started = time.time()
    try:
        response = answer_question_from_frozen_artifacts(
            body.question,
            settings=settings,
            top_k=body.top_k,
        )
    except Exception as exc:
        logger.exception("Ask failed: %r", body.question)
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
