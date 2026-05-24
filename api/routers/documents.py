"""
Document management endpoints.

PDF preprocessing (OCR + layout detection) is CPU-bound and can take
several minutes for large PDFs. It is offloaded to a thread pool executor
so the asyncio event loop stays free to answer ALB health checks during
processing. Without this, health checks time out after 90s and ECS kills
the task mid-preprocessing.
"""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
import threading
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse

from api.dependencies import get_current_user
from config import user_scoped_settings
from document_process.pipeline import preprocess_document
from rag.retrieve import index_all_processed_documents

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/documents", tags=["documents"])

# Only one OCR pipeline runs at a time — Paddle models are large and running
# two concurrent jobs would risk OOM on the 8 GB Fargate container.
_pipeline_semaphore = threading.Semaphore(1)


def _run_pipeline(
    dest: Path,
    settings,
    jobs: dict,
    document_id: str,
    global_settings=None,
) -> None:
    """Run preprocess → index in a background thread and update jobs dict."""
    # Use global (non-scoped) settings for S3 sync so that user_id subdirectories
    # are preserved in the S3 key and restored correctly on container restart.
    # If no global_settings provided, fall back to the passed settings.
    s3_settings = global_settings or settings
    with _pipeline_semaphore:
        try:
            jobs[document_id]["status"] = "preprocessing"

            def _on_progress(pages_done: int, total_pages: int) -> None:
                jobs[document_id]["pages_done"] = pages_done
                jobs[document_id]["total_pages"] = total_pages

            result = preprocess_document(
                dest, settings=settings, force=True, document_id=document_id, on_progress=_on_progress
            )
            jobs[document_id]["status"] = "indexing"
            index_all_processed_documents(settings=settings)
            jobs[document_id].update(
                status="ready",
                chunk_count=result.chunk_count,
                page_count=result.page_count,
                error=None,
            )
            logger.info("Pipeline complete for document_id=%s", document_id)

            if s3_settings.s3_bucket_name:
                from storage.s3 import sync_embedded_to_s3, sync_processed_to_s3

                try:
                    sync_processed_to_s3(s3_settings)
                    sync_embedded_to_s3(s3_settings)
                except Exception as exc:
                    logger.warning("S3 sync after upload pipeline failed: %s", exc)
        except Exception as exc:
            logger.exception("Pipeline failed for document_id=%s", document_id)
            jobs[document_id].update(status="error", error=str(exc))


@router.post("/preprocess")
async def preprocess(request: Request, file: UploadFile, user: dict = Depends(get_current_user)) -> JSONResponse:
    """
    Upload a PDF and run OCR + layout detection + chunking on it.

    The uploaded file is saved to the raw_documents_dir and then
    preprocessed into frozen artifacts under processed_documents_dir.

    Request: multipart/form-data with a field named "file"
    Response: {"document_id": "...", "chunk_count": N, "page_count": N, "warnings": [...]}

    Note: This is a synchronous endpoint. For large PDFs it may take
    1-2 minutes to return. The connection must stay open.
    """
    settings = request.app.state.settings
    scoped = user_scoped_settings(settings, user["id"])

    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    dest = scoped.raw_documents_dir / file.filename
    logger.info("Saving uploaded file: %s", dest)
    try:
        with dest.open("wb") as out:
            shutil.copyfileobj(file.file, out)
    finally:
        await file.close()

    logger.info("Starting preprocessing: %s", dest)
    try:
        # Run the CPU-bound pipeline in a thread pool so the event loop
        # remains free to answer ALB health checks during long processing.
        # Without this, 48-page PDFs take >90s and ECS kills the task.
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, lambda: preprocess_document(dest, settings=scoped, force=True))
    except Exception as exc:
        logger.exception("Preprocessing failed for %s", dest)
        raise HTTPException(status_code=500, detail=f"Preprocessing failed: {exc}") from exc

    logger.info("Preprocessing complete: document_id=%s chunks=%d", result.document_id, result.chunk_count)

    if settings.s3_bucket_name:
        from storage.s3 import sync_processed_to_s3

        try:
            sync_processed_to_s3(settings)
        except Exception as exc:
            # S3 sync failure is not fatal — artifacts are still on local disk.
            logger.warning("S3 sync after preprocess failed: %s", exc)

    return JSONResponse(
        {
            "document_id": result.document_id,
            "chunk_count": result.chunk_count,
            "page_count": result.page_count,
            "warnings": [w.model_dump() for w in result.warnings],
        }
    )


@router.post("/index")
async def build_index(request: Request, user: dict = Depends(get_current_user)) -> JSONResponse:
    """
    Build the vector index from all preprocessed document artifacts.

    This embeds every chunk using OpenAI and writes the vector store to
    vectorstore_dir/store.json. Existing store is replaced.

    Response: {"indexed_documents": N, "total_chunks": M}
    """
    settings = request.app.state.settings
    scoped = user_scoped_settings(settings, user["id"])

    if not scoped.openai_api_key:
        raise HTTPException(
            status_code=422,
            detail="OPENAI_API_KEY is required for indexing. Set it in your environment.",
        )

    store_path = scoped.vectorstore_dir / "store.json"
    if store_path.exists():
        store_path.unlink()

    logger.info("Building vector index from preprocessed documents")
    try:
        indexed = index_all_processed_documents(settings=scoped)
    except Exception as exc:
        logger.exception("Indexing failed")
        raise HTTPException(status_code=500, detail=f"Indexing failed: {exc}") from exc

    total_chunks = sum(indexed.values())
    logger.info("Indexing complete: %d documents, %d chunks", len(indexed), total_chunks)

    if settings.s3_bucket_name:
        from storage.s3 import sync_embedded_to_s3

        try:
            sync_embedded_to_s3(settings)
        except Exception as exc:
            logger.warning("S3 sync after index failed: %s", exc)

    return JSONResponse(
        {
            "indexed_documents": len(indexed),
            "total_chunks": total_chunks,
            "documents": indexed,
        }
    )


@router.get("")
async def list_documents(request: Request, user: dict = Depends(get_current_user)) -> JSONResponse:
    """
    Return all documents — in-progress jobs and ready artifacts on disk.
    """
    settings = request.app.state.settings
    scoped = user_scoped_settings(settings, user["id"])
    jobs: dict = request.app.state.jobs
    documents = []
    seen: set[str] = set()

    # In-progress and recently finished jobs (from memory)
    for document_id, job in jobs.items():
        seen.add(document_id)
        documents.append(
            {
                "document_id": document_id,
                "source_filename": job.get("source_filename", document_id),
                "status": job["status"],
                "chunk_count": job.get("chunk_count"),
                "page_count": job.get("page_count"),
                "error": job.get("error"),
            }
        )

    # Ready documents from filesystem (not in active jobs, e.g. from previous server runs)
    processed_dir = scoped.processed_documents_dir
    if processed_dir.exists():
        for doc_dir in sorted(p for p in processed_dir.iterdir() if p.is_dir()):
            document_id = doc_dir.name
            if document_id in seen:
                continue
            doc_path = doc_dir / "document.json"
            chunks_path = doc_dir / "chunks.json"
            if not doc_path.exists():
                continue
            doc_data = json.loads(doc_path.read_text(encoding="utf-8"))
            chunk_count = len(json.loads(chunks_path.read_text(encoding="utf-8"))) if chunks_path.exists() else None
            documents.append(
                {
                    "document_id": document_id,
                    "source_filename": doc_data.get("source_filename", document_id),
                    "status": "ready",
                    "chunk_count": chunk_count,
                    "page_count": doc_data.get("page_count"),
                    "error": None,
                }
            )

    return JSONResponse({"documents": documents})


@router.post("/upload")
async def upload(request: Request, file: UploadFile, user: dict = Depends(get_current_user)) -> JSONResponse:
    """
    Upload a PDF and automatically run preprocess + index in the background.

    Returns immediately with document_id and status="preprocessing".
    Poll GET /documents/status/{document_id} every 3 s to track progress.
    """
    settings = request.app.state.settings
    scoped = user_scoped_settings(settings, user["id"])
    jobs: dict = request.app.state.jobs

    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    scoped.raw_documents_dir.mkdir(parents=True, exist_ok=True)
    dest = scoped.raw_documents_dir / file.filename
    logger.info("Saving uploaded file: %s", dest)
    try:
        with dest.open("wb") as out:
            shutil.copyfileobj(file.file, out)
    finally:
        await file.close()

    document_id = dest.stem
    jobs[document_id] = {
        "status": "preprocessing",
        "error": None,
        "chunk_count": None,
        "page_count": None,
        "source_filename": file.filename,
    }
    logger.info("Starting pipeline for document_id=%s", document_id)

    thread = threading.Thread(
        target=_run_pipeline,
        args=(dest, scoped, jobs, document_id, settings),
        daemon=True,
    )
    thread.start()

    return JSONResponse({"document_id": document_id, "status": "preprocessing"})


@router.get("/status/{document_id}")
async def document_status(document_id: str, request: Request, user: dict = Depends(get_current_user)) -> JSONResponse:
    """
    Return the current pipeline status for a document.

    Status values: preprocessing | indexing | ready | error
    """
    settings = request.app.state.settings
    scoped = user_scoped_settings(settings, user["id"])
    jobs: dict = request.app.state.jobs

    if document_id in jobs:
        job = jobs[document_id]
        is_preprocessing = job["status"] == "preprocessing"
        return JSONResponse(
            {
                "document_id": document_id,
                "status": job["status"],
                "error": job.get("error"),
                "chunk_count": job.get("chunk_count"),
                "page_count": job.get("page_count"),
                "pages_done": job.get("pages_done") if is_preprocessing else None,
                "total_pages": job.get("total_pages") if is_preprocessing else None,
            }
        )

    # Fall back to filesystem for docs processed before this server run
    doc_dir = (scoped.processed_documents_dir / document_id).resolve()
    if not str(doc_dir).startswith(str(scoped.processed_documents_dir.resolve())):
        raise HTTPException(status_code=400, detail="Invalid document_id.")
    if not doc_dir.exists():
        raise HTTPException(status_code=404, detail=f"Document {document_id!r} not found.")

    chunks_path = doc_dir / "chunks.json"
    doc_path = doc_dir / "document.json"
    chunk_count = len(json.loads(chunks_path.read_text(encoding="utf-8"))) if chunks_path.exists() else None
    doc_data = json.loads(doc_path.read_text(encoding="utf-8")) if doc_path.exists() else {}
    return JSONResponse(
        {
            "document_id": document_id,
            "status": "ready",
            "error": None,
            "chunk_count": chunk_count,
            "page_count": doc_data.get("page_count"),
            "pages_done": None,
            "total_pages": None,
        }
    )
