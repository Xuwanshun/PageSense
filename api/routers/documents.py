"""
Document management endpoints.

PDF preprocessing (30-120s) is handled synchronously — the HTTP request
blocks until the pipeline finishes. The connection must remain open.
"""
from __future__ import annotations

import logging
import shutil

from fastapi import APIRouter, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse

from document_Process.pipeline import preprocess_document
from rag.retrieve import index_all_processed_documents

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("/preprocess")
async def preprocess(request: Request, file: UploadFile) -> JSONResponse:
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

    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    dest = settings.raw_documents_dir / file.filename
    logger.info("Saving uploaded file: %s", dest)
    try:
        with dest.open("wb") as out:
            shutil.copyfileobj(file.file, out)
    finally:
        await file.close()

    logger.info("Starting preprocessing: %s", dest)
    try:
        result = preprocess_document(dest, settings=settings, force=True)
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

    return JSONResponse({
        "document_id": result.document_id,
        "chunk_count": result.chunk_count,
        "page_count": result.page_count,
        "warnings": [w.model_dump() for w in result.warnings],
    })


@router.post("/index")
async def build_index(request: Request) -> JSONResponse:
    """
    Build the vector index from all preprocessed document artifacts.

    This embeds every chunk using OpenAI and writes the vector store to
    vectorstore_dir/store.json. Existing store is replaced.

    Response: {"indexed_documents": N, "total_chunks": M}
    """
    settings = request.app.state.settings

    if not settings.openai_api_key:
        raise HTTPException(
            status_code=422,
            detail="OPENAI_API_KEY is required for indexing. Set it in your environment.",
        )

    store_path = settings.vectorstore_dir / "store.json"
    if store_path.exists():
        store_path.unlink()

    logger.info("Building vector index from preprocessed documents")
    try:
        indexed = index_all_processed_documents(settings=settings)
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

    return JSONResponse({
        "indexed_documents": len(indexed),
        "total_chunks": total_chunks,
        "documents": indexed,
    })
