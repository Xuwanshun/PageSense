"""Stage 2 — Build the vector index from preprocessed document artifacts."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from config import Settings
from document_Process.models import ProcessedChunk, ProcessedDocument
from rag.retrieve import DocumentRetriever

logger = logging.getLogger(__name__)


def index_document(
    document_id_or_path: str | Path,
    *,
    settings: Settings | None = None,
) -> int:
    """Index one preprocessed document into Pool A (sections) and Pool B (blocks).

    Returns the number of block records indexed.
    """
    resolved = settings or Settings()
    retriever = DocumentRetriever(resolved)
    document_dir = _resolve_document_dir(document_id_or_path, resolved)
    document, chunks = _load_document_chunks(document_dir)
    return retriever.index_processed_chunks(
        chunks,
        document_id=document.document_id if document else document_dir.name,
        source_filename=document.source_filename if document else None,
    )


def index_all_documents(
    *,
    settings: Settings | None = None,
) -> dict[str, int]:
    """Re-index every document in the processed directory.

    Clears existing index stores before rebuilding, so the result is always
    consistent with the current artifacts on disk.
    Returns {document_id: block_count}.
    """
    resolved = settings or Settings()
    retriever = DocumentRetriever(resolved)

    retriever.section_store.clear()
    retriever.block_store.clear()

    indexed: dict[str, int] = {}
    for document_dir in sorted(p for p in resolved.processed_documents_dir.iterdir() if p.is_dir()):
        document, chunks = _load_document_chunks(document_dir)
        if not chunks:
            continue
        document_id = document.document_id if document else document_dir.name
        indexed[document_id] = retriever.index_processed_chunks(
            chunks,
            document_id=document_id,
            source_filename=document.source_filename if document else None,
        )
        logger.info("Indexed %d blocks for document %s", indexed[document_id], document_id)

    return indexed


# ── Private helpers ────────────────────────────────────────────────────────────


def _load_document_chunks(document_dir: Path) -> tuple[ProcessedDocument | None, list[ProcessedChunk]]:
    doc_payload = _load_json(_artifact_path(document_dir, "document.json"))
    chunks_payload = _load_json(_artifact_path(document_dir, "chunks.json")) or []
    document = ProcessedDocument.model_validate(doc_payload) if isinstance(doc_payload, dict) else None
    chunks = [ProcessedChunk.model_validate(item) for item in chunks_payload if isinstance(item, dict)]
    return document, chunks


def _resolve_document_dir(document_id_or_path: str | Path, settings: Settings) -> Path:
    candidate = Path(document_id_or_path)
    if candidate.exists():
        return candidate
    return settings.processed_documents_dir / candidate


def _artifact_path(document_dir: Path, filename: str) -> Path:
    direct = document_dir / filename
    return direct if direct.exists() else document_dir / "structured" / filename


def _load_json(path: Path) -> Any:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))
