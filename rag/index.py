"""Stage 2 — Build the vector index from preprocessed document artifacts."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from config import Settings
from document_Process.models import ProcessedChunk, ProcessedDocument
from rag.chunk import ChunkRecord, document_record_from_summary
from rag.retrieve import DocumentRetriever

logger = logging.getLogger(__name__)


def index_document(
    document_id_or_path: str | Path,
    *,
    settings: Settings | None = None,
) -> int:
    """Index one preprocessed document into Pool 0 (documents), Pool A (sections), Pool B (blocks).

    Returns the number of block records indexed.
    """
    resolved = settings or Settings()
    retriever = DocumentRetriever(resolved)
    document_dir = _resolve_document_dir(document_id_or_path, resolved)
    document, chunks = _load_document_chunks(document_dir)
    doc_id = document.document_id if document else document_dir.name
    source_filename = document.source_filename if document else None
    _index_pool0(document_dir, doc_id, source_filename, chunks, retriever)
    return retriever.index_processed_chunks(
        chunks,
        document_id=doc_id,
        source_filename=source_filename,
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

    retriever.document_store.clear()
    retriever.section_store.clear()
    retriever.block_store.clear()

    indexed: dict[str, int] = {}
    for document_dir in sorted(
        p for p in resolved.processed_documents_dir.iterdir() if p.is_dir()
    ):
        document, chunks = _load_document_chunks(document_dir)
        if not chunks:
            continue
        document_id = document.document_id if document else document_dir.name
        source_filename = document.source_filename if document else None
        _index_pool0(document_dir, document_id, source_filename, chunks, retriever)
        indexed[document_id] = retriever.index_processed_chunks(
            chunks,
            document_id=document_id,
            source_filename=source_filename,
        )
        logger.info(
            "Indexed %d blocks for document %s", indexed[document_id], document_id
        )

    total_blocks = sum(indexed.values())
    if total_blocks > 5000 and not resolved.prefer_chroma:
        logger.warning(
            "Pool B contains %d blocks across %d documents. "
            "JsonVectorStore loads all vectors into memory on first query. "
            "Consider setting PREFER_CHROMA=true for better memory efficiency at this scale.",
            total_blocks,
            len(indexed),
        )

    return indexed


# ── Private helpers ────────────────────────────────────────────────────────────


def _index_pool0(
    document_dir: Path,
    doc_id: str,
    source_filename: str | None,
    chunks: list[ProcessedChunk],
    retriever: DocumentRetriever,
) -> None:
    """Embed and upsert one DocumentRecord into Pool 0 (documents.json)."""
    manifest_path = _artifact_path(document_dir, "manifest.json")
    manifest_data = _load_json(manifest_path) or {}
    fname = source_filename or manifest_data.get("source_filename") or doc_id[:8]

    # Read document_summary from manifest (written since schema 8.0.0).
    # Fall back to document.json for backward compatibility with older artifacts.
    document_summary = manifest_data.get("document_summary") or ""
    if not document_summary:
        doc_path = _artifact_path(document_dir, "document.json")
        doc_data = _load_json(doc_path) or {}
        document_summary = (
            doc_data.get("document_summary", "")
            or doc_data.get("descriptor", {}).get("summary", "")
            or ""
        )
    page_count = manifest_data.get("page_count") or 0
    # Count distinct section titles as proxy for section_count
    section_count = len(
        {
            c.metadata.get("parent_title")
            for c in chunks
            if c.metadata.get("parent_title")
        }
    )

    doc_record = document_record_from_summary(
        doc_id=doc_id,
        source_filename=fname,
        document_summary=document_summary,
        chunks=chunks,
        page_count=page_count,
        section_count=section_count,
    )
    embedding = retriever.embedding_backend.embed_texts([doc_record.text])[0]
    retriever.document_store.upsert(
        [
            ChunkRecord(
                chunk_id=doc_record.doc_id,
                text=doc_record.text,
                metadata=doc_record.metadata,
            )
        ],
        [embedding],
    )
    logger.info("Pool 0: indexed document %s (%s)", doc_id[:12], fname)


def _load_document_chunks(
    document_dir: Path,
) -> tuple[ProcessedDocument | None, list[ProcessedChunk]]:
    doc_payload = _load_json(_artifact_path(document_dir, "document.json"))
    chunks_payload = _load_json(_artifact_path(document_dir, "chunks.json")) or []
    document = (
        ProcessedDocument.model_validate(doc_payload)
        if isinstance(doc_payload, dict)
        else None
    )
    chunks = [
        ProcessedChunk.model_validate(item)
        for item in chunks_payload
        if isinstance(item, dict)
    ]
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
