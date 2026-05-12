from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any

from config import Settings
from document_Process.models import ProcessedChunk
from rag.chunk import (
    ChunkRecord,
    RetrievedChunk,
    block_records_from_processed_chunks,
    section_records_from_processed_chunks,
)
from rag.embed import EmbeddingBackend, build_embedding_backend

logger = logging.getLogger(__name__)

# ── Retrieval constants ────────────────────────────────────────────────────────

_SECTION_THRESHOLD = 0.55
_TOP_SECTIONS = 3
_NEIGHBOR_WINDOW = 2

_BOOST_FIGURE = 0.15
_BOOST_TABLE = 0.10
_BOOST_ADJACENT = 0.05

_VISUAL_TERMS = frozenset(
    [
        "figure",
        "chart",
        "diagram",
        "shows",
        "illustrated",
        "image",
        "plot",
        "visual",
    ]
)
_DATA_TERMS = [
    "how many",
    "how much",
    "compare",
    "total",
    "percent",
    "percentage",
    "list",
    "table",
    "count",
    "number",
    "rate",
    "breakdown",
]


# ── Retrieval post-processing ──────────────────────────────────────────────────


def _token_overlap(query: str, content: str) -> int:
    query_tokens = set(query.lower().split())
    content_lower = content.lower()
    return sum(1 for t in query_tokens if t in content_lower)


def _apply_query_boosts(
    blocks: list[RetrievedChunk], query: str
) -> list[RetrievedChunk]:
    """Additive score boosts based on block type, query intent, and token overlap."""
    lowered = query.lower()
    has_visual = any(t in lowered for t in _VISUAL_TERMS)
    has_data = any(phrase in lowered for phrase in _DATA_TERMS)

    result: list[RetrievedChunk] = []
    for block in blocks:
        boost = 0.0
        block_type = block.metadata.get("block_type", "")
        content = block.metadata.get("content") or block.text

        if block_type == "figure_description" and has_visual:
            boost += _BOOST_FIGURE
        if block_type == "table" and has_data:
            boost += _BOOST_TABLE
        if block.metadata.get("has_adjacent_figure"):
            boost += _BOOST_ADJACENT
        boost += _token_overlap(query, content) * 0.01

        result.append(
            RetrievedChunk(
                chunk_id=block.chunk_id,
                text=block.text,
                metadata=block.metadata,
                score=block.score + boost,
            )
        )
    return sorted(result, key=lambda b: b.score, reverse=True)


def _expand_neighbors(
    matched: list[RetrievedChunk],
    all_blocks: list[RetrievedChunk],
    *,
    window: int = _NEIGHBOR_WINDOW,
) -> list[RetrievedChunk]:
    """Expand each matched block by ±window neighbors within the same section."""
    section_map: dict[tuple[str, str], list[RetrievedChunk]] = {}
    for block in all_blocks:
        key = (
            str(block.metadata.get("doc_id") or ""),
            str(block.metadata.get("section_id") or ""),
        )
        section_map.setdefault(key, []).append(block)
    for blocks in section_map.values():
        blocks.sort(key=lambda b: b.metadata.get("block_index", 0))

    position: dict[str, tuple[tuple[str, str], int]] = {}
    for key, blocks in section_map.items():
        for pos, block in enumerate(blocks):
            position[block.chunk_id] = (key, pos)

    seen: set[str] = set()
    result: list[RetrievedChunk] = []
    for hit in matched:
        loc = position.get(hit.chunk_id)
        if loc is None:
            if hit.chunk_id not in seen:
                seen.add(hit.chunk_id)
                result.append(hit)
            continue
        key, pos = loc
        section = section_map[key]
        for neighbor in section[max(0, pos - window) : pos + window + 1]:
            if neighbor.chunk_id not in seen:
                seen.add(neighbor.chunk_id)
                result.append(neighbor)
    return result


def _sort_by_reading_order(blocks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    return sorted(
        blocks,
        key=lambda b: (
            str(b.metadata.get("doc_id") or ""),
            str(b.metadata.get("section_id") or ""),
            int(b.metadata.get("block_index", 0)),
        ),
    )


# ── Vector store ──────────────────────────────────────────────────────────────


class JsonVectorStore:
    def __init__(self, store_path: Path) -> None:
        self.store_path = store_path
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        self._cached_rows: list[dict[str, Any]] | None = None
        self._cache_mtime: float | None = None

    def clear(self) -> None:
        if self.store_path.exists():
            self.store_path.unlink()
        self._cached_rows = None
        self._cache_mtime = None

    def _load_rows(self) -> list[dict[str, Any]]:
        if not self.store_path.exists():
            self._cached_rows = None
            self._cache_mtime = None
            return []
        mtime = self.store_path.stat().st_mtime
        if self._cached_rows is not None and self._cache_mtime == mtime:
            return self._cached_rows
        self._cached_rows = json.loads(self.store_path.read_text(encoding="utf-8")).get(
            "rows", []
        )
        self._cache_mtime = mtime
        return self._cached_rows

    def _save_rows(self, rows: list[dict[str, Any]]) -> None:
        self.store_path.write_text(
            json.dumps({"rows": rows}, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        self._cached_rows = None
        self._cache_mtime = None

    def upsert(self, chunks: list[ChunkRecord], embeddings: list[list[float]]) -> None:
        existing = {row["chunk_id"]: row for row in self._load_rows()}
        for chunk, emb in zip(chunks, embeddings, strict=False):
            existing[chunk.chunk_id] = {
                "chunk_id": chunk.chunk_id,
                "text": chunk.text,
                "metadata": chunk.metadata,
                "embedding": emb,
            }
        self._save_rows(list(existing.values()))

    def query(
        self,
        embedding: list[float],
        top_k: int,
        *,
        doc_filter: list[str] | None = None,
        filter_doc_ids: set[str] | None = None,
    ) -> list[RetrievedChunk]:
        # filter_doc_ids is the new Pool-0-scoped filter; doc_filter is the legacy list form
        combined: set[str] | None = None
        if filter_doc_ids is not None:
            combined = filter_doc_ids
        elif doc_filter is not None:
            combined = set(doc_filter)

        scored: list[RetrievedChunk] = []
        for row in self._load_rows():
            if combined is not None:
                meta = row.get("metadata", {})
                # Records may store doc_id under "doc_id" or "document_id"
                row_doc = meta.get("doc_id") or meta.get("document_id") or ""
                if row_doc not in combined:
                    continue
            scored.append(
                RetrievedChunk(
                    chunk_id=row["chunk_id"],
                    text=row.get("text", ""),
                    metadata=row.get("metadata", {}),
                    score=_cosine_similarity(embedding, row.get("embedding", [])),
                )
            )
        return sorted(scored, key=lambda c: c.score, reverse=True)[:top_k]

    def get_all_chunks(
        self, *, doc_filter: list[str] | None = None
    ) -> list[RetrievedChunk]:
        filter_set = set(doc_filter) if doc_filter else None
        return [
            RetrievedChunk(
                chunk_id=row["chunk_id"],
                text=row.get("text", ""),
                metadata=row.get("metadata", {}),
                score=0.0,
            )
            for row in self._load_rows()
            if filter_set is None
            or row.get("metadata", {}).get("document_id") in filter_set
        ]


# ── DocumentRetriever ──────────────────────────────────────────────────────────


class DocumentRetriever:
    """Two-pool retriever: Pool A (sections) + Pool B (blocks).

    Retrieval pipeline:
    1. Section pre-filter — cosine search Pool A → top sections above threshold.
    2. Block search       — dense search Pool B within candidate sections.
    3. Boost + expand     — additive type boosts + ±2 neighbor expansion.
    4. Reading order      — final results sorted for coherent context presentation.
    """

    def __init__(
        self,
        settings: Settings,
        *,
        embedding_backend: EmbeddingBackend | None = None,
        document_store: JsonVectorStore | None = None,
        section_store: JsonVectorStore | None = None,
        block_store: JsonVectorStore | None = None,
    ) -> None:
        self.settings = settings
        self.embedding_backend = embedding_backend or build_embedding_backend(settings)
        self.document_store = document_store or JsonVectorStore(
            settings.vectorstore_dir / "documents.json"
        )
        self.section_store = section_store or JsonVectorStore(
            settings.vectorstore_dir / "sections.json"
        )
        self.block_store = block_store or JsonVectorStore(
            settings.vectorstore_dir / "blocks.json"
        )

    # ── Indexing ───────────────────────────────────────────────────────────────

    def index_processed_chunks(
        self,
        chunks: list[ProcessedChunk],
        *,
        document_id: str | None = None,
        source_filename: str | None = None,
    ) -> int:
        """Index chunks into Pool A (sections) and Pool B (blocks)."""
        section_records = section_records_from_processed_chunks(
            chunks, document_id=document_id, source_filename=source_filename
        )
        block_records = block_records_from_processed_chunks(
            chunks, document_id=document_id, source_filename=source_filename
        )

        if section_records:
            sec_embs = self.embedding_backend.embed_texts(
                [r.text for r in section_records]
            )
            self.section_store.upsert(
                [
                    ChunkRecord(
                        chunk_id=r.section_id,
                        text=r.text,
                        metadata={**r.metadata, "pool": "section"},
                    )
                    for r in section_records
                ],
                sec_embs,
            )

        if block_records:
            blk_embs = self.embedding_backend.embed_texts(
                [r.text for r in block_records]
            )
            self.block_store.upsert(
                [
                    ChunkRecord(
                        chunk_id=r.block_id,
                        text=r.text,
                        metadata={**r.metadata, "pool": "block"},
                    )
                    for r in block_records
                ],
                blk_embs,
            )

        return len(block_records)

    # ── Retrieval ──────────────────────────────────────────────────────────────

    def retrieve(
        self,
        question: str,
        top_k: int | None = None,
        *,
        doc_filter: list[str] | None = None,
    ) -> list[RetrievedChunk]:
        """3-step retrieval: section pre-filter → block search → boost + expand."""
        k = top_k or self.settings.default_top_k
        query_embedding = self.embedding_backend.embed_texts([question])[0]

        candidate_section_ids = self._candidate_sections(query_embedding, doc_filter)
        raw_blocks = self._search_blocks(
            query_embedding, candidate_section_ids, k * 2, doc_filter
        )

        boosted = _apply_query_boosts(raw_blocks, question)
        all_blocks = self.block_store.get_all_chunks(doc_filter=doc_filter or None)
        expanded = _expand_neighbors(boosted[:k], all_blocks)
        return _sort_by_reading_order(expanded)

    # ── Private helpers ────────────────────────────────────────────────────────

    def _candidate_sections(
        self,
        query_embedding: list[float],
        doc_filter: list[str] | None,
    ) -> set[str]:
        results = self.section_store.query(
            query_embedding, _TOP_SECTIONS * 2, doc_filter=doc_filter or None
        )
        above = [r for r in results if r.score >= _SECTION_THRESHOLD][:_TOP_SECTIONS]
        return {str(r.metadata.get("section_id") or "") for r in above}

    def _search_blocks(
        self,
        query_embedding: list[float],
        candidate_section_ids: set[str],
        fetch_k: int,
        doc_filter: list[str] | None,
    ) -> list[RetrievedChunk]:
        raw = self.block_store.query(
            query_embedding, fetch_k * 3, doc_filter=doc_filter or None
        )
        if candidate_section_ids:
            in_sections = [
                b
                for b in raw
                if str(b.metadata.get("section_id") or "") in candidate_section_ids
            ]
            if in_sections:
                return in_sections[:fetch_k]
        return raw[:fetch_k]


# ── Shared utility ─────────────────────────────────────────────────────────────


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right:
        return 0.0
    size = min(len(left), len(right))
    dot = sum(left[i] * right[i] for i in range(size))
    left_norm = math.sqrt(sum(v * v for v in left[:size])) or 1.0
    right_norm = math.sqrt(sum(v * v for v in right[:size])) or 1.0
    return dot / (left_norm * right_norm)
