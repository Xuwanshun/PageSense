from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from document_Process.models import ProcessedChunk

# ── Region-type classifiers ────────────────────────────────────────────────────

_FIGURE_TYPES = frozenset(["figure", "chart", "figure_description"])
_TABLE_TYPES = frozenset(["table"])
_CAPTION_TYPES = frozenset(["caption", "figure_caption", "table_caption"])
_LIST_TYPES = frozenset(["list"])


def _classify_block_type(region_types: list[str]) -> str:
    for rt in region_types:
        if rt in _FIGURE_TYPES:
            return "figure_description"
        if rt in _TABLE_TYPES:
            return "table"
        if rt in _CAPTION_TYPES:
            return "caption"
        if rt in _LIST_TYPES:
            return "list"
    return "paragraph"


def _build_blocks(text: str, region_types: list[str]) -> list[dict[str, Any]]:
    block_type = _classify_block_type(region_types)
    has_adjacent_figure = any(rt in _FIGURE_TYPES for rt in region_types)
    return [
        {
            "type": block_type,
            "content": seg.strip(),
            "has_adjacent_figure": has_adjacent_figure,
        }
        for seg in text.split("\n\n")
        if seg.strip()
    ]


# ── Record types ───────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: str
    text: str
    metadata: dict[str, Any]
    score: float


@dataclass(frozen=True)
class ChunkRecord:
    """Legacy chunk-level record — kept for backward compatibility."""

    chunk_id: str
    text: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class SectionRecord:
    """Pool A — one record per unique section, embedded as section_title + summary."""

    section_id: str
    text: str  # embed text: "section_title — section_summary"
    metadata: dict[str, Any]


@dataclass(frozen=True)
class BlockRecord:
    """Pool B — one record per block, embedded with section context prefix."""

    block_id: str
    text: str  # embed text: "[section_title] > content | figure_desc"
    metadata: dict[str, Any]


@dataclass(frozen=True)
class DocumentRecord:
    """Pool 0 — one record per document, embedded from document_summary."""

    doc_id: str
    text: str  # embed text: "{source_filename}: {document_summary}"
    metadata: dict[str, Any]


# ── Section record builder ─────────────────────────────────────────────────────


def _section_id(doc_id: str, section_title: str) -> str:
    return f"{doc_id}:section:{abs(hash(section_title)) % 1_000_000}"


def section_records_from_processed_chunks(
    chunks: list[ProcessedChunk],
    *,
    document_id: str | None = None,
    source_filename: str | None = None,
) -> list[SectionRecord]:
    """Build Pool A — one SectionRecord per unique section across all chunks."""
    sections: dict[str, dict[str, Any]] = {}

    for chunk in chunks:
        doc_id = document_id or chunk.metadata.get("document_id") or ""
        section_title = chunk.metadata.get("parent_title") or "untitled"
        section_subtitle = chunk.metadata.get("parent_subtitle") or ""
        section_summary = chunk.metadata.get("section_summary") or ""
        page = chunk.page_number or 0
        sid = _section_id(doc_id, section_title)

        if sid not in sections:
            sections[sid] = {
                "doc_id": doc_id,
                "section_title": section_title,
                "section_subtitle": section_subtitle,
                "section_summary": section_summary,
                "page_start": page,
                "page_end": page,
            }
        else:
            s = sections[sid]
            s["page_start"] = min(s["page_start"], page)
            s["page_end"] = max(s["page_end"], page)
            if not s["section_summary"] and section_summary:
                s["section_summary"] = section_summary
            if not s["section_subtitle"] and section_subtitle:
                s["section_subtitle"] = section_subtitle

    records = []
    for sid, s in sections.items():
        title = s["section_title"]
        subtitle = s["section_subtitle"]
        summary = s["section_summary"]

        # Build embed text: "title — subtitle: summary" / "title — summary" / "title"
        if subtitle and summary:
            embed_text = f"{title} — {subtitle}: {summary}"
        elif summary:
            embed_text = f"{title} — {summary}"
        else:
            embed_text = title

        records.append(
            SectionRecord(
                section_id=sid,
                text=embed_text,
                metadata={
                    "doc_id": s["doc_id"],
                    "document_id": s["doc_id"],
                    "section_id": sid,
                    "section_title": title,
                    "section_subtitle": subtitle,
                    "section_summary": summary,
                    "page_start": s["page_start"],
                    "page_end": s["page_end"],
                    "source_filename": source_filename,
                },
            )
        )
    return records


# ── Document record builder (Pool 0) ──────────────────────────────────────────


def document_record_from_summary(
    doc_id: str,
    source_filename: str,
    document_summary: str,
    *,
    chunks: list[ProcessedChunk] | None = None,
    page_count: int = 0,
    section_count: int = 0,
) -> DocumentRecord:
    """Build one Pool 0 DocumentRecord for a processed document.

    Embed text: "{source_filename}: {document_summary}"
    Falls back to aggregating first sentences of section summaries when
    document_summary is empty (e.g. FAST_MODE artifacts).
    Minimum embed text is always source_filename alone.
    """
    summary = (document_summary or "").strip()
    if not summary and chunks:
        # Aggregate first sentence of each unique section summary
        seen: set[str] = set()
        parts: list[str] = []
        for chunk in chunks:
            sec_sum = (chunk.metadata.get("section_summary") or "").strip()
            if sec_sum and sec_sum not in seen:
                seen.add(sec_sum)
                first_sentence = sec_sum.split(".")[0].strip()
                if first_sentence:
                    parts.append(first_sentence)
        summary = ". ".join(parts[:5])

    if summary:
        embed_text = f"{source_filename}: {summary}"
    else:
        embed_text = source_filename

    return DocumentRecord(
        doc_id=doc_id,
        text=embed_text,
        metadata={
            "doc_id": doc_id,
            "source_filename": source_filename,
            "page_count": page_count,
            "section_count": section_count,
        },
    )


# ── Block record builder ───────────────────────────────────────────────────────


def _block_id(chunk_id: str, local_index: int) -> str:
    return f"{chunk_id}:block:{local_index}"


def block_records_from_processed_chunks(
    chunks: list[ProcessedChunk],
    *,
    document_id: str | None = None,
    source_filename: str | None = None,
) -> list[BlockRecord]:
    """Build Pool B — one BlockRecord per block within each chunk."""
    records: list[BlockRecord] = []

    for chunk in chunks:
        text = chunk.page_content or chunk.text
        if not text.strip():
            continue

        doc_id = document_id or chunk.metadata.get("document_id") or ""
        section_title = chunk.metadata.get("parent_title") or "untitled"
        page = chunk.page_number or 0
        region_types = chunk.region_types or []
        blocks_list: list[dict[str, Any]] = chunk.metadata.get(
            "blocks"
        ) or _build_blocks(text, region_types)
        sid = _section_id(doc_id, section_title)
        chunk_block_idx = chunk.metadata.get("block_index") or 0

        # Find the first figure description in this chunk for adjacent-context enrichment
        figure_desc = next(
            (
                b["content"]
                for b in blocks_list
                if b.get("type") == "figure_description"
            ),
            None,
        )

        crop_references = chunk.crop_references or []

        for local_idx, block in enumerate(blocks_list):
            content = block.get("content", "").strip()
            block_type = block.get("type", "paragraph")
            has_adj = block.get("has_adjacent_figure", False)

            if not content:
                continue

            # Enrich embed text with adjacent figure description for non-figure blocks
            adj_desc = (
                figure_desc if has_adj and block_type != "figure_description" else None
            )
            embed_text = f"[{section_title}] > {content}"
            if adj_desc:
                embed_text += f" | {adj_desc}"

            # Signal that a visual crop exists for table/figure blocks so the
            # embedding captures "has_visual_crop" even without VLM summaries.
            block_crop_refs: list[str] = []
            if block_type in ("table", "figure_description") and crop_references:
                block_crop_refs = crop_references
                embed_text += f" [has visual crop: {crop_references[0]}]"

            linked_figure_id = (
                chunk.metadata.get("linked_region_id", "") or ""
                if block_type == "caption"
                else ""
            )
            records.append(
                BlockRecord(
                    block_id=_block_id(chunk.chunk_id, local_idx),
                    text=embed_text,
                    metadata={
                        "doc_id": doc_id,
                        "document_id": doc_id,
                        "section_id": sid,
                        "section_title": section_title,
                        "block_index": chunk_block_idx * 100 + local_idx,
                        "page": page,
                        "page_number": page,
                        "block_type": block_type,
                        "has_adjacent_figure": has_adj,
                        "content": content,
                        "chunk_id": chunk.chunk_id,
                        "source_filename": source_filename,
                        "source_region_ids": [],  # kept for qa.py compat
                        "crop_references": block_crop_refs,
                        "linked_figure_id": linked_figure_id,
                    },
                )
            )

    return records
