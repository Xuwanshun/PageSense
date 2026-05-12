"""Stage 5 — Chunking & Summarization.

Chunks ordered text blocks using a character budget with overlap carry-forward.
Optionally generates section and document summaries via an LLM.

LLM summarization is skipped when fast_mode=True, use_document_intelligence=False,
or no API key is present. Section summaries are written into each Section.summary;
the document-level summary is returned in SummarizeResult.document_summary.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any

from config import Settings
from document_Process.models.internal import (
    HierarchyResult,
    LoadResult,
    Section,
    SummarizeResult,
)
from document_Process.models.legacy import (
    CroppedRegionAsset,
    LayoutRegion,
    OrderedTextBlock,
    ProcessedChunk,
    VisualRegionSummary,
)

logger = logging.getLogger(__name__)

_SUMMARIZE_SYSTEM = "You are a document analysis assistant. Return only valid JSON, no markdown. Use null if unsure."


class SummarizeStage:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def run(
        self, load_result: LoadResult, hier_result: HierarchyResult
    ) -> SummarizeResult:
        logger.info("Stage 5 — Summarize: %s block(s)", len(hier_result.ordered_blocks))
        target_chars = self.settings.preprocess_chunk_size
        overlap_chars = self.settings.preprocess_chunk_overlap
        chunks = build_chunks(
            document_id=load_result.document_id,
            source_file=load_result.source_filename,
            ordered_blocks=hier_result.ordered_blocks,
            regions=load_result.regions,
            target_chars=target_chars,
            overlap_chars=overlap_chars,
        )

        skip_llm = (
            self.settings.fast_mode
            or not self.settings.use_document_intelligence
            or not self.settings.openai_api_key
        )
        if skip_llm:
            return SummarizeResult(chunks=chunks, document_summary="")

        document_summary = asyncio.run(
            self._summarize_document(hier_result.document.sections, load_result)
        )

        # Propagate section summaries back into chunk metadata so rag/chunk.py can read them.
        summary_by_title = {
            sec.title: sec.summary
            for sec in hier_result.document.sections
            if sec.summary
        }
        for chunk in chunks:
            title = chunk.metadata.get("parent_title", "")
            if title and title in summary_by_title:
                chunk.metadata["section_summary"] = summary_by_title[title]

        return SummarizeResult(chunks=chunks, document_summary=document_summary)

    async def _summarize_document(
        self, sections: list[Section], load_result: LoadResult
    ) -> str:
        semaphore = asyncio.Semaphore(self.settings.llm_concurrency_limit)
        tasks = [self._summarize_section(sec, semaphore) for sec in sections]
        await asyncio.gather(*tasks)

        # Document-level summary from aggregated section summaries
        if not self.settings.use_document_summary:
            return ""

        structure_lines = [
            f"[{sec.title}] {sec.summary[:100]}" for sec in sections if sec.summary
        ]
        if not structure_lines:
            return ""

        try:
            return await asyncio.get_event_loop().run_in_executor(
                None,
                self._call_document_summary_sync,
                structure_lines,
                len(sections),
            )
        except Exception as exc:
            logger.warning("Document summary generation failed: %s", exc)
            return ""

    async def _summarize_section(
        self, section: Section, semaphore: asyncio.Semaphore
    ) -> None:
        async with semaphore:
            try:
                summary = await asyncio.get_event_loop().run_in_executor(
                    None, self._call_section_summary_sync, section
                )
                section.summary = summary
            except Exception as exc:
                logger.warning(
                    "Section summarization failed for %s: %s", section.title, exc
                )

    def _call_section_summary_sync(self, section: Section) -> str:
        from openai import OpenAI

        client = OpenAI(
            api_key=self.settings.openai_api_key or "",
            base_url=self.settings.openai_base_url,
        )
        few_shot = (
            "Examples:\n"
            '1. Introduction: {"summary": "Overview of the report scope and methodology.", '
            '"key_topics": ["scope", "methodology"]}\n'
            '2. Data analysis: {"summary": "Revenue analysis showing 23% YoY growth.", '
            '"key_topics": ["revenue", "growth"]}\n'
        )
        user_prompt = (
            f"Section title: {section.title}\n"
            f"Text preview:\n{section.flat_text[:800]}\n\n"
            f"{few_shot}"
            "Return JSON with fields: summary (str), key_topics (list)."
        )
        response = client.chat.completions.create(
            model=self.settings.synthesis_model,
            messages=[
                {"role": "system", "content": _SUMMARIZE_SYSTEM},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0,
        )
        raw = (response.choices[0].message.content or "").strip()
        result = _safe_parse_json(raw)
        return str(result.get("summary") or "")

    def _call_document_summary_sync(
        self, structure_lines: list[str], section_count: int
    ) -> str:
        from openai import OpenAI

        client = OpenAI(
            api_key=self.settings.openai_api_key or "",
            base_url=self.settings.openai_base_url,
        )
        user_prompt = (
            f"Section count: {section_count}\n"
            f"Document structure:\n{chr(10).join(structure_lines)}\n\n"
            "Return JSON with: summary (str), topics (list)."
        )
        response = client.chat.completions.create(
            model=self.settings.synthesis_model,
            messages=[
                {"role": "system", "content": _SUMMARIZE_SYSTEM},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0,
        )
        raw = (response.choices[0].message.content or "").strip()
        result = _safe_parse_json(raw)
        return str(result.get("summary") or "")


# ── Private helpers ────────────────────────────────────────────────────────────


def _block_section_title(
    block: OrderedTextBlock, region_by_id: dict[str, LayoutRegion]
) -> str | None:
    for rid in block.region_ids:
        region = region_by_id.get(rid)
        if region:
            pt = region.metadata.get("parent_title")
            if pt is not None:
                return str(pt)
    return None


def _overlap_blocks(
    blocks: list[OrderedTextBlock], overlap_chars: int
) -> list[OrderedTextBlock]:
    if overlap_chars <= 0:
        return []
    kept: list[OrderedTextBlock] = []
    total = 0
    for block in reversed(blocks):
        kept.insert(0, block)
        total += len(block.text)
        if total >= overlap_chars:
            break
    return kept


def _safe_parse_json(text: str) -> dict[str, Any]:
    stripped = re.sub(r"```(?:json)?\s*|\s*```", "", text).strip()
    try:
        r = json.loads(stripped)
        if isinstance(r, dict):
            return r
    except Exception:
        pass
    match = re.search(r"\{.*\}", stripped, re.DOTALL)
    if match:
        try:
            r = json.loads(match.group())
            if isinstance(r, dict):
                return r
        except Exception:
            pass
    return {}


# ── Compatibility shim for tests ──────────────────────────────────────────────


def build_chunks(
    *,
    document_id: str,
    source_file: str,
    ordered_blocks: list[OrderedTextBlock],
    regions: list[LayoutRegion],
    target_chars: int = 1800,
    overlap_chars: int = 200,
) -> list[ProcessedChunk]:
    """Character-budget chunking with overlap carry-forward.

    Kept as a free function so the test suite can call it directly.
    """
    region_by_id: dict[str, LayoutRegion] = {r.region_id: r for r in regions}

    blocks_by_page: dict[int, list[OrderedTextBlock]] = {}
    for block in ordered_blocks:
        blocks_by_page.setdefault(block.page_number, []).append(block)

    chunks: list[ProcessedChunk] = []
    next_index = 1

    for page_number, blocks in sorted(blocks_by_page.items()):
        current: list[OrderedTextBlock] = []
        current_section: str | None = None
        for block in blocks:
            block_section = _block_section_title(block, region_by_id)

            # Section boundary: flush with no overlap carry-forward across sections.
            if current and block_section != current_section:
                chunks.append(
                    _make_chunk(
                        document_id,
                        source_file,
                        page_number,
                        next_index,
                        current,
                        region_by_id,
                    )
                )
                next_index += 1
                current = []
            # Character budget: flush with overlap carry-forward within the same section.
            elif (
                current
                and len("\n\n".join(b.text for b in current + [block])) > target_chars
            ):
                chunks.append(
                    _make_chunk(
                        document_id,
                        source_file,
                        page_number,
                        next_index,
                        current,
                        region_by_id,
                    )
                )
                next_index += 1
                current = _overlap_blocks(current, overlap_chars)

            if not current:
                current_section = block_section
            current.append(block)
        if current:
            chunks.append(
                _make_chunk(
                    document_id,
                    source_file,
                    page_number,
                    next_index,
                    current,
                    region_by_id,
                )
            )
            next_index += 1

    return chunks


def _make_chunk(
    document_id: str,
    source_filename: str,
    page_number: int,
    index: int,
    blocks: list[OrderedTextBlock],
    region_by_id: dict[str, LayoutRegion],
) -> ProcessedChunk:
    chunk_id = f"{document_id}:chunk:{index}"
    text = "\n\n".join(b.text for b in blocks if b.text.strip()).strip()
    region_ids = sorted({rid for b in blocks for rid in b.region_ids})
    crop_refs = [
        region_by_id[rid].crop_path
        for rid in region_ids
        if rid in region_by_id and region_by_id[rid].crop_path
    ]
    region_types = sorted(
        {region_by_id[rid].region_type for rid in region_ids if rid in region_by_id}
    )
    bbox_refs = [b.bbox.as_list() for b in blocks if b.bbox is not None]
    item_ids = [iid for b in blocks for iid in b.item_ids]
    ordered_block_ids = [b.block_id for b in blocks]

    parent_title: str | None = None
    parent_subtitle: str | None = None
    linked_region_id: str | None = None
    for rid in region_ids:
        region = region_by_id.get(rid)
        if region:
            if parent_title is None:
                pt = region.metadata.get("parent_title")
                if pt is not None:
                    parent_title = str(pt)
                    ps = region.metadata.get("parent_subtitle")
                    parent_subtitle = str(ps) if ps else None
            if linked_region_id is None and region.metadata.get("linked_region_id"):
                linked_region_id = str(region.metadata["linked_region_id"])

    metadata: dict[str, Any] = {
        "document_id": document_id,
        "source_filename": source_filename,
        "page_number": page_number,
        "crop_references": [r for r in crop_refs if r],
        "parent_title": parent_title,
        "parent_subtitle": parent_subtitle,
        "linked_region_id": linked_region_id,
    }

    return ProcessedChunk(
        chunk_id=chunk_id,
        text="",
        page_content=text,
        page_number=page_number,
        ordered_block_ids=ordered_block_ids,
        item_ids=item_ids,
        source_region_ids=region_ids,
        region_types=region_types,
        bbox_references=bbox_refs,
        crop_references=[r for r in crop_refs if r],
        metadata=metadata,
    )


def build_visual_summaries(
    *,
    regions: list[LayoutRegion],
    ordered_blocks: list[OrderedTextBlock],
    chunks: list[Any],
    cropped_assets: list[CroppedRegionAsset],
) -> list[VisualRegionSummary]:
    """Build VisualRegionSummary list for table/figure regions.

    Kept as a free function so the test suite can call it directly.
    """
    asset_by_region = {a.region_id: a for a in cropped_assets}
    chunks_by_region: dict[str, list[Any]] = {}
    for chunk in chunks:
        for rid in getattr(chunk, "source_region_ids", []):
            chunks_by_region.setdefault(rid, []).append(chunk)
    blocks_by_page: dict[int, list[OrderedTextBlock]] = {}
    for block in ordered_blocks:
        blocks_by_page.setdefault(block.page_number, []).append(block)

    summaries: list[VisualRegionSummary] = []
    for region in regions:
        if region.region_type not in {"table", "figure"}:
            continue
        page_blocks = blocks_by_page.get(region.page_number, [])
        overlapping = [
            b
            for b in page_blocks
            if b.bbox is not None and b.bbox.intersection_area(region.bbox) > 0
        ]
        if not overlapping:
            overlapping = sorted(
                [b for b in page_blocks if b.bbox is not None],
                key=lambda b: min(
                    abs(b.bbox.y0 - region.bbox.y1),
                    abs(b.bbox.y1 - region.bbox.y0),
                ),
            )[:3]
        region_chunks = chunks_by_region.get(region.region_id, [])
        block_text = " ".join(b.text for b in overlapping if b.text.strip()).strip()
        chunk_text = " ".join(c.text for c in region_chunks if c.text.strip()).strip()
        summary_text = (
            block_text
            or chunk_text
            or f"Detected {region.region_type} region on page {region.page_number}."
        )[:1200]
        asset = asset_by_region.get(region.region_id)
        summaries.append(
            VisualRegionSummary(
                summary_id=f"summary_{region.region_id}",
                region_id=region.region_id,
                asset_id=asset.asset_id if asset else None,
                page_number=region.page_number,
                region_type=region.region_type,
                crop_path=asset.crop_path if asset else region.crop_path,
                linked_block_ids=[b.block_id for b in overlapping],
                linked_chunk_ids=[c.chunk_id for c in region_chunks],
                summary_text=summary_text,
                metadata={
                    "label": region.metadata.get("label"),
                    "detector": region.metadata.get("detector"),
                },
            )
        )
    return summaries
