"""Stage 5 — Hierarchical Chunking & Summarization.

Chunks text within each lowest-level hierarchy node, respecting target_chars
and overlap_chars.  Then builds LLM summaries bottom-up:
  subsection chunks → subsection summary
  subsection summaries → section summary
  section summaries → document descriptor + summary embedding

When fast_mode=True or use_document_intelligence=False, all LLM calls are
skipped and only flat chunking is done.

LLM summarization calls run concurrently with asyncio.Semaphore(llm_concurrency_limit).
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any

from config import Settings
from document_Process.cache import StageCache
from document_Process.models import (
    LayoutRegion,
    OrderedTextBlock,
    ProcessedChunk,
    Stage1Result,
    Stage4Result,
)
from document_Process.models.stage3 import VisualDescription
from document_Process.models.stage4 import HierarchyNode
from document_Process.models.stage5 import (
    ChunkWithContext,
    DocumentDescriptor,
    Stage5Result,
    SummaryNode,
)

logger = logging.getLogger(__name__)

_SUMMARIZE_SYSTEM = "You are a document analysis assistant. Return only valid JSON, no markdown. Use null if unsure."


class ChunkingStage:
    stage_name = "chunking"
    stage_version = "1.0"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    async def run(
        self,
        s1: Stage1Result,
        s4: Stage4Result,
        *,
        visual_descriptions: list[VisualDescription],
    ) -> Stage5Result:
        logger.info("Stage 5 — Chunking: %s block(s)", len(s4.ordered_blocks))
        target_chars = self.settings.preprocess_chunk_size
        overlap_chars = self.settings.preprocess_chunk_overlap
        region_by_id: dict[str, LayoutRegion] = {r.region_id: r for r in s1.regions}

        chunks = self._build_all_chunks(s1, s4, target_chars, overlap_chars, region_by_id)

        skip_llm = self.settings.fast_mode or not self.settings.use_document_intelligence
        if skip_llm or not self.settings.openai_api_key:
            descriptor = DocumentDescriptor(
                chunk_size=target_chars,
                overlap=overlap_chars,
            )
            return Stage5Result(
                document_id=s1.document_id,
                chunks=chunks,
                document_descriptor=descriptor,
                fast_mode_active=self.settings.fast_mode,
            )

        summary_tree, descriptor, embedding = await self._build_summaries(s4, chunks)

        # Possibly refine chunk params from adaptive strategy
        if self.settings.use_adaptive_chunking:
            target_chars = descriptor.chunk_size or target_chars
            overlap_chars = descriptor.overlap or overlap_chars
            if (
                target_chars != self.settings.preprocess_chunk_size
                or overlap_chars != self.settings.preprocess_chunk_overlap
            ):
                logger.info(
                    "Adaptive chunking: re-chunking with chunk_size=%s overlap=%s",
                    target_chars,
                    overlap_chars,
                )
                chunks = self._build_all_chunks(s1, s4, target_chars, overlap_chars, region_by_id)

        return Stage5Result(
            document_id=s1.document_id,
            chunks=chunks,
            summary_tree=summary_tree,
            document_descriptor=descriptor,
            summary_embedding=embedding,
        )

    def cache_key(self, s1: Stage1Result, s4: Stage4Result) -> str:
        return StageCache.compute_key(
            s1.document_id,
            self.stage_name,
            self.stage_version,
            str(self.settings.preprocess_chunk_size),
            str(self.settings.preprocess_chunk_overlap),
            str(self.settings.fast_mode),
            self.settings.openai_model,
            str(self.settings.use_document_intelligence),
        )

    # ── Chunking ───────────────────────────────────────────────────────────────

    def _build_all_chunks(
        self,
        s1: Stage1Result,
        s4: Stage4Result,
        target_chars: int,
        overlap_chars: int,
        region_by_id: dict[str, LayoutRegion],
    ) -> list[ChunkWithContext]:
        blocks_by_page: dict[int, list[OrderedTextBlock]] = {}
        for block in s4.ordered_blocks:
            blocks_by_page.setdefault(block.page_number, []).append(block)

        block_to_node: dict[str, str] = {}
        for node in s4.all_nodes.values():
            for bid in node.block_ids:
                block_to_node[bid] = node.node_id

        chunks: list[ChunkWithContext] = []
        next_index = 1
        for page_number, blocks in sorted(blocks_by_page.items()):
            current: list[OrderedTextBlock] = []
            for block in blocks:
                if current and len("\n\n".join(b.text for b in current + [block])) > target_chars:
                    chunk = self._make_chunk(
                        s1.document_id,
                        s1.source_filename,
                        page_number,
                        next_index,
                        current,
                        region_by_id,
                        block_to_node,
                        s4.all_nodes,
                    )
                    chunks.append(chunk)
                    next_index += 1
                    current = _overlap_blocks(current, overlap_chars)
                current.append(block)
            if current:
                chunk = self._make_chunk(
                    s1.document_id,
                    s1.source_filename,
                    page_number,
                    next_index,
                    current,
                    region_by_id,
                    block_to_node,
                    s4.all_nodes,
                )
                chunks.append(chunk)
                next_index += 1
        return chunks

    def _make_chunk(
        self,
        document_id: str,
        source_filename: str,
        page_number: int,
        index: int,
        blocks: list[OrderedTextBlock],
        region_by_id: dict[str, LayoutRegion],
        block_to_node: dict[str, str],
        all_nodes: dict[str, HierarchyNode],
    ) -> ChunkWithContext:
        chunk_id = f"{document_id}:chunk:{index}"
        text = "\n\n".join(b.text for b in blocks if b.text.strip()).strip()
        region_ids = sorted({rid for b in blocks for rid in b.region_ids})
        crop_refs = [
            region_by_id[rid].crop_path for rid in region_ids if rid in region_by_id and region_by_id[rid].crop_path
        ]
        region_types = sorted({region_by_id[rid].region_type for rid in region_ids if rid in region_by_id})
        bbox_refs = [b.bbox.as_list() for b in blocks if b.bbox is not None]
        item_ids = [iid for b in blocks for iid in b.item_ids]
        ordered_block_ids = [b.block_id for b in blocks]

        # Resolve parent context from first block's hierarchy node
        parent_title: str | None = None
        parent_subtitle: str | None = None
        hierarchy_node_id: str | None = None
        for bid in ordered_block_ids:
            nid = block_to_node.get(bid)
            if nid:
                node = all_nodes.get(nid)
                if node:
                    parent_title = node.title
                    parent_subtitle = node.subtitle
                    hierarchy_node_id = nid
                    break

        # Fall back to region metadata if no node resolved
        if parent_title is None:
            for rid in region_ids:
                region = region_by_id.get(rid)
                if region:
                    pt = region.metadata.get("parent_title")
                    if pt is not None:
                        parent_title = str(pt)
                        ps = region.metadata.get("parent_subtitle")
                        parent_subtitle = str(ps) if ps else None
                        break

        metadata: dict[str, Any] = {
            "document_id": document_id,
            "source_filename": source_filename,
            "page_number": page_number,
            "chunk_id": chunk_id,
            "ordered_block_ids": ordered_block_ids,
            "item_ids": item_ids,
            "region_ids": region_ids,
            "region_types": region_types,
            "bbox_references": bbox_refs,
            "crop_references": [r for r in crop_refs if r],
            # These keys are read by rag/hybrid.py, rerank.py, compress.py
            "parent_title": parent_title,
            "parent_subtitle": parent_subtitle,
        }

        return ChunkWithContext(
            chunk_id=chunk_id,
            text=text,
            page_content=text,
            page_number=page_number,
            ordered_block_ids=ordered_block_ids,
            item_ids=item_ids,
            source_region_ids=region_ids,
            region_types=region_types,
            bbox_references=bbox_refs,
            crop_references=[r for r in crop_refs if r],
            metadata=metadata,
            hierarchy_node_id=hierarchy_node_id,
            parent_title=parent_title,
            parent_subtitle=parent_subtitle,
        )

    # ── LLM summarization ──────────────────────────────────────────────────────

    async def _build_summaries(
        self,
        s4: Stage4Result,
        chunks: list[ChunkWithContext],
    ) -> tuple[dict[str, SummaryNode], DocumentDescriptor, list[float]]:
        semaphore = asyncio.Semaphore(self.settings.llm_concurrency_limit)
        summary_tree: dict[str, SummaryNode] = {}

        section_nodes = [s4.all_nodes[nid] for nid in s4.sections if nid in s4.all_nodes]
        tasks = [self._summarize_node(node, [], semaphore) for node in section_nodes]
        section_results = await asyncio.gather(*tasks)
        for node_result in section_results:
            summary_tree[node_result.node_id] = node_result

        # Document descriptor from aggregated section summaries
        descriptor = await asyncio.get_event_loop().run_in_executor(
            None,
            self._generate_descriptor_sync,
            section_nodes,
            list(summary_tree.values()),
            s4,
        )

        # Summary embedding
        embedding: list[float] = []
        if descriptor.summary:
            try:
                from rag.embed import build_embedding_backend

                backend = build_embedding_backend(self.settings)
                embeddings = backend.embed_texts([descriptor.summary])
                if embeddings:
                    embedding = embeddings[0]
            except Exception as exc:
                logger.warning("Failed to embed document summary: %s", exc)

        return summary_tree, descriptor, embedding

    async def _summarize_node(
        self,
        node: HierarchyNode,
        child_summaries: list[SummaryNode],
        semaphore: asyncio.Semaphore,
    ) -> SummaryNode:
        async with semaphore:
            try:
                return await asyncio.get_event_loop().run_in_executor(
                    None, self._summarize_node_sync, node, child_summaries
                )
            except Exception as exc:
                logger.warning("Section summarization failed for %s: %s", node.node_id, exc)
                return SummaryNode(
                    node_id=node.node_id,
                    level=node.level,
                    title=node.title,
                    summary_text="",
                )

    def _summarize_node_sync(
        self,
        node: HierarchyNode,
        child_summaries: list[SummaryNode],
    ) -> SummaryNode:
        from document_Process.clients import OpenAIJSONModelClient

        client = OpenAIJSONModelClient(
            model=self.settings.descriptor_model,
            api_key=self.settings.openai_api_key or "",
            base_url=self.settings.openai_base_url,
        )
        few_shot = (
            "Examples:\n"
            '1. Plain introduction: {"summary": "Overview of the report scope and methodology.", '
            '"key_topics": ["scope", "methodology"], "has_data": false, "section_type": "introduction"}\n'
            '2. Data-heavy analysis: {"summary": "Revenue analysis showing 23% YoY growth.", '
            '"key_topics": ["revenue", "growth"], "has_data": true, "section_type": "analysis"}\n'
        )
        user_prompt = (
            f"Section title: {node.title}\n"
            f"Text preview:\n{node.flat_text[:800]}\n\n"
            f"{few_shot}"
            "Return JSON with fields: summary, key_topics (list), has_data (bool), section_type (str)."
        )
        raw = client.generate_text(
            system_prompt=_SUMMARIZE_SYSTEM,
            user_prompt=user_prompt,
        )
        result = _safe_parse_json(raw)
        return SummaryNode(
            node_id=node.node_id,
            level=node.level,
            title=node.title,
            summary_text=str(result.get("summary") or ""),
            key_topics=list(result.get("key_topics") or []),
            has_data=bool(result.get("has_data", False)),
            section_type=str(result.get("section_type") or "unknown"),
        )

    def _generate_descriptor_sync(
        self,
        section_nodes: list[HierarchyNode],
        section_summaries: list[SummaryNode],
        s4: Stage4Result,
    ) -> DocumentDescriptor:
        if not self.settings.openai_api_key:
            return DocumentDescriptor(
                chunk_size=self.settings.preprocess_chunk_size,
                overlap=self.settings.preprocess_chunk_overlap,
            )
        try:
            from document_Process.clients import OpenAIJSONModelClient

            client = OpenAIJSONModelClient(
                model=self.settings.descriptor_model,
                api_key=self.settings.openai_api_key,
                base_url=self.settings.openai_base_url,
            )
            structure_lines = [f"[{s.title}] {s.summary_text[:100]}" for s in section_summaries]
            user_prompt = (
                f"Section count: {len(section_nodes)}\n"
                f"Document structure:\n{chr(10).join(structure_lines)}\n\n"
                "Return JSON with: summary, topics (list), doc_type, domain, "
                "well_structured (bool), visual_heavy (bool), data_heavy (bool), "
                "likely_questions (list), chunk_strategy, chunk_size (int), "
                "overlap (int), keep_tables_intact (bool), strategy_reason."
            )
            raw = client.generate_text(
                system_prompt=_SUMMARIZE_SYSTEM,
                user_prompt=user_prompt,
            )
            result = _safe_parse_json(raw)
            return DocumentDescriptor(
                summary=str(result.get("summary") or ""),
                topics=list(result.get("topics") or []),
                doc_type=str(result.get("doc_type") or "unknown"),
                domain=str(result.get("domain") or "unknown"),
                well_structured=bool(result.get("well_structured", True)),
                visual_heavy=bool(result.get("visual_heavy", False)),
                data_heavy=bool(result.get("data_heavy", False)),
                likely_questions=list(result.get("likely_questions") or []),
                chunk_strategy=str(result.get("chunk_strategy") or "semantic_fixed"),
                chunk_size=int(result.get("chunk_size") or self.settings.preprocess_chunk_size),
                overlap=int(result.get("overlap") or self.settings.preprocess_chunk_overlap),
                keep_tables_intact=bool(result.get("keep_tables_intact", False)),
                strategy_reason=str(result.get("strategy_reason") or ""),
            )
        except Exception as exc:
            logger.warning("Document descriptor generation failed: %s", exc)
            return DocumentDescriptor(
                chunk_size=self.settings.preprocess_chunk_size,
                overlap=self.settings.preprocess_chunk_overlap,
            )


# ── Private helpers ────────────────────────────────────────────────────────────


def _overlap_blocks(blocks: list[OrderedTextBlock], overlap_chars: int) -> list[OrderedTextBlock]:
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


# ── Compatibility shim (preserves old build_chunks() interface for tests) ─────


def build_chunks(
    *,
    document_id: str,
    source_file: str,
    ordered_blocks: list[OrderedTextBlock],
    regions: list[LayoutRegion],
    target_chars: int = 1800,
    overlap_chars: int = 200,
) -> list[ProcessedChunk]:
    region_by_id: dict[str, LayoutRegion] = {r.region_id: r for r in regions}
    block_to_node: dict[str, str] = {}
    all_nodes: dict[str, Any] = {}

    stage = ChunkingStage.__new__(ChunkingStage)

    blocks_by_page: dict[int, list[OrderedTextBlock]] = {}
    for block in ordered_blocks:
        blocks_by_page.setdefault(block.page_number, []).append(block)

    chunks: list[ChunkWithContext] = []
    next_index = 1
    for page_number, blocks in sorted(blocks_by_page.items()):
        current: list[OrderedTextBlock] = []
        for block in blocks:
            if current and len("\n\n".join(b.text for b in current + [block])) > target_chars:
                chunk = stage._make_chunk(
                    document_id,
                    source_file,
                    page_number,
                    next_index,
                    current,
                    region_by_id,
                    block_to_node,
                    all_nodes,
                )
                chunks.append(chunk)
                next_index += 1
                current = _overlap_blocks(current, overlap_chars)
            current.append(block)
        if current:
            chunk = stage._make_chunk(
                document_id,
                source_file,
                page_number,
                next_index,
                current,
                region_by_id,
                block_to_node,
                all_nodes,
            )
            chunks.append(chunk)
            next_index += 1
    return chunks
