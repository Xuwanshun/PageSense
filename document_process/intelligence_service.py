from __future__ import annotations

import base64
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from config import Settings
from document_Process.clients import OpenAIJSONModelClient
from document_Process.models import LayoutRegion, OrderedTextBlock

logger = logging.getLogger(__name__)

_TITLE_LABELS = {"title", "doc_title", "paragraph_title"}
_SUBTITLE_LABELS = {"subtitle"}
_VISUAL_REGION_TYPES = {"table", "figure"}
_CHART_LABELS = {"chart", "graph"}


@dataclass
class Section:
    title: str
    subtitle: str | None
    regions: list[LayoutRegion] = field(default_factory=list)
    text_blocks: int = 0
    tables: int = 0
    charts: int = 0
    figures: int = 0
    flat_text: str = ""
    page_start: int = 1


@dataclass
class IntelligenceResult:
    sections: list[Section]
    descriptor: dict[str, Any]
    strategy: dict[str, Any]
    visual_summaries: dict[str, dict[str, Any]]
    summary_embedding: list[float] = field(default_factory=list)


class DocumentIntelligenceService:
    """
    Adds a document intelligence layer to the preprocessing pipeline.

    Runs after layout detection and cropping; produces:
    - Title-propagated regions (parent_title / parent_subtitle stamped into metadata)
    - Section groupings with flat_text and visual counts
    - Richer structured VLM reading for table/figure regions (extends USE_VLM_SUMMARIES)
    - A document-level descriptor with a summary embedding for pre-filtering
    - An adaptive chunking strategy (when USE_ADAPTIVE_CHUNKING is enabled)
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def process(
        self,
        regions: list[LayoutRegion],
        document_id: str,
        file_name: str,
        page_count: int,
        *,
        ordered_blocks: list[OrderedTextBlock] | None = None,
    ) -> IntelligenceResult:
        blocks = ordered_blocks or []

        # Step A: stamp parent_title / parent_subtitle onto each non-title region
        self._assign_titles(regions, blocks)

        # Step B: group regions into sections
        sections = self._group_into_sections(regions, blocks)

        # Step C: richer VLM reading for visual regions (extends USE_VLM_SUMMARIES)
        visual_summaries: dict[str, dict[str, Any]] = {}
        if self.settings.use_vlm_summaries and self.settings.openai_api_key:
            for section in sections:
                for region in section.regions:
                    if region.region_type in _VISUAL_REGION_TYPES and region.crop_path:
                        crop_path = Path(region.crop_path)
                        if crop_path.exists():
                            try:
                                result = self._read_visual(region, section)
                                if result:
                                    visual_summaries[region.crop_path] = result
                            except Exception as exc:
                                logger.warning(
                                    "VLM visual reading failed for %s: %s",
                                    region.region_id,
                                    exc,
                                )

        # Step D: section and document summarization
        descriptor: dict[str, Any] = {}
        summary_embedding: list[float] = []
        if self.settings.use_document_intelligence and self.settings.openai_api_key:
            section_summaries: list[dict[str, Any]] = []
            for section in sections:
                visual_for_section = {
                    region.crop_path: visual_summaries[region.crop_path]
                    for region in section.regions
                    if region.crop_path and region.crop_path in visual_summaries
                }
                section_summaries.append(self._summarize_section(section, visual_for_section))

            descriptor = self._generate_descriptor(section_summaries, sections, page_count, file_name)

            summary_text = str(descriptor.get("summary") or "")
            if summary_text:
                try:
                    from rag.embed import build_embedding_backend

                    backend = build_embedding_backend(self.settings)
                    embeddings = backend.embed_texts([summary_text])
                    if embeddings:
                        summary_embedding = embeddings[0]
                except Exception as exc:
                    logger.warning("Failed to embed document summary for %s: %s", document_id, exc)

        # Step E: choose chunking strategy
        strategy = self._decide_strategy(descriptor, sections, page_count)

        return IntelligenceResult(
            sections=sections,
            descriptor=descriptor,
            strategy=strategy,
            visual_summaries=visual_summaries,
            summary_embedding=summary_embedding,
        )

    # ── Step A ────────────────────────────────────────────────────────────────

    def _assign_titles(
        self,
        regions: list[LayoutRegion],
        ordered_blocks: list[OrderedTextBlock],
    ) -> None:
        """Stamp parent_title / parent_subtitle into each non-title region's metadata in-place."""
        block_text_by_region: dict[str, str] = {}
        for block in ordered_blocks:
            for region_id in block.region_ids:
                existing = block_text_by_region.get(region_id, "")
                block_text_by_region[region_id] = (
                    (existing + " " + block.text).strip() if existing else block.text
                )

        current_title = "untitled"
        current_subtitle: str | None = None

        for region in sorted(regions, key=lambda r: (r.page_number, r.bbox.y0)):
            label = region.metadata.get("label", "")
            if label in _TITLE_LABELS:
                raw = block_text_by_region.get(region.region_id, "")
                current_title = raw[:200] if raw else f"section_{region.region_id}"
                current_subtitle = None
            elif label in _SUBTITLE_LABELS:
                raw = block_text_by_region.get(region.region_id, "")
                current_subtitle = raw[:200] if raw else None
            else:
                region.metadata["parent_title"] = current_title
                region.metadata["parent_subtitle"] = current_subtitle

    # ── Step B ────────────────────────────────────────────────────────────────

    def _group_into_sections(
        self,
        regions: list[LayoutRegion],
        ordered_blocks: list[OrderedTextBlock],
    ) -> list[Section]:
        """Group mutated regions by parent_title preserving document order."""
        block_text_by_region: dict[str, str] = {}
        for block in ordered_blocks:
            for region_id in block.region_ids:
                existing = block_text_by_region.get(region_id, "")
                block_text_by_region[region_id] = (
                    (existing + " " + block.text).strip() if existing else block.text
                )

        section_map: dict[str, Section] = {}
        section_order: list[str] = []

        for region in sorted(regions, key=lambda r: (r.page_number, r.bbox.y0)):
            label = region.metadata.get("label", "")
            if label in _TITLE_LABELS or label in _SUBTITLE_LABELS:
                continue

            parent_title = str(region.metadata.get("parent_title") or "untitled")
            parent_subtitle = region.metadata.get("parent_subtitle")

            if parent_title not in section_map:
                section_map[parent_title] = Section(
                    title=parent_title,
                    subtitle=parent_subtitle if isinstance(parent_subtitle, str) else None,
                    page_start=region.page_number,
                )
                section_order.append(parent_title)

            section = section_map[parent_title]
            section.regions.append(region)
            section.page_start = min(section.page_start, region.page_number)

            if region.region_type == "text_block":
                section.text_blocks += 1
            elif region.region_type == "table":
                section.tables += 1
            elif region.region_type == "figure":
                if label.lower() in _CHART_LABELS:
                    section.charts += 1
                else:
                    section.figures += 1

        for key in section_order:
            section = section_map[key]
            text_parts = [section.title]
            for region in section.regions:
                if region.region_type == "text_block":
                    region_text = block_text_by_region.get(region.region_id, "")
                    if region_text:
                        text_parts.append(region_text)
            section.flat_text = "\n".join(text_parts)[:2000]

        return [section_map[key] for key in section_order]

    # ── Step C ────────────────────────────────────────────────────────────────

    def _read_visual(self, region: LayoutRegion, section: Section) -> dict[str, Any]:
        """Call vision API for a single cropped region; return structured JSON."""
        from openai import OpenAI  # lazy import

        if not self.settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is required for VLM visual reading.")

        crop_path = Path(region.crop_path)  # type: ignore[arg-type]
        image_b64 = base64.b64encode(crop_path.read_bytes()).decode()
        context_snippet = section.flat_text[:300]

        few_shot = (
            "Examples of expected output:\n"
            '1. Simple bar chart: {"type": "figure", "summary": "Bar chart showing quarterly revenue by region.", '
            '"key_finding": "North America leads with $4.2B.", "data_extracted": "Q1 NA=$1.1B EU=$0.8B", '
            '"confidence": "high", "retrieval_mode": "text_only"}\n'
            '2. Complex table: {"type": "table", "summary": "Financial summary table covering 2019-2023.", '
            '"key_finding": "Net income doubled from 2021 to 2023.", "data_extracted": "2023 net income $2.1B", '
            '"confidence": "medium", "retrieval_mode": "text_and_image"}\n'
        )
        rules = (
            "Rules:\n"
            "- type: table / figure / chart\n"
            "- summary: 1-2 sentence retrieval-quality description\n"
            "- key_finding: the single most important piece of information\n"
            "- data_extracted: key numeric values or labels\n"
            "- confidence: high / medium / low\n"
            "- retrieval_mode: text_only / text_and_image / image_only\n"
            "Return ONLY valid JSON with exactly these 6 fields."
        )
        user_content: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": (
                    f"Document: {section.title or 'untitled'}\n"
                    f"Section subtitle: {section.subtitle or ''}\n"
                    f"Page: {region.page_number}\n"
                    f"Region type: {region.region_type}\n"
                    f"Surrounding text: {context_snippet}\n\n"
                    f"{few_shot}\n{rules}"
                ),
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{image_b64}",
                    "detail": "high",
                },
            },
        ]
        client = OpenAI(api_key=self.settings.openai_api_key, base_url=self.settings.openai_base_url)
        response = client.chat.completions.create(
            model=self.settings.vlm_model,
            messages=[
                {"role": "system", "content": "You are a document analysis assistant. Return only valid JSON."},
                {"role": "user", "content": user_content},
            ],
            max_tokens=512,
            temperature=0,
        )
        raw = (response.choices[0].message.content or "").strip()
        result = _safe_parse_json(raw)
        result["crop_path"] = region.crop_path
        result["page"] = region.page_number
        result["parent_title"] = region.metadata.get("parent_title", "untitled")
        return result

    # ── Step D ────────────────────────────────────────────────────────────────

    def _summarize_section(
        self,
        section: Section,
        visual_summaries_for_section: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        """Summarize a single section using the descriptor model."""
        client = self._build_descriptor_client()
        visual_findings = " | ".join(
            str(vs.get("key_finding") or vs.get("summary") or "")
            for vs in visual_summaries_for_section.values()
            if vs.get("key_finding") or vs.get("summary")
        )
        few_shot = (
            "Examples:\n"
            '1. Plain introduction: {"summary": "Overview of the report scope and methodology.", '
            '"key_topics": ["scope", "methodology"], "has_data": false, "section_type": "introduction"}\n'
            '2. Data-heavy analysis: {"summary": "Revenue analysis showing 23% YoY growth from APAC expansion.", '
            '"key_topics": ["revenue", "APAC", "growth"], "has_data": true, "section_type": "analysis"}\n'
        )
        user_prompt = (
            f"Section title: {section.title}\n"
            f"Visual counts — tables: {section.tables}, charts: {section.charts}, figures: {section.figures}\n"
            f"Text preview:\n{section.flat_text[:800]}\n"
            f"Visual findings: {visual_findings or 'none'}\n\n"
            f"{few_shot}"
            "Return JSON with fields: summary, key_topics (list), has_data (bool), section_type (str)."
        )
        raw = client.generate_text(
            system_prompt="You are a document analysis assistant. Return only valid JSON, no markdown. Use null if unsure.",
            user_prompt=user_prompt,
        )
        result = _safe_parse_json(raw)
        result.setdefault("summary", "")
        result.setdefault("key_topics", [])
        result.setdefault("has_data", False)
        result.setdefault("section_type", "unknown")
        return result

    def _generate_descriptor(
        self,
        section_summaries: list[dict[str, Any]],
        sections: list[Section],
        page_count: int,
        file_name: str,
    ) -> dict[str, Any]:
        """Generate a document-level descriptor from aggregated section summaries."""
        client = self._build_descriptor_client()
        total_visuals = sum(s.tables + s.charts + s.figures for s in sections)
        structure_lines: list[str] = []
        for summary, section in zip(section_summaries, sections, strict=False):
            section_type = str(summary.get("section_type") or "unknown")
            has_data = str(summary.get("has_data") or False)
            preview = str(summary.get("summary") or "")[:150]
            structure_lines.append(f"[{section.title}] type={section_type} has_data={has_data} | {preview}")

        few_shot = (
            "Examples:\n"
            '1. Academic paper: {"summary": "Research paper on neural architecture search with ablation study.", '
            '"topics": ["NAS", "deep learning"], "doc_type": "academic_paper", "domain": "computer_science", '
            '"well_structured": true, "visual_heavy": false, "data_heavy": true, '
            '"likely_questions": ["What method is proposed?", "What are the results?"]}\n'
            '2. Financial report: {"summary": "Annual financial report with revenue, expenses, and forward guidance.", '
            '"topics": ["revenue", "expenses"], "doc_type": "financial_report", "domain": "finance", '
            '"well_structured": true, "visual_heavy": true, "data_heavy": true, '
            '"likely_questions": ["What is the annual revenue?", "What is net income?"]}\n'
        )
        user_prompt = (
            f"File: {file_name}\n"
            f"Pages: {page_count}\n"
            f"Total visual regions: {total_visuals}\n"
            f"Section count: {len(sections)}\n"
            f"Document structure:\n{chr(10).join(structure_lines)}\n\n"
            f"{few_shot}"
            "Return JSON with fields: summary, topics (list), doc_type, domain, "
            "well_structured (bool), visual_heavy (bool), data_heavy (bool), likely_questions (list)."
        )
        raw = client.generate_text(
            system_prompt="You are a document analysis assistant. Return only valid JSON, no markdown. Use null if unsure.",
            user_prompt=user_prompt,
        )
        result = _safe_parse_json(raw)
        result.setdefault("summary", "")
        result.setdefault("topics", [])
        result.setdefault("doc_type", "unknown")
        result.setdefault("domain", "unknown")
        result.setdefault("well_structured", True)
        result.setdefault("visual_heavy", False)
        result.setdefault("data_heavy", False)
        result.setdefault("likely_questions", [])
        return result

    def _build_descriptor_client(self) -> OpenAIJSONModelClient:
        if not self.settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is required for document intelligence.")
        return OpenAIJSONModelClient(
            model=self.settings.descriptor_model,
            api_key=self.settings.openai_api_key,
            base_url=self.settings.openai_base_url,
        )

    # ── Step E ────────────────────────────────────────────────────────────────

    def _decide_strategy(
        self,
        descriptor: dict[str, Any],
        sections: list[Section],
        page_count: int,
    ) -> dict[str, Any]:
        """Choose a chunking strategy; apply hard overrides for table-dominant or unstructured docs."""
        if not self.settings.use_adaptive_chunking:
            return {
                "strategy": "semantic_fixed",
                "chunk_size": self.settings.preprocess_chunk_size,
                "overlap": self.settings.preprocess_chunk_overlap,
                "keep_tables_intact": False,
                "reason": "default settings — USE_ADAPTIVE_CHUNKING is disabled",
            }

        if not self.settings.openai_api_key:
            return {
                "strategy": "semantic_fixed",
                "chunk_size": self.settings.preprocess_chunk_size,
                "overlap": self.settings.preprocess_chunk_overlap,
                "keep_tables_intact": False,
                "reason": "OPENAI_API_KEY not set — using default strategy",
            }

        total_sections = len(sections)
        total_tables = sum(s.tables for s in sections)
        total_figures = sum(s.figures + s.charts for s in sections)
        total_text = sum(s.text_blocks for s in sections)
        well_structured = descriptor.get("well_structured", True)
        doc_type = str(descriptor.get("doc_type") or "unknown")
        domain = str(descriptor.get("domain") or "unknown")
        visual_heavy = descriptor.get("visual_heavy", False)
        data_heavy = descriptor.get("data_heavy", False)

        profile = (
            f"doc_type={doc_type} domain={domain} pages={page_count} sections={total_sections} "
            f"tables={total_tables} figures={total_figures} text_blocks={total_text} "
            f"visual_heavy={visual_heavy} data_heavy={data_heavy} well_structured={well_structured}"
        )
        strategy_defs = (
            "Strategy definitions:\n"
            "- semantic_section: split at section boundaries; best for well-structured documents\n"
            "- layout_aware: preserve table/figure integrity; best for data-heavy documents\n"
            "- recursive_large: large overlapping chunks; best for dense unstructured text\n"
            "- semantic_fixed: fixed-size with overlap (default fallback)\n"
        )
        few_shot = (
            "Examples:\n"
            '1. Structured academic paper → {"strategy": "semantic_section", "chunk_size": 1500, '
            '"overlap": 150, "keep_tables_intact": false, "reason": "well-structured with clear sections"}\n'
            '2. Table-heavy financial report → {"strategy": "layout_aware", "chunk_size": 2000, '
            '"overlap": 100, "keep_tables_intact": true, "reason": "many tables requiring intact preservation"}\n'
            '3. Unstructured dense report → {"strategy": "recursive_large", "chunk_size": 2500, '
            '"overlap": 400, "keep_tables_intact": false, "reason": "unstructured text needs large context"}\n'
        )
        client = self._build_descriptor_client()
        raw = client.generate_text(
            system_prompt="You are a document chunking strategist. Return only valid JSON.",
            user_prompt=(
                f"Document profile:\n{profile}\n\n"
                f"{strategy_defs}\n{few_shot}"
                "Return JSON: strategy, chunk_size (int), overlap (int), keep_tables_intact (bool), reason (str)."
            ),
        )
        result = _safe_parse_json(raw)
        result.setdefault("strategy", "semantic_fixed")
        result.setdefault("chunk_size", self.settings.preprocess_chunk_size)
        result.setdefault("overlap", self.settings.preprocess_chunk_overlap)
        result.setdefault("keep_tables_intact", False)
        result.setdefault("reason", "LLM-selected strategy")

        # Hard overrides
        if total_sections > 0 and total_tables > total_sections / 2:
            result["strategy"] = "layout_aware"
            result["keep_tables_intact"] = True
            result["override"] = "table-dominant document"
        elif not well_structured and total_sections < 3:
            result["strategy"] = "recursive_large"
            result["chunk_size"] = max(int(result.get("chunk_size") or 0), 2000)
            result["override"] = "unstructured document with few sections"

        return result


# ── Utility ───────────────────────────────────────────────────────────────────


def _safe_parse_json(text: str) -> dict[str, Any]:
    """Parse JSON from LLM output; never raises, always returns a dict."""
    stripped = re.sub(r"```(?:json)?\s*|\s*```", "", text).strip()

    try:
        result = json.loads(stripped)
        if isinstance(result, dict):
            return result
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", stripped, re.DOTALL)
    if match:
        try:
            result = json.loads(match.group())
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            pass

    recovered: dict[str, Any] = {"parse_error": True}
    for key_match in re.finditer(r'"(\w+)"\s*:\s*"([^"]*)"', stripped):
        recovered[key_match.group(1)] = key_match.group(2)
    return recovered
