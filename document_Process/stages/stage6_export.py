"""Stage 6 — Export.

Writes all artifacts to data/processed/<document_id>/ and returns a manifest.

Required artifact files (consumed by the RAG system without modification):
  document.json        → deserializes as ProcessedDocument (rag/retrieve.py)
  chunks.json          → deserializes as list[ProcessedChunk] (rag/retrieve.py)
  visual_summaries.json → indexed by region_id (rag/qa.py)

Additional artifacts:
  manifest.json, ocr.json, layout.json, reading_order.json,
  hierarchy.json, metadata.json, structured.json
"""

from __future__ import annotations

import json
import logging
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from document_Process.cache import StageCache
from document_Process.models import (
    CroppedRegionAsset,
    LayoutRegion,
    OrderedTextBlock,
    ProcessedDocument,
    ProcessedManifest,
    ProcessingMetadata,
    Stage1Result,
    Stage2Result,
    Stage3Result,
    Stage4Result,
    Stage5Result,
    VisualRegionSummary,
)
from document_Process.models.stage6 import ExportManifest

logger = logging.getLogger(__name__)


class ExportStage:
    stage_name = "export"
    stage_version = "1.0"

    def run(
        self,
        s1: Stage1Result,
        s2: Stage2Result,
        s3: Stage3Result,
        s4: Stage4Result,
        s5: Stage5Result,
    ) -> ExportManifest:
        logger.info(
            "Stage 6 — Export: %s chunk(s) to %s",
            len(s5.chunks),
            s1.working_dir,
        )
        working_dir = s1.working_dir
        timestamp = datetime.now(UTC).isoformat().replace("+00:00", "Z")

        # ── Build legacy models required by rag/ ──────────────────────────────
        cropped_assets = self._collect_cropped_assets(s1)
        document = self._build_processed_document(s1, s4, s5, cropped_assets)
        visual_summaries = self._build_visual_summaries(s1, s3, s4, s5, cropped_assets)
        metadata = self._build_metadata(s1, s2, s4, s5, timestamp)

        # ── Write artifacts ───────────────────────────────────────────────────
        _write_json(
            working_dir / "manifest.json",
            _build_manifest_payload(
                s1,
                s5,
                timestamp,
                self.stage_version,
            ),
        )
        _write_json(working_dir / "ocr.json", [p.model_dump(mode="json") for p in s1.ocr_pages])
        _write_json(
            working_dir / "reading_order.json",
            {
                "resolver": s2.resolver,
                "document_order_item_ids": s2.document_order_item_ids,
                "pages": [p.model_dump(mode="json") for p in s2.pages],
            },
        )
        _write_json(
            working_dir / "layout.json",
            {
                "regions": [r.model_dump(mode="json") for r in s1.regions],
                "associations": [a.model_dump(mode="json") for a in s4.region_associations],
            },
        )
        _write_json(
            working_dir / "hierarchy.json",
            {
                "root": s4.hierarchy_root.model_dump(mode="json"),
                "nodes": {nid: n.model_dump(mode="json") for nid, n in s4.all_nodes.items()},
                "summary_tree": {nid: sn.model_dump(mode="json") for nid, sn in s5.summary_tree.items()},
            },
        )
        _write_json(working_dir / "visual_summaries.json", [vs.model_dump(mode="json") for vs in visual_summaries])

        # document.json: ProcessedDocument + optional descriptor + summary_embedding
        doc_payload = document.model_dump(mode="json")
        if s5.document_descriptor.summary:
            doc_payload["descriptor"] = s5.document_descriptor.model_dump(mode="json")
        if s5.summary_embedding:
            doc_payload["summary_embedding"] = s5.summary_embedding
        _write_json(working_dir / "document.json", doc_payload)

        _write_json(working_dir / "chunks.json", [c.model_dump(mode="json") for c in s5.chunks])
        _write_json(working_dir / "metadata.json", metadata.model_dump(mode="json"))
        _write_json(working_dir / "structured.json", self._build_structured(s1, s4, s5, visual_summaries))

        return ExportManifest(
            document_id=s1.document_id,
            source_filename=s1.source_filename,
            source_path=s1.source_path,
            working_dir=str(working_dir),
            page_count=s1.page_count,
            chunk_count=len(s5.chunks),
            processing_timestamp=timestamp,
            fast_mode=s5.fast_mode_active,
            artifacts={
                "document": "document.json",
                "chunks": "chunks.json",
                "visual_summaries": "visual_summaries.json",
                "manifest": "manifest.json",
                "ocr": "ocr.json",
                "layout": "layout.json",
                "reading_order": "reading_order.json",
                "hierarchy": "hierarchy.json",
                "metadata": "metadata.json",
                "structured": "structured.json",
            },
        )

    def cache_key(self, s1: Stage1Result, s5: Stage5Result) -> str:
        return StageCache.compute_key(
            s1.document_id,
            self.stage_name,
            self.stage_version,
            s5.stage_version,
        )

    # ── Build legacy ProcessedDocument (required by rag/retrieve.py) ──────────

    def _collect_cropped_assets(self, s1: Stage1Result) -> list[CroppedRegionAsset]:
        assets = []
        for region in s1.regions:
            if region.crop_path:
                assets.append(
                    CroppedRegionAsset(
                        asset_id=f"asset_{region.region_id}",
                        region_id=region.region_id,
                        page_number=region.page_number,
                        region_type=region.region_type,
                        crop_path=region.crop_path,
                        bbox=region.bbox,
                    )
                )
        return assets

    def _build_processed_document(
        self,
        s1: Stage1Result,
        s4: Stage4Result,
        s5: Stage5Result,
        cropped_assets: list[CroppedRegionAsset],
    ) -> ProcessedDocument:
        full_text = "\n\n".join(b.text for b in s4.ordered_blocks if b.text.strip()).strip()
        return ProcessedDocument(
            document_id=s1.document_id,
            source_filename=s1.source_filename,
            source_path=str(s1.original_copy_path),
            page_count=s1.page_count,
            full_ordered_text=full_text,
            region_summaries=[
                {
                    "region_id": r.region_id,
                    "region_type": r.region_type,
                    "page_number": r.page_number,
                    "bbox": r.bbox.as_list(),
                    "crop_path": r.crop_path,
                    "detector": r.metadata.get("detector"),
                    "label": r.metadata.get("label"),
                    "confidence": r.confidence,
                }
                for r in s1.regions
            ],
            cropped_assets=[a.model_dump(mode="json") for a in cropped_assets],
            crop_references=[a.crop_path for a in cropped_assets],
            processing_summary={
                "page_count": s1.page_count,
                "region_count": len(s1.regions),
                "cropped_asset_count": len(cropped_assets),
                "chunk_count": len(s5.chunks),
            },
            agent_input={},
            agent_output={},
        )

    def _build_visual_summaries(
        self,
        s1: Stage1Result,
        s3: Stage3Result,
        s4: Stage4Result,
        s5: Stage5Result,
        cropped_assets: list[CroppedRegionAsset],
    ) -> list[VisualRegionSummary]:
        asset_by_region = {a.region_id: a for a in cropped_assets}
        desc_by_region = {d.region_id: d for d in s3.visual_descriptions}
        chunks_by_region: dict[str, list[Any]] = {}
        for chunk in s5.chunks:
            for rid in chunk.source_region_ids:
                chunks_by_region.setdefault(rid, []).append(chunk)
        blocks_by_page: dict[int, list[Any]] = {}
        for block in s4.ordered_blocks:
            blocks_by_page.setdefault(block.page_number, []).append(block)

        summaries: list[VisualRegionSummary] = []
        for region in s1.regions:
            if region.region_type not in {"table", "figure"}:
                continue

            # summary_text: prefer VLM description, fall back to OCR text context
            desc = desc_by_region.get(region.region_id)
            if desc and desc.summary:
                summary_text = desc.summary
                if desc.key_finding:
                    summary_text = f"{summary_text} {desc.key_finding}"
            elif desc and desc.inline_text:
                summary_text = _strip_tags(desc.inline_text)
            else:
                page_blocks = blocks_by_page.get(region.page_number, [])
                nearby = [
                    b.text for b in page_blocks if b.bbox is not None and b.bbox.intersection_area(region.bbox) > 0
                ]
                summary_text = (
                    " ".join(nearby).strip() or f"Detected {region.region_type} region on page {region.page_number}."
                )

            region_chunks = chunks_by_region.get(region.region_id, [])
            asset = asset_by_region.get(region.region_id)
            is_meaningful = desc.is_meaningful if desc else True

            summaries.append(
                VisualRegionSummary(
                    summary_id=f"summary_{region.region_id}",
                    region_id=region.region_id,
                    asset_id=asset.asset_id if asset else None,
                    page_number=region.page_number,
                    region_type=region.region_type,
                    crop_path=asset.crop_path if asset else region.crop_path,
                    linked_block_ids=[],
                    linked_chunk_ids=[c.chunk_id for c in region_chunks],
                    summary_text=summary_text[:1200],
                    is_meaningful=is_meaningful,
                    metadata={
                        "label": region.metadata.get("label"),
                        "detector": region.metadata.get("detector"),
                        "vlm_confidence": desc.confidence if desc else None,
                    },
                )
            )
        return summaries

    def _build_metadata(
        self,
        s1: Stage1Result,
        s2: Stage2Result,
        s4: Stage4Result,
        s5: Stage5Result,
        timestamp: str,
    ) -> ProcessingMetadata:
        all_issues = s1.issues + s4.issues + s5.issues
        warnings = [i for i in all_issues if i.level == "warning"]
        errors = [i for i in all_issues if i.level == "error"]
        ocr_confidences = [
            item.confidence for page in s1.ocr_pages for item in page.items if item.confidence is not None
        ]
        region_confidences = [r.confidence for r in s1.regions if r.confidence is not None]
        return ProcessingMetadata(
            processing_timestamp=timestamp,
            schema_version="7.0.0",
            ocr_engine="PaddleOCR PP-OCRv5",
            reading_order_model=s2.resolver,
            layout_detection_model="PP-DocLayout_plus-L",
            agent_model=None,
            confidence_summary={
                "ocr_item_count": len(ocr_confidences),
                "ocr_average_confidence": round(sum(ocr_confidences) / len(ocr_confidences), 4)
                if ocr_confidences
                else None,
                "region_count": len(s1.regions),
                "region_average_confidence": round(sum(region_confidences) / len(region_confidences), 4)
                if region_confidences
                else None,
                "chunk_count": len(s5.chunks),
            },
            warnings=warnings,
            errors=errors,
        )

    def _build_structured(
        self,
        s1: Stage1Result,
        s4: Stage4Result,
        s5: Stage5Result,
        visual_summaries: list[VisualRegionSummary],
    ) -> dict[str, Any]:
        vs_lookup = {vs.region_id: vs for vs in visual_summaries}
        chunks_by_section: dict[str, list[Any]] = {}
        for chunk in s5.chunks:
            key = str(chunk.parent_title or "untitled")
            chunks_by_section.setdefault(key, []).append(chunk)

        sections = []
        seen: set[str] = set()
        for node_id in s4.sections:
            node = s4.all_nodes.get(node_id)
            if node is None or node.title in seen:
                continue
            seen.add(node.title)
            summary_node = s5.summary_tree.get(node_id)
            blocks = []
            for chunk in chunks_by_section.get(node.title, []):
                block_type = "text"
                if "table" in chunk.region_types:
                    block_type = "table"
                elif "figure" in chunk.region_types:
                    block_type = "figure"
                content = chunk.page_content or chunk.text
                for rid in chunk.source_region_ids:
                    vs = vs_lookup.get(rid)
                    if vs and vs.is_meaningful:
                        content = vs.summary_text
                        break
                blocks.append(
                    {
                        "block_id": chunk.chunk_id,
                        "type": block_type,
                        "page_number": chunk.page_number,
                        "content": content,
                    }
                )
            sections.append(
                {
                    "section_id": node_id,
                    "title": node.title,
                    "subtitle": node.subtitle,
                    "page_start": node.page_start,
                    "description": summary_node.summary_text if summary_node else "",
                    "blocks": blocks,
                }
            )

        return {
            "document_id": s1.document_id,
            "source_filename": s1.source_filename,
            "page_count": s1.page_count,
            "description": s5.document_descriptor.summary,
            "sections": sections,
        }


# ── Private helpers ────────────────────────────────────────────────────────────


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _strip_tags(text: str) -> str:
    return re.sub(r"<[^>]+>", "", text).strip()


def build_visual_summaries(
    *,
    regions: list[LayoutRegion],
    ordered_blocks: list[OrderedTextBlock],
    chunks: list[Any],
    cropped_assets: list[CroppedRegionAsset],
) -> list[VisualRegionSummary]:
    """Module-level wrapper for tests and legacy callers."""
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
        overlapping = [b for b in page_blocks if b.bbox is not None and b.bbox.intersection_area(region.bbox) > 0]
        if not overlapping:
            overlapping = sorted(
                [b for b in page_blocks if b.bbox is not None],
                key=lambda b: min(
                    abs(b.bbox.y0 - region.bbox.y1),  # type: ignore[union-attr]
                    abs(b.bbox.y1 - region.bbox.y0),  # type: ignore[union-attr]
                ),
            )[:3]
        region_chunks = chunks_by_region.get(region.region_id, [])
        block_text = " ".join(b.text for b in overlapping if b.text.strip()).strip()
        chunk_text = " ".join(c.text for c in region_chunks if c.text.strip()).strip()
        summary_text = (
            block_text or chunk_text or f"Detected {region.region_type} region on page {region.page_number}."
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


def _build_manifest_payload(
    s1: Stage1Result,
    s5: Stage5Result,
    timestamp: str,
    stage_version: str,
) -> dict[str, Any]:
    return ProcessedManifest(
        schema_version="7.0.0",
        pipeline_stage="preprocessing",
        processing_status="completed",
        document_id=s1.document_id,
        source_filename=s1.source_filename,
        source_path=s1.source_path,
        working_dir=str(s1.working_dir),
        page_count=s1.page_count,
        chunk_count=len(s5.chunks),
        processing_timestamp=timestamp,
        artifacts={
            "document": "document.json",
            "chunks": "chunks.json",
            "visual_summaries": "visual_summaries.json",
            "ocr": "ocr.json",
            "layout": "layout.json",
            "reading_order": "reading_order.json",
            "hierarchy": "hierarchy.json",
            "metadata": "metadata.json",
            "structured": "structured.json",
        },
    ).model_dump(mode="json")
