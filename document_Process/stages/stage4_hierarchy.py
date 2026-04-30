"""Stage 4 — Hierarchy Assignment & Section Grouping.

Stamps each OCR block with its parent title/subtitle context, then groups
blocks into a hierarchical tree: document → sections → subsections → blocks.

For well-structured documents (papers, reports) the heading detector uses
layout label hints ("title", "doc_title", "paragraph_title") to identify
section boundaries. Unstructured documents fall back to line-bucket grouping.
"""

from __future__ import annotations

import logging
from typing import Any

from document_Process.cache import StageCache
from document_Process.models import (
    BoundingBox,
    LayoutRegion,
    OCRPageResult,
    OCRTextItem,
    OrderedTextBlock,
    RegionAssociation,
    Stage1Result,
    Stage2Result,
    Stage3Result,
    Stage4Result,
)
from document_Process.models.stage4 import HierarchyLevel, HierarchyNode

logger = logging.getLogger(__name__)

_TITLE_LABELS = {"title", "doc_title", "paragraph_title"}
_SUBTITLE_LABELS = {"subtitle"}


class HierarchyStage:
    stage_name = "hierarchy"
    stage_version = "1.0"

    def run(self, s1: Stage1Result, s2: Stage2Result, s3: Stage3Result) -> Stage4Result:
        logger.info("Stage 4 — Hierarchy: %s page(s)", s1.page_count)

        item_lookup: dict[str, OCRTextItem] = {item.item_id: item for page in s1.ocr_pages for item in page.items}
        region_by_id: dict[str, LayoutRegion] = {r.region_id: r for r in s1.regions}

        # ── Step A: stamp parent_title / parent_subtitle onto each region ─────
        self._assign_titles(s1.regions, item_lookup)

        # ── Build ordered blocks (absorbs AssociationService) ─────────────────
        associations, ordered_blocks = self._build_blocks(s1, s2, item_lookup, region_by_id)

        # ── Step B: build hierarchy tree ──────────────────────────────────────
        root, all_nodes = self._build_hierarchy(ordered_blocks, s1.regions)
        section_ids = [nid for nid, node in all_nodes.items() if node.level == HierarchyLevel.SECTION]

        return Stage4Result(
            document_id=s1.document_id,
            ordered_blocks=ordered_blocks,
            hierarchy_root=root,
            all_nodes=all_nodes,
            sections=section_ids,
            region_associations=associations,
        )

    def cache_key(self, s1: Stage1Result, s2: Stage2Result, s3: Stage3Result) -> str:
        return StageCache.compute_key(
            s1.document_id,
            self.stage_name,
            self.stage_version,
            s3.stage_version,
        )

    # ── Step A: title propagation ──────────────────────────────────────────────

    def _assign_titles(
        self,
        regions: list[LayoutRegion],
        item_lookup: dict[str, OCRTextItem],
    ) -> None:
        block_text_by_region: dict[str, str] = {}
        for item in item_lookup.values():
            if item.region_id:
                existing = block_text_by_region.get(item.region_id, "")
                block_text_by_region[item.region_id] = (existing + " " + item.text).strip() if existing else item.text

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

    # ── Build ordered blocks (replaces AssociationService) ────────────────────

    def _build_blocks(
        self,
        s1: Stage1Result,
        s2: Stage2Result,
        item_lookup: dict[str, OCRTextItem],
        region_by_id: dict[str, LayoutRegion],
    ) -> tuple[list[RegionAssociation], list[OrderedTextBlock]]:
        regions_by_page: dict[int, list[LayoutRegion]] = {}
        for region in s1.regions:
            regions_by_page.setdefault(region.page_number, []).append(region)

        associations: list[RegionAssociation] = []
        ordered_blocks: list[OrderedTextBlock] = []
        global_index = 1

        for page_order in s2.pages:
            page_number = page_order.page_number
            page_items = [item_lookup[iid] for iid in page_order.ordered_item_ids if iid in item_lookup]
            page_regions = regions_by_page.get(page_number, [])
            page_blocks: list[OrderedTextBlock] = []
            current_items: list[OCRTextItem] = []
            current_region_id: str | None = None
            current_line_bucket: int | None = None

            for item in page_items:
                matched_region, overlap_ratio = _best_region_match(item, page_regions)
                item.region_id = matched_region.region_id if matched_region else None
                line_bucket = int(item.bbox.y0 // 20)
                associations.append(
                    RegionAssociation(
                        association_id=f"assoc_{len(associations) + 1}",
                        page_number=page_number,
                        item_id=item.item_id,
                        region_id=item.region_id,
                        region_type=matched_region.region_type if matched_region else None,
                        overlap_ratio=round(overlap_ratio, 4),
                    )
                )

                group_region_id = item.region_id
                if current_items and (group_region_id != current_region_id or line_bucket != current_line_bucket):
                    global_index = _flush_block(page_number, current_items, global_index, page_blocks, ordered_blocks)
                    current_items = []

                current_items.append(item)
                current_region_id = group_region_id
                current_line_bucket = line_bucket

            if current_items:
                global_index = _flush_block(page_number, current_items, global_index, page_blocks, ordered_blocks)

            if not page_blocks and page_items:
                fallback = _build_fallback_blocks(page_number, page_items, global_index)
                ordered_blocks.extend(fallback)
                global_index += len(fallback)
                for block in fallback:
                    for item_id in block.item_ids:
                        if item_id in item_lookup:
                            item_lookup[item_id].block_id = block.block_id

        return associations, ordered_blocks

    # ── Step B: hierarchy tree ─────────────────────────────────────────────────

    def _build_hierarchy(
        self,
        blocks: list[OrderedTextBlock],
        regions: list[LayoutRegion],
    ) -> tuple[HierarchyNode, dict[str, HierarchyNode]]:
        region_by_id: dict[str, LayoutRegion] = {r.region_id: r for r in regions}

        doc_node = HierarchyNode(
            node_id="doc_root",
            level=HierarchyLevel.DOCUMENT,
            title="Document",
            page_start=1,
            page_end=max((b.page_number for b in blocks), default=1),
        )
        all_nodes: dict[str, HierarchyNode] = {"doc_root": doc_node}

        section_map: dict[str, str] = {}  # parent_title → section node_id
        section_order: list[str] = []

        for block in blocks:
            region_ids = block.region_ids
            parent_title = "untitled"
            parent_subtitle: str | None = None
            for rid in region_ids:
                region = region_by_id.get(rid)
                if region:
                    pt = region.metadata.get("parent_title")
                    if pt is not None:
                        parent_title = str(pt)
                        parent_subtitle = region.metadata.get("parent_subtitle")
                        break

            if parent_title not in section_map:
                section_node_id = f"section_{len(section_map)}"
                section_node = HierarchyNode(
                    node_id=section_node_id,
                    level=HierarchyLevel.SECTION,
                    title=parent_title,
                    subtitle=parent_subtitle,
                    page_start=block.page_number,
                    page_end=block.page_number,
                    parent_node_id="doc_root",
                )
                section_map[parent_title] = section_node_id
                section_order.append(section_node_id)
                all_nodes[section_node_id] = section_node
                doc_node.child_node_ids.append(section_node_id)

            sec_id = section_map[parent_title]
            sec_node = all_nodes[sec_id]
            sec_node.page_end = max(sec_node.page_end, block.page_number)
            sec_node.block_ids.append(block.block_id)
            for rid in region_ids:
                if rid not in sec_node.region_ids:
                    sec_node.region_ids.append(rid)
            sec_node.flat_text = (sec_node.flat_text + " " + block.text).strip()

        doc_node.child_node_ids = section_order
        return doc_node, all_nodes


# ── Private helpers ────────────────────────────────────────────────────────────


def _best_region_match(item: OCRTextItem, regions: list[LayoutRegion]) -> tuple[LayoutRegion | None, float]:
    best: LayoutRegion | None = None
    best_ratio = 0.0
    item_area = item.bbox.area() or 1.0
    for region in regions:
        overlap = item.bbox.intersection_area(region.bbox)
        if overlap <= 0:
            continue
        ratio = overlap / item_area
        if ratio > best_ratio:
            best = region
            best_ratio = ratio
    return best, best_ratio


def _flush_block(
    page_number: int,
    items: list[OCRTextItem],
    global_index: int,
    page_blocks: list[OrderedTextBlock],
    ordered_blocks: list[OrderedTextBlock],
) -> int:
    block = _make_block(page_number, items, global_index)
    for item in items:
        item.block_id = block.block_id
    page_blocks.append(block)
    ordered_blocks.append(block)
    return global_index + 1


def _build_fallback_blocks(page_number: int, items: list[OCRTextItem], start_index: int) -> list[OrderedTextBlock]:
    blocks: list[OrderedTextBlock] = []
    current: list[OCRTextItem] = []
    current_bucket: int | None = None
    next_index = start_index
    for item in items:
        bucket = int(item.bbox.y0 // 20)
        if current and bucket != current_bucket:
            block = _make_block(page_number, current, next_index)
            for it in current:
                it.block_id = block.block_id
            blocks.append(block)
            next_index += 1
            current = []
        current.append(item)
        current_bucket = bucket
    if current:
        block = _make_block(page_number, current, next_index)
        for it in current:
            it.block_id = block.block_id
        blocks.append(block)
    return blocks


class AssociationService:
    """Thin wrapper preserving the old associate() interface for tests."""

    def associate(
        self,
        ocr_pages: list[OCRPageResult],
        reading_order: dict[str, Any],
        regions: list[LayoutRegion],
    ) -> tuple[list[RegionAssociation], list[OrderedTextBlock], dict[str, Any]]:

        item_lookup: dict[str, OCRTextItem] = {item.item_id: item for page in ocr_pages for item in page.items}
        regions_by_page: dict[int, list[LayoutRegion]] = {}
        for region in regions:
            regions_by_page.setdefault(region.page_number, []).append(region)

        associations: list[RegionAssociation] = []
        ordered_blocks: list[OrderedTextBlock] = []
        page_payloads: list[dict[str, Any]] = []
        global_index = 1

        for page_entry in reading_order.get("pages", []):
            page_number = int(page_entry["page_number"])
            page_items = [item_lookup[iid] for iid in page_entry.get("ordered_item_ids", []) if iid in item_lookup]
            page_regions = regions_by_page.get(page_number, [])
            page_blocks: list[OrderedTextBlock] = []
            current_items: list[OCRTextItem] = []
            current_region_id: str | None = None
            current_line_bucket: int | None = None

            for item in page_items:
                matched_region, overlap_ratio = _best_region_match(item, page_regions)
                item.region_id = matched_region.region_id if matched_region else None
                line_bucket = int(item.bbox.y0 // 20)
                associations.append(
                    RegionAssociation(
                        association_id=f"assoc_{len(associations) + 1}",
                        page_number=page_number,
                        item_id=item.item_id,
                        region_id=item.region_id,
                        region_type=matched_region.region_type if matched_region else None,
                        overlap_ratio=round(overlap_ratio, 4),
                    )
                )
                group_region_id = item.region_id
                if current_items and (group_region_id != current_region_id or line_bucket != current_line_bucket):
                    global_index = _flush_block(page_number, current_items, global_index, page_blocks, ordered_blocks)
                    current_items = []
                current_items.append(item)
                current_region_id = group_region_id
                current_line_bucket = line_bucket

            if current_items:
                global_index = _flush_block(page_number, current_items, global_index, page_blocks, ordered_blocks)

            if not page_blocks and page_items:
                fallback = _build_fallback_blocks(page_number, page_items, global_index)
                ordered_blocks.extend(fallback)
                global_index += len(fallback)

            page_payloads.append(
                {
                    "page_number": page_number,
                    "blocks": [b.model_dump(mode="json") for b in page_blocks],
                    "text": "\n".join(b.text for b in page_blocks if b.text.strip()).strip(),
                }
            )

        return (
            associations,
            ordered_blocks,
            {"pages": page_payloads, "full_text": "\n\n".join(p["text"] for p in page_payloads if p["text"]).strip()},
        )


def _make_block(page_number: int, items: list[OCRTextItem], reading_order: int) -> OrderedTextBlock:
    return OrderedTextBlock(
        block_id=f"p{page_number}_block_{reading_order}",
        page_number=page_number,
        text=" ".join(it.text.strip() for it in items if it.text.strip()).strip(),
        item_ids=[it.item_id for it in items],
        region_ids=sorted({it.region_id for it in items if it.region_id}),
        bbox=BoundingBox.merge([it.bbox for it in items]),
        reading_order=reading_order,
    )
