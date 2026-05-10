"""Stage 4 — Hierarchy Assignment & Section Grouping.

Stamps each OCR block with its parent title/subtitle context, then groups
blocks into a clean tree: Document → Section[] → Block[].

Section boundaries are detected from layout labels ("title", "doc_title",
"paragraph_title"). Visual regions are inserted into the block stream at their
reading-order position. Each section's flat_text joins all block texts in order.
"""

from __future__ import annotations

import logging
from document_Process.models.base import BoundingBox
from document_Process.models.internal import (
    Block,
    Document,
    HierarchyResult,
    LoadResult,
    OrderResult,
    Section,
    VisualRegion,
)
from document_Process.models.legacy import (
    LayoutRegion,
    OCRTextItem,
    OrderedTextBlock,
    RegionAssociation,
)

logger = logging.getLogger(__name__)

_TITLE_LABELS = {"title", "doc_title", "paragraph_title"}
_SUBTITLE_LABELS = {"subtitle"}


class HierarchyStage:
    def run(
        self,
        load_result: LoadResult,
        order_result: OrderResult,
        visual_regions: list[VisualRegion],
    ) -> HierarchyResult:
        logger.info("Stage 4 — Hierarchy: %s page(s)", load_result.page_count)

        item_lookup: dict[str, OCRTextItem] = {
            item.item_id: item for page in load_result.ocr_pages for item in page.items
        }
        region_by_id: dict[str, LayoutRegion] = {r.region_id: r for r in load_result.regions}
        visual_by_id: dict[str, VisualRegion] = {v.region_id: v for v in visual_regions}

        # Build ordered blocks first — this sets item.region_id via bbox overlap matching,
        # which _assign_titles needs to map OCR text to title regions.
        associations, ordered_blocks = self._build_blocks(load_result, order_result, item_lookup, region_by_id)

        # Stamp parent_title / parent_subtitle on each non-title region (item.region_id now set)
        self._assign_titles(load_result.regions, item_lookup)

        # Build Document → Section → Block hierarchy
        document = self._build_hierarchy(
            ordered_blocks,
            load_result.regions,
            visual_by_id,
            load_result.document_id,
            load_result.source_filename,
            load_result.source_path,
            load_result.page_count,
        )

        return HierarchyResult(
            document=document,
            ordered_blocks=ordered_blocks,
            region_associations=associations,
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

    # ── Step B: build ordered blocks ──────────────────────────────────────────

    def _build_blocks(
        self,
        load_result: LoadResult,
        order_result: OrderResult,
        item_lookup: dict[str, OCRTextItem],
        region_by_id: dict[str, LayoutRegion],
    ) -> tuple[list[RegionAssociation], list[OrderedTextBlock]]:
        regions_by_page: dict[int, list[LayoutRegion]] = {}
        for region in load_result.regions:
            regions_by_page.setdefault(region.page_number, []).append(region)

        associations: list[RegionAssociation] = []
        ordered_blocks: list[OrderedTextBlock] = []
        global_index = 1

        for page_order in order_result.page_orders:
            page_number = int(page_order["page_number"])
            page_items = [item_lookup[iid] for iid in page_order["ordered_item_ids"] if iid in item_lookup]
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

    # ── Step C: build Document → Section → Block hierarchy ────────────────────

    def _build_hierarchy(
        self,
        ordered_blocks: list[OrderedTextBlock],
        regions: list[LayoutRegion],
        visual_by_id: dict[str, VisualRegion],
        document_id: str,
        source_filename: str,
        source_path: str,
        page_count: int,
    ) -> Document:
        region_by_id: dict[str, LayoutRegion] = {r.region_id: r for r in regions}

        # Map parent_title → Section
        section_map: dict[str, Section] = {}
        section_order: list[str] = []
        block_counter = 0

        for block in ordered_blocks:
            parent_title = "untitled"
            parent_subtitle: str | None = None
            for rid in block.region_ids:
                region = region_by_id.get(rid)
                if region:
                    pt = region.metadata.get("parent_title")
                    if pt is not None:
                        parent_title = str(pt)
                        parent_subtitle = region.metadata.get("parent_subtitle")
                        break

            if parent_title not in section_map:
                sec = Section(
                    section_id=f"section_{len(section_map)}",
                    title=parent_title,
                    subtitle=parent_subtitle,
                    page_start=block.page_number,
                    page_end=block.page_number,
                )
                section_map[parent_title] = sec
                section_order.append(parent_title)
            else:
                section_map[parent_title].page_end = max(section_map[parent_title].page_end, block.page_number)

            sec = section_map[parent_title]

            # Check if any region in this block is a visual region — insert description
            inserted_visual: set[str] = set()
            for rid in block.region_ids:
                if rid in visual_by_id and rid not in inserted_visual:
                    vr = visual_by_id[rid]
                    vblock = Block(
                        block_id=f"visual_{rid}",
                        page_number=vr.page_number,
                        text=vr.inline_text,
                        reading_order=block_counter,
                        region_ids=[rid],
                    )
                    sec.blocks.append(vblock)
                    block_counter += 1
                    inserted_visual.add(rid)

            # Add the text block itself (skip if it's a pure visual region block with no real text)
            if block.text.strip():
                tblock = Block(
                    block_id=block.block_id,
                    page_number=block.page_number,
                    text=block.text,
                    reading_order=block_counter,
                    bbox=block.bbox,
                    item_ids=list(block.item_ids),
                    region_ids=list(block.region_ids),
                )
                sec.blocks.append(tblock)
                block_counter += 1

        # Build flat_text for each section in reading order
        for sec in section_map.values():
            sec.flat_text = "\n\n".join(b.text for b in sec.blocks if b.text.strip()).strip()

        sections = [section_map[t] for t in section_order]
        return Document(
            document_id=document_id,
            source_filename=source_filename,
            source_path=source_path,
            page_count=page_count,
            sections=sections,
        )


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


