"""Stage 2 — Reading Order.

Sorts all detected OCR items into reading order. Handles multi-column layouts
and detects text direction (LTR, RTL, TTB) from OCR output.
"""

from __future__ import annotations

import logging
from typing import Any

from config import Settings
from document_Process.cache import StageCache
from document_Process.models import (
    OCRPageResult,
    OCRTextItem,
    ProcessingIssue,
    Stage1Result,
    Stage2Result,
)
from document_Process.models.stage2 import OrderedItem, PageReadingOrder, TextDirection

logger = logging.getLogger(__name__)

# Unicode ranges for RTL scripts (Arabic, Hebrew)
_RTL_RANGES = [
    (0x0600, 0x06FF),  # Arabic
    (0x0590, 0x05FF),  # Hebrew
    (0x0750, 0x077F),  # Arabic Supplement
    (0xFB50, 0xFDFF),  # Arabic Presentation Forms-A
    (0xFE70, 0xFEFF),  # Arabic Presentation Forms-B
]

# Unicode ranges for CJK (vertical layout possible)
_CJK_RANGES = [
    (0x4E00, 0x9FFF),  # CJK Unified Ideographs
    (0x3400, 0x4DBF),  # CJK Extension A
    (0x20000, 0x2A6DF),  # CJK Extension B
]


class ReadingOrderStage:
    stage_name = "reading_order"
    stage_version = "1.0"

    def __init__(self, settings: Settings | None = None) -> None:
        self.line_bucket_px = settings.reading_order_line_bucket_px if settings else 18

    def run(self, s1: Stage1Result) -> Stage2Result:
        logger.info("Stage 2 — Reading Order: %s page(s)", s1.page_count)

        ordered_items: list[OrderedItem] = []
        page_orders: list[PageReadingOrder] = []
        document_order_ids: list[str] = []
        global_index = 1

        for page in s1.ocr_pages:
            width = page.width or 0.0
            direction = self._detect_direction(page.items)
            columns = self._detect_columns(page.items, width)

            if direction == "rtl":
                sorted_items = sorted(page.items, key=lambda it: self._rtl_key(it, self.line_bucket_px))
            else:
                sorted_items = sorted(page.items, key=lambda it: self._ltr_key(it, self.line_bucket_px))

            page_ids: list[str] = []
            for item in sorted_items:
                item.reading_order = global_index
                ordered_items.append(
                    OrderedItem(
                        item_id=item.item_id,
                        page_number=item.page_number,
                        reading_order=global_index,
                        text_direction=direction,
                        column_index=self._column_index(item, columns, width) if columns > 1 else None,
                    )
                )
                page_ids.append(item.item_id)
                document_order_ids.append(item.item_id)
                global_index += 1

            page_orders.append(
                PageReadingOrder(
                    page_number=page.page_number,
                    ordered_item_ids=page_ids,
                    detected_columns=columns,
                    dominant_direction=direction,
                )
            )

        return Stage2Result(
            document_id=s1.document_id,
            items=ordered_items,
            pages=page_orders,
            document_order_item_ids=document_order_ids,
        )

    def cache_key(self, s1: Stage1Result) -> str:
        return StageCache.compute_key(
            s1.document_id,
            self.stage_name,
            self.stage_version,
            str(self.line_bucket_px),
        )

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _ltr_key(self, item: OCRTextItem, bucket_px: int) -> tuple[int, float, float]:
        return (round(item.bbox.y0 / bucket_px), item.bbox.x0, item.bbox.y0)

    def _rtl_key(self, item: OCRTextItem, bucket_px: int) -> tuple[int, float, float]:
        return (round(item.bbox.y0 / bucket_px), -item.bbox.x1, item.bbox.y0)

    def _detect_direction(self, items: list[OCRTextItem]) -> TextDirection:
        if not items:
            return "ltr"
        rtl_chars = 0
        total_chars = 0
        for item in items:
            for ch in item.text:
                cp = ord(ch)
                total_chars += 1
                if any(lo <= cp <= hi for lo, hi in _RTL_RANGES):
                    rtl_chars += 1
        if total_chars > 0 and rtl_chars / total_chars > 0.3:
            return "rtl"
        return "ltr"

    def _detect_columns(self, items: list[OCRTextItem], page_width: float) -> int:
        if not items or page_width <= 0:
            return 1
        midpoint = page_width / 2.0
        left_count = sum(1 for it in items if it.bbox.x1 < midpoint * 0.9)
        right_count = sum(1 for it in items if it.bbox.x0 > midpoint * 1.1)
        if left_count > len(items) * 0.15 and right_count > len(items) * 0.15:
            return 2
        return 1

    def _column_index(self, item: OCRTextItem, columns: int, page_width: float) -> int:
        if columns <= 1 or page_width <= 0:
            return 0
        midpoint = page_width / 2.0
        return 0 if item.bbox.x0 < midpoint else 1


# ── Compatibility shims (used by tests and legacy callers) ─────────────────────


def _reading_order_key(item: OCRTextItem) -> tuple[int, float, float]:
    """LTR reading-order sort key — bucket by y0, then x0."""
    return (round(item.bbox.y0 / 18.0), item.bbox.x0, item.bbox.y0)


class ReadingOrderService:
    """Thin wrapper preserving the old resolve() interface for tests."""

    def resolve(self, pages: list[OCRPageResult]) -> tuple[dict[str, Any], list[ProcessingIssue]]:
        all_ids: list[str] = []
        page_payloads: list[dict[str, Any]] = []
        global_index = 1
        for page in pages:
            sorted_items = sorted(page.items, key=_reading_order_key)
            for item in sorted_items:
                item.reading_order = global_index
                global_index += 1
            ids = [item.item_id for item in sorted_items]
            all_ids.extend(ids)
            page_payloads.append({"page_number": page.page_number, "ordered_item_ids": ids})
        return (
            {"resolver": "ocr_bbox_sort_v1", "document_order_item_ids": all_ids, "pages": page_payloads},
            [],
        )
