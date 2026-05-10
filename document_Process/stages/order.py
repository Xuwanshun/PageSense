"""Stage 2 — Reading Order.

Sorts all detected OCR items into reading order. Handles multi-column layouts
and detects text direction (LTR, RTL) from OCR output.
"""

from __future__ import annotations

import logging
from typing import Any

from config import Settings
from document_Process.models.internal import LoadResult, OrderResult
from document_Process.models.legacy import OCRTextItem

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


class OrderStage:
    def __init__(self, settings: Settings | None = None) -> None:
        self.line_bucket_px = settings.reading_order_line_bucket_px if settings else 18

    def run(self, load_result: LoadResult) -> OrderResult:
        logger.info("Stage 2 — Reading Order: %s page(s)", load_result.page_count)

        document_order_ids: list[str] = []
        page_orders: list[dict[str, Any]] = []
        global_index = 1

        for page in load_result.ocr_pages:
            width = page.width or 0.0
            direction = _detect_direction(page.items)
            columns = _detect_columns(page.items, width)

            if direction == "rtl":
                sorted_items = sorted(page.items, key=lambda it: _rtl_key(it, self.line_bucket_px))
            else:
                sorted_items = sorted(page.items, key=lambda it: _ltr_key(it, self.line_bucket_px))

            page_ids: list[str] = []
            for item in sorted_items:
                item.reading_order = global_index
                page_ids.append(item.item_id)
                document_order_ids.append(item.item_id)
                global_index += 1

            page_orders.append(
                {
                    "page_number": page.page_number,
                    "ordered_item_ids": page_ids,
                    "detected_columns": columns,
                    "dominant_direction": direction,
                }
            )

        return OrderResult(
            document_order_item_ids=document_order_ids,
            page_orders=page_orders,
        )


# ── Helpers ────────────────────────────────────────────────────────────────────


def _ltr_key(item: OCRTextItem, bucket_px: int) -> tuple[int, float, float]:
    return (round(item.bbox.y0 / bucket_px), item.bbox.x0, item.bbox.y0)


def _rtl_key(item: OCRTextItem, bucket_px: int) -> tuple[int, float, float]:
    return (round(item.bbox.y0 / bucket_px), -item.bbox.x1, item.bbox.y0)


def _detect_direction(items: list[OCRTextItem]) -> str:
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


def _detect_columns(items: list[OCRTextItem], page_width: float) -> int:
    if not items or page_width <= 0:
        return 1
    midpoint = page_width / 2.0
    left_count = sum(1 for it in items if it.bbox.x1 < midpoint * 0.9)
    right_count = sum(1 for it in items if it.bbox.x0 > midpoint * 1.1)
    if left_count > len(items) * 0.15 and right_count > len(items) * 0.15:
        return 2
    return 1


def _column_index(item: OCRTextItem, columns: int, page_width: float) -> int:
    if columns <= 1 or page_width <= 0:
        return 0
    midpoint = page_width / 2.0
    return 0 if item.bbox.x0 < midpoint else 1


