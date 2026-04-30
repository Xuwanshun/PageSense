from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from document_Process.models.base import ProcessingIssue

TextDirection = Literal["ltr", "rtl", "ttb"]


class OrderedItem(BaseModel):
    item_id: str
    page_number: int
    reading_order: int
    text_direction: TextDirection = "ltr"
    column_index: int | None = None


class PageReadingOrder(BaseModel):
    page_number: int
    ordered_item_ids: list[str]
    detected_columns: int = 1
    dominant_direction: TextDirection = "ltr"


class Stage2Result(BaseModel):
    document_id: str
    resolver: str = "line_bucket_v2"
    items: list[OrderedItem]
    pages: list[PageReadingOrder]
    document_order_item_ids: list[str]
    issues: list[ProcessingIssue] = Field(default_factory=list)
    stage_version: str = "1.0"
