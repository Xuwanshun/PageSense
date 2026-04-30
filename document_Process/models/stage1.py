from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from document_Process.models.base import ProcessingIssue
from document_Process.models.legacy import LayoutRegion, OCRPageResult


class PageContext(BaseModel):
    page_number: int
    width: float | None = None
    height: float | None = None
    page_image_path: Path


class Stage1Result(BaseModel):
    document_id: str
    source_filename: str
    source_path: str
    working_dir: Path
    original_copy_path: Path
    page_count: int
    pages: list[PageContext]
    ocr_pages: list[OCRPageResult]
    regions: list[LayoutRegion]
    issues: list[ProcessingIssue] = Field(default_factory=list)
    stage_version: str = "1.0"
