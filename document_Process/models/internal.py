"""Internal pipeline dataclasses — not consumed by rag/ directly.

These are the stage I/O types used inside document_Process/. They are plain
dataclasses (not Pydantic) so they carry zero serialization overhead. At export
time, pipeline.py converts them into the legacy Pydantic models that rag/ reads.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from document_Process.models.base import ProcessingIssue
from document_Process.models.legacy import (
    LayoutRegion,
    OCRPageResult,
    OrderedTextBlock,
    ProcessedChunk,
    RegionAssociation,
)


@dataclass
class PageResult:
    """A single rendered page (image on disk + dimensions)."""

    page_number: int
    width: float
    height: float
    page_image_path: Path


@dataclass
class LoadResult:
    """Output of Stage 1 (load + OCR + layout + crops)."""

    document_id: str
    source_filename: str
    source_path: str
    working_dir: Path
    original_copy_path: Path
    page_count: int
    pages: list[PageResult]
    ocr_pages: list[OCRPageResult]
    regions: list[LayoutRegion]
    issues: list[ProcessingIssue] = field(default_factory=list)


@dataclass
class OrderResult:
    """Output of Stage 2 (reading order)."""

    document_order_item_ids: list[str]
    page_orders: list[dict[str, Any]]


@dataclass
class VisualRegion:
    """A figure/table region with its VLM description (or placeholder)."""

    region_id: str
    page_number: int
    region_type: str
    crop_path: str | None
    inline_text: str
    summary: str | None = None
    is_meaningful: bool = True


@dataclass
class Block:
    """An ordered text/visual block within a section."""

    block_id: str
    page_number: int
    text: str
    reading_order: int
    bbox: Any | None = None
    item_ids: list[str] = field(default_factory=list)
    region_ids: list[str] = field(default_factory=list)


@dataclass
class Section:
    """Groups related blocks under a heading."""

    section_id: str
    title: str
    subtitle: str | None
    page_start: int
    page_end: int
    blocks: list[Block] = field(default_factory=list)
    flat_text: str = ""
    summary: str = ""


@dataclass
class Document:
    """Top-level document structure."""

    document_id: str
    source_filename: str
    source_path: str
    page_count: int
    sections: list[Section] = field(default_factory=list)
    summary: str = ""


@dataclass
class HierarchyResult:
    """Output of Stage 4 (hierarchy building)."""

    document: Document
    ordered_blocks: list[OrderedTextBlock]
    region_associations: list[RegionAssociation]


@dataclass
class SummarizeResult:
    """Output of Stage 5 (chunking + summarization)."""

    chunks: list[ProcessedChunk]
    document_summary: str = ""
