"""
Legacy models preserved verbatim for rag/ backward compatibility.

These must not be modified — rag/chunk.py, rag/retrieve.py, and rag/qa.py
deserialize JSON artifacts into these exact shapes.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from document_Process.models.base import BoundingBox, ProcessingIssue, RegionType


class OCRTextItem(BaseModel):
    item_id: str
    page_number: int
    text: str
    bbox: BoundingBox
    confidence: float | None = None
    source: str = "paddleocr"
    region_id: str | None = None
    block_id: str | None = None
    reading_order: int | None = None


class OCRPageResult(BaseModel):
    page_number: int
    width: float | None = None
    height: float | None = None
    items: list[OCRTextItem] = Field(default_factory=list)
    text_source: str = "paddleocr"
    page_image_path: str | None = None


class LayoutRegion(BaseModel):
    region_id: str
    region_type: RegionType
    page_number: int
    bbox: BoundingBox
    confidence: float | None = None
    crop_path: str | None = None
    source: str = "paddle_layout_detection"
    metadata: dict[str, Any] = Field(default_factory=dict)


class OrderedTextBlock(BaseModel):
    block_id: str
    page_number: int
    text: str
    item_ids: list[str] = Field(default_factory=list)
    region_ids: list[str] = Field(default_factory=list)
    bbox: BoundingBox | None = None
    reading_order: int


class RegionAssociation(BaseModel):
    association_id: str
    page_number: int
    item_id: str
    block_id: str | None = None
    region_id: str | None = None
    region_type: str | None = None
    overlap_ratio: float = 0.0


class CroppedRegionAsset(BaseModel):
    asset_id: str
    region_id: str
    page_number: int
    region_type: RegionType
    crop_path: str
    bbox: BoundingBox


class VisualRegionSummary(BaseModel):
    summary_id: str
    region_id: str
    asset_id: str | None = None
    page_number: int
    region_type: RegionType
    crop_path: str | None = None
    linked_block_ids: list[str] = Field(default_factory=list)
    linked_chunk_ids: list[str] = Field(default_factory=list)
    summary_text: str
    is_meaningful: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)


class ProcessedChunk(BaseModel):
    chunk_id: str
    text: str
    page_content: str
    page_number: int | None = None
    ordered_block_ids: list[str] = Field(default_factory=list)
    item_ids: list[str] = Field(default_factory=list)
    source_region_ids: list[str] = Field(default_factory=list)
    region_types: list[str] = Field(default_factory=list)
    bbox_references: list[list[float]] = Field(default_factory=list)
    crop_references: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ProcessingMetadata(BaseModel):
    processing_timestamp: str
    schema_version: str
    ocr_engine: str
    reading_order_model: str
    layout_detection_model: str
    agent_model: str | None = None
    confidence_summary: dict[str, Any] = Field(default_factory=dict)
    warnings: list[ProcessingIssue] = Field(default_factory=list)
    errors: list[ProcessingIssue] = Field(default_factory=list)


class ProcessedDocument(BaseModel):
    document_id: str
    source_filename: str
    source_path: str
    page_count: int
    full_ordered_text: str
    region_summaries: list[dict[str, Any]] = Field(default_factory=list)
    cropped_assets: list[dict[str, Any]] = Field(default_factory=list)
    crop_references: list[str] = Field(default_factory=list)
    processing_summary: dict[str, Any] = Field(default_factory=dict)
    agent_input: dict[str, Any] = Field(default_factory=dict)
    agent_output: dict[str, Any] = Field(default_factory=dict)


class ProcessedManifest(BaseModel):
    schema_version: str
    pipeline_stage: Literal["preprocessing"]
    processing_status: Literal["completed"]
    document_id: str
    source_filename: str
    source_path: str
    working_dir: str
    page_count: int
    chunk_count: int
    processing_timestamp: str
    artifacts: dict[str, str] = Field(default_factory=dict)
