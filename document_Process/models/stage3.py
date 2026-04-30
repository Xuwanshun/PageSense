from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from document_Process.models.base import ProcessingIssue, RegionType

VLMDetailLevel = Literal["full", "caption", "fallback", "skipped"]


class VisualDescription(BaseModel):
    description_id: str
    region_id: str
    page_number: int
    region_type: RegionType
    crop_path: str | None = None
    detail_level: VLMDetailLevel
    inline_tag: Literal["figure", "table", "chart", "image"]
    inline_text: str
    summary: str | None = None
    key_finding: str | None = None
    data_extracted: str | None = None
    confidence: Literal["high", "medium", "low"] | None = None
    is_meaningful: bool = True
    vlm_error: str | None = None


class Stage3Result(BaseModel):
    document_id: str
    visual_descriptions: list[VisualDescription]
    enriched_text_by_page: dict[int, str] = Field(default_factory=dict)
    fast_mode_active: bool = False
    issues: list[ProcessingIssue] = Field(default_factory=list)
    stage_version: str = "1.0"
