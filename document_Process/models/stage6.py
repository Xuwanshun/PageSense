from __future__ import annotations

from pydantic import BaseModel, Field


class ExportManifest(BaseModel):
    schema_version: str = "7.0.0"
    pipeline_stage: str = "preprocessing"
    processing_status: str = "completed"
    document_id: str
    source_filename: str
    source_path: str
    working_dir: str
    page_count: int
    chunk_count: int
    processing_timestamp: str
    fast_mode: bool = False
    artifacts: dict[str, str] = Field(default_factory=dict)
