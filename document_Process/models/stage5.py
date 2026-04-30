from __future__ import annotations

from pydantic import BaseModel, Field

from document_Process.models.base import ProcessingIssue
from document_Process.models.legacy import ProcessedChunk
from document_Process.models.stage4 import HierarchyLevel


class ChunkWithContext(ProcessedChunk):
    """ProcessedChunk extended with hierarchy context.

    Subclasses ProcessedChunk so rag/chunk.py's chunk_record_from_processed_chunk()
    works without modification. parent_title and parent_subtitle are written to
    both the top-level fields and the metadata dict so rag/hybrid.py can find
    them via chunk.metadata.get("parent_title").
    """

    hierarchy_node_id: str | None = None
    parent_title: str | None = None
    parent_subtitle: str | None = None


class SummaryNode(BaseModel):
    node_id: str
    level: HierarchyLevel
    title: str
    summary_text: str
    key_topics: list[str] = Field(default_factory=list)
    has_data: bool = False
    section_type: str = "unknown"


class DocumentDescriptor(BaseModel):
    summary: str = ""
    topics: list[str] = Field(default_factory=list)
    doc_type: str = "unknown"
    domain: str = "unknown"
    well_structured: bool = True
    visual_heavy: bool = False
    data_heavy: bool = False
    likely_questions: list[str] = Field(default_factory=list)
    chunk_strategy: str = "semantic_fixed"
    chunk_size: int = 1800
    overlap: int = 200
    keep_tables_intact: bool = False
    strategy_reason: str = ""


class Stage5Result(BaseModel):
    document_id: str
    chunks: list[ChunkWithContext]
    summary_tree: dict[str, SummaryNode] = Field(default_factory=dict)
    document_descriptor: DocumentDescriptor = Field(default_factory=DocumentDescriptor)
    summary_embedding: list[float] = Field(default_factory=list)
    fast_mode_active: bool = False
    issues: list[ProcessingIssue] = Field(default_factory=list)
    stage_version: str = "1.0"
