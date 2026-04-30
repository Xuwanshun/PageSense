from __future__ import annotations

from enum import IntEnum

from pydantic import BaseModel, Field

from document_Process.models.base import ProcessingIssue
from document_Process.models.legacy import OrderedTextBlock, RegionAssociation


class HierarchyLevel(IntEnum):
    DOCUMENT = 0
    SECTION = 1
    SUBSECTION = 2
    BLOCK = 3


class HierarchyNode(BaseModel):
    node_id: str
    level: HierarchyLevel
    title: str
    subtitle: str | None = None
    page_start: int
    page_end: int
    parent_node_id: str | None = None
    child_node_ids: list[str] = Field(default_factory=list)
    block_ids: list[str] = Field(default_factory=list)
    region_ids: list[str] = Field(default_factory=list)
    flat_text: str = ""


class Stage4Result(BaseModel):
    document_id: str
    ordered_blocks: list[OrderedTextBlock]
    hierarchy_root: HierarchyNode
    all_nodes: dict[str, HierarchyNode]
    sections: list[str]
    region_associations: list[RegionAssociation]
    issues: list[ProcessingIssue] = Field(default_factory=list)
    stage_version: str = "1.0"
