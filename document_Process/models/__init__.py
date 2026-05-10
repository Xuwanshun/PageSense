"""
Re-exports all public models so existing import paths remain valid.

    from document_Process.models import ProcessedChunk   # still works
    from document_Process.models import BoundingBox       # still works
"""

from document_Process.models.base import BoundingBox, ProcessingIssue, RegionType
from document_Process.models.internal import (
    Block,
    Document,
    HierarchyResult,
    LoadResult,
    OrderResult,
    PageResult,
    Section,
    SummarizeResult,
    VisualRegion,
)
from document_Process.models.legacy import (
    CroppedRegionAsset,
    LayoutRegion,
    OCRPageResult,
    OCRTextItem,
    OrderedTextBlock,
    ProcessedChunk,
    ProcessedDocument,
    ProcessedManifest,
    ProcessingMetadata,
    RegionAssociation,
    VisualRegionSummary,
)

__all__ = [
    # base
    "BoundingBox",
    "ProcessingIssue",
    "RegionType",
    # legacy (rag/ compat)
    "OCRTextItem",
    "OCRPageResult",
    "LayoutRegion",
    "OrderedTextBlock",
    "RegionAssociation",
    "CroppedRegionAsset",
    "VisualRegionSummary",
    "ProcessedChunk",
    "ProcessingMetadata",
    "ProcessedDocument",
    "ProcessedManifest",
    # internal pipeline types
    "PageResult",
    "LoadResult",
    "OrderResult",
    "VisualRegion",
    "Block",
    "Section",
    "Document",
    "HierarchyResult",
    "SummarizeResult",
]
