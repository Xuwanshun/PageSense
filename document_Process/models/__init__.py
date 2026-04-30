"""
Re-exports all public models so existing import paths remain valid.

    from document_Process.models import ProcessedChunk   # still works
    from document_Process.models import Stage1Result     # new
"""

from document_Process.models.base import BoundingBox, ProcessingIssue, RegionType
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
from document_Process.models.stage1 import PageContext, Stage1Result
from document_Process.models.stage2 import (
    OrderedItem,
    PageReadingOrder,
    Stage2Result,
    TextDirection,
)
from document_Process.models.stage3 import Stage3Result, VisualDescription, VLMDetailLevel
from document_Process.models.stage4 import HierarchyLevel, HierarchyNode, Stage4Result
from document_Process.models.stage5 import (
    ChunkWithContext,
    DocumentDescriptor,
    Stage5Result,
    SummaryNode,
)
from document_Process.models.stage6 import ExportManifest

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
    # stage1
    "PageContext",
    "Stage1Result",
    # stage2
    "TextDirection",
    "OrderedItem",
    "PageReadingOrder",
    "Stage2Result",
    # stage3
    "VLMDetailLevel",
    "VisualDescription",
    "Stage3Result",
    # stage4
    "HierarchyLevel",
    "HierarchyNode",
    "Stage4Result",
    # stage5
    "ChunkWithContext",
    "SummaryNode",
    "DocumentDescriptor",
    "Stage5Result",
    # stage6
    "ExportManifest",
]
