"""document_Process pipeline — PDF/image → frozen artifacts.

Five-stage flow (DocumentPipeline.run):

  1. LoadStage       — SHA-256 ID, PDF render, OCR, layout detection, region cropping
  2. OrderStage      — LTR/RTL sort with multi-column detection
  3. VisualStage     — VLM figure descriptions (placeholder by default)
  4. HierarchyStage  — title propagation, block grouping, Document→Section→Block tree
  5. SummarizeStage  — chunking + optional LLM section/document summarization

Output artifacts (data/processed/<document_id>/):
  document.json       — ProcessedDocument (consumed by rag/retrieve.py)
  chunks.json         — list[ProcessedChunk] (consumed by rag/retrieve.py)
  manifest.json       — lightweight metadata

Idempotency: preprocess_document() short-circuits when all three artifacts
already exist (unless force=True).
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from config import Settings
from document_Process.models.internal import (
    HierarchyResult,
    LoadResult,
    SummarizeResult,
)
from document_Process.models.legacy import (
    CroppedRegionAsset,
    ProcessedDocument,
)
from document_Process.stages.hierarchy import HierarchyStage
from document_Process.stages.load import LoadStage
from document_Process.stages.order import OrderStage
from document_Process.stages.summarize import SummarizeStage
from document_Process.stages.visual import VisualStage

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ProcessingResult:
    document_id: str
    working_dir: Path
    page_count: int
    chunk_count: int
    warnings: list[str]


# Keep old name as an alias so rag/ imports that use PreprocessingResult still work
PreprocessingResult = ProcessingResult


class DocumentPipeline:
    """Wires the five preprocessing stages."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._load = LoadStage(settings)
        self._order = OrderStage(settings)
        self._visual = VisualStage(settings)
        self._hierarchy = HierarchyStage()
        self._summarize = SummarizeStage(settings)

    def run(
        self, source_path: Path, *, document_id: str | None = None
    ) -> ProcessingResult:
        if self.settings.fast_mode or not self.settings.use_document_intelligence:
            logger.warning(
                "FAST_MODE or USE_DOCUMENT_INTELLIGENCE=false is active. "
                "Section embeddings will be title-only (no summaries). Pool A retrieval quality "
                "will degrade for sections with short or ambiguous titles (e.g. '3.1', 'Introduction'). "
                "Set USE_DOCUMENT_INTELLIGENCE=true for production use."
            )
        logger.info("Starting document preprocessing: %s", source_path)

        load = self._load.run(source_path, document_id=document_id)
        order = self._order.run(load)
        visual = self._visual.run(load)
        hier = self._hierarchy.run(load, order, visual)
        summ = self._summarize.run(load, hier)

        _export(load.working_dir, load, hier, summ)
        (load.working_dir / "done").touch()

        warnings = [i.message for i in load.issues if i.level == "warning"]
        logger.info(
            "Finished preprocessing %s: %s page(s), %s chunk(s)",
            load.document_id,
            load.page_count,
            len(summ.chunks),
        )
        return ProcessingResult(
            document_id=load.document_id,
            working_dir=load.working_dir,
            page_count=load.page_count,
            chunk_count=len(summ.chunks),
            warnings=warnings,
        )


# Keep old class name for any callers that used DocumentPreprocessingPipeline
DocumentPreprocessingPipeline = DocumentPipeline


def preprocess_document(
    source_name_or_path: str | Path,
    *,
    settings: Settings | None = None,
    document_id: str | None = None,
    force: bool = False,
) -> ProcessingResult:
    resolved_settings = settings or Settings()
    source_path = Path(source_name_or_path)
    if not source_path.is_absolute() and source_path.parent == Path("."):
        source_path = resolved_settings.raw_documents_dir / source_path

    loader = LoadStage(resolved_settings)
    resolved_id = document_id or loader._build_document_id(source_path)
    working_dir = resolved_settings.processed_documents_dir / resolved_id
    chunks_path = working_dir / "chunks.json"
    document_path = working_dir / "document.json"
    manifest_path = working_dir / "manifest.json"

    if (
        not force
        and manifest_path.exists()
        and chunks_path.exists()
        and document_path.exists()
    ):
        logger.info("Reusing frozen preprocessing artifacts for %s", source_path)
        chunk_count = len(json.loads(chunks_path.read_text(encoding="utf-8")))
        manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        return ProcessingResult(
            document_id=resolved_id,
            working_dir=working_dir,
            page_count=int(manifest_payload.get("page_count") or 0),
            chunk_count=chunk_count,
            warnings=[],
        )

    pipeline = DocumentPipeline(resolved_settings)
    return pipeline.run(source_path, document_id=resolved_id)


# ── Export ────────────────────────────────────────────────────────────────────


def _export(
    working_dir: Path,
    load: LoadResult,
    hier: HierarchyResult,
    summ: SummarizeResult,
) -> None:
    timestamp = datetime.now(UTC).isoformat().replace("+00:00", "Z")

    cropped_assets = _collect_cropped_assets(load)

    # Build ProcessedDocument (legacy shape consumed by rag/retrieve.py)
    full_text = "\n\n".join(
        b.text for b in hier.ordered_blocks if b.text.strip()
    ).strip()
    processed_doc = ProcessedDocument(
        document_id=load.document_id,
        source_filename=load.source_filename,
        source_path=str(load.original_copy_path),
        page_count=load.page_count,
        full_ordered_text=full_text,
        region_summaries=[
            {
                "region_id": r.region_id,
                "region_type": r.region_type,
                "page_number": r.page_number,
                "bbox": r.bbox.as_list(),
                "crop_path": r.crop_path,
                "detector": r.metadata.get("detector"),
                "label": r.metadata.get("label"),
                "confidence": r.confidence,
            }
            for r in load.regions
        ],
        cropped_assets=[a.model_dump(mode="json") for a in cropped_assets],
        crop_references=[a.crop_path for a in cropped_assets],
        processing_summary={
            "page_count": load.page_count,
            "region_count": len(load.regions),
            "cropped_asset_count": len(cropped_assets),
            "chunk_count": len(summ.chunks),
        },
    )

    doc_payload = processed_doc.model_dump(mode="json")
    doc_payload.pop("agent_input", None)
    doc_payload.pop("agent_output", None)
    if summ.document_summary:
        doc_payload["descriptor"] = {"summary": summ.document_summary}

    _write_json(working_dir / "document.json", doc_payload)
    _write_json(
        working_dir / "chunks.json", [c.model_dump(mode="json") for c in summ.chunks]
    )
    _write_json(
        working_dir / "manifest.json",
        {
            "schema_version": "8.0.0",
            "pipeline_stage": "preprocessing",
            "processing_status": "completed",
            "document_id": load.document_id,
            "source_filename": load.source_filename,
            "source_path": load.source_path,
            "working_dir": str(working_dir),
            "page_count": load.page_count,
            "chunk_count": len(summ.chunks),
            "document_summary": summ.document_summary or "",
            "processing_timestamp": timestamp,
            "artifacts": {
                "document": "document.json",
                "chunks": "chunks.json",
                "manifest": "manifest.json",
            },
        },
    )


def _collect_cropped_assets(load: LoadResult) -> list[CroppedRegionAsset]:
    assets = []
    for region in load.regions:
        if region.crop_path:
            assets.append(
                CroppedRegionAsset(
                    asset_id=f"asset_{region.region_id}",
                    region_id=region.region_id,
                    page_number=region.page_number,
                    region_type=region.region_type,
                    crop_path=region.crop_path,
                    bbox=region.bbox,
                )
            )
    return assets


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _strip_tags(text: str) -> str:
    return re.sub(r"<[^>]+>", "", text).strip()
