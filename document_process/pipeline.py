from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from config import Settings
from document_process.models import ProcessingIssue
from document_process.services import (
    AssociationService,
    CroppingService,
    DocumentLoaderService,
    LayoutDetectionService,
    OCRService,
    ReadingOrderService,
    build_chunks,
    build_document_artifacts,
    build_visual_summaries,
    export_artifacts,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PreprocessingResult:
    document_id: str
    working_dir: Path
    document_json_path: Path
    page_count: int
    chunk_count: int
    warnings: list[str]


class DocumentPreprocessingPipeline:
    """Input -> PaddleOCR -> reading order -> Paddle layout -> crop -> frozen artifacts."""

    def __init__(
        self,
        settings: Settings,
        *,
        loader: DocumentLoaderService | None = None,
        ocr: OCRService | None = None,
        reading_order: ReadingOrderService | None = None,
        layout: LayoutDetectionService | None = None,
        association: AssociationService | None = None,
        cropping: CroppingService | None = None,
    ) -> None:
        self.settings = settings
        self.loader = loader or DocumentLoaderService(settings)
        self.ocr = ocr or OCRService()
        self.reading_order = reading_order or ReadingOrderService()
        self.layout = layout or LayoutDetectionService()
        self.association = association or AssociationService()
        self.cropping = cropping or CroppingService()

    def run(self, source_path: Path, *, document_id: str | None = None) -> PreprocessingResult:
        logger.info("Starting document preprocessing for %s", source_path)
        loaded = self.loader.load(source_path, document_id=document_id)
        issues: list[ProcessingIssue] = []

        ocr_pages, ocr_issues = self.ocr.extract(loaded.pages)
        issues.extend(ocr_issues)

        reading_order, order_issues = self.reading_order.resolve(ocr_pages)
        issues.extend(order_issues)

        regions, layout_issues, layout_model = self.layout.detect(loaded.pages, ocr_pages)
        issues.extend(layout_issues)

        associations, ordered_blocks, ordered_text = self.association.associate(ocr_pages, reading_order, regions)

        cropped_assets, crop_issues = self.cropping.crop_visual_regions(
            pages=loaded.pages,
            regions=regions,
            output_dir=loaded.working_dir / "crops",
        )
        issues.extend(crop_issues)

        intel_result = None
        if (
            self.settings.use_document_intelligence
            or self.settings.use_vlm_summaries
            or self.settings.use_adaptive_chunking
        ):
            from document_process.intelligence_service import DocumentIntelligenceService

            intel_service = DocumentIntelligenceService(self.settings)
            intel_result = intel_service.process(
                regions=regions,
                document_id=loaded.document_id,
                file_name=loaded.original_copy_path.name,
                page_count=len(loaded.pages),
                ordered_blocks=ordered_blocks,
            )

        target_chars = self.settings.preprocess_chunk_size
        overlap_chars = self.settings.preprocess_chunk_overlap
        if intel_result is not None:
            target_chars = int(intel_result.strategy.get("chunk_size", target_chars))
            overlap_chars = int(intel_result.strategy.get("overlap", overlap_chars))

        chunks = build_chunks(
            document_id=loaded.document_id,
            source_file=loaded.original_copy_path.name,
            ordered_blocks=ordered_blocks,
            regions=regions,
            target_chars=target_chars,
            overlap_chars=overlap_chars,
        )

        if intel_result is not None:
            region_parent: dict[str, tuple[str, str | None]] = {
                region.region_id: (
                    str(region.metadata.get("parent_title") or ""),
                    region.metadata.get("parent_subtitle"),
                )
                for region in regions
                if region.metadata.get("parent_title") is not None
            }
            for chunk in chunks:
                for region_id in chunk.source_region_ids:
                    if region_id in region_parent:
                        parent_title, parent_subtitle = region_parent[region_id]
                        chunk.metadata["parent_title"] = parent_title
                        if parent_subtitle is not None:
                            chunk.metadata["parent_subtitle"] = parent_subtitle
                        break

        visual_summaries = build_visual_summaries(
            regions=regions,
            ordered_blocks=ordered_blocks,
            chunks=chunks,
            cropped_assets=cropped_assets,
        )

        if intel_result is not None and intel_result.visual_summaries:
            intel_vs = intel_result.visual_summaries
            visual_summaries = [
                summary.model_copy(
                    update={"metadata": {**summary.metadata, "intelligence": intel_vs[summary.crop_path]}}
                )
                if summary.crop_path and summary.crop_path in intel_vs
                else summary
                for summary in visual_summaries
            ]

        if self.settings.use_vlm_summaries:
            from document_process.vlm import enrich_summaries_with_vlm

            visual_summaries = enrich_summaries_with_vlm(visual_summaries, settings=self.settings)

        document, metadata = build_document_artifacts(
            loaded=loaded,
            ocr_pages=ocr_pages,
            ordered_text=ordered_text,
            regions=regions,
            cropped_assets=cropped_assets,
            chunks=chunks,
            reading_order_model=reading_order.get("resolver", "unknown"),
            layout_detection_model=layout_model,
            issues=issues,
        )

        document_json_path = export_artifacts(
            working_dir=loaded.working_dir,
            loaded=loaded,
            raw_ocr=ocr_pages,
            reading_order=reading_order,
            ordered_text=ordered_text,
            regions=regions,
            region_associations=associations,
            cropped_assets=cropped_assets,
            visual_summaries=visual_summaries,
            chunks=chunks,
            document=document,
            metadata=metadata,
            descriptor=intel_result.descriptor if intel_result is not None else None,
            summary_embedding=intel_result.summary_embedding if intel_result is not None else None,
        )
        logger.info(
            "Finished preprocessing document %s with %s page(s) and %s chunk(s)",
            loaded.document_id,
            len(loaded.pages),
            len(chunks),
        )
        return PreprocessingResult(
            document_id=loaded.document_id,
            working_dir=loaded.working_dir,
            document_json_path=document_json_path,
            page_count=len(loaded.pages),
            chunk_count=len(chunks),
            warnings=[issue.message for issue in issues if issue.level == "warning"],
        )


def preprocess_document(
    source_name_or_path: str | Path,
    *,
    settings: Settings | None = None,
    document_id: str | None = None,
    force: bool = False,
) -> PreprocessingResult:
    resolved_settings = settings or Settings()
    source_path = Path(source_name_or_path)
    if not source_path.is_absolute() and source_path.parent == Path("."):
        source_path = resolved_settings.raw_documents_dir / source_path
    loader = DocumentLoaderService(resolved_settings)
    resolved_id = document_id or loader._build_document_id(source_path)
    working_dir = resolved_settings.processed_documents_dir / resolved_id
    manifest_path = working_dir / "manifest.json"
    chunks_path = working_dir / "chunks.json"
    document_path = working_dir / "document.json"
    if not force and manifest_path.exists() and chunks_path.exists() and document_path.exists():
        logger.info("Reusing frozen preprocessing for %s", source_path)
        chunk_count = len(json.loads(chunks_path.read_text(encoding="utf-8")))
        metadata_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        return PreprocessingResult(
            document_id=resolved_id,
            working_dir=working_dir,
            document_json_path=document_path,
            page_count=int(metadata_payload.get("page_count") or 0),
            chunk_count=chunk_count,
            warnings=[],
        )
    pipeline = DocumentPreprocessingPipeline(resolved_settings, loader=loader)
    return pipeline.run(source_path, document_id=resolved_id)
