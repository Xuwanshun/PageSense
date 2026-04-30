"""
document_Process pipeline — PDF/image → frozen artifacts
=========================================================

Six-stage flow (DocumentPreprocessingPipeline.run):

  1. LoadDetectStage       — SHA-256 ID, PDF render, OCR, layout detection, region cropping
  2. ReadingOrderStage     — LTR/RTL/TTB sort with multi-column detection
  3. VisualUnderstandingStage — async VLM descriptions with tiered detail
  4. HierarchyStage        — title propagation, block grouping, section tree
  5. ChunkingStage         — async hierarchical chunking + LLM summarization
  6. ExportStage           — all artifacts to data/processed/<document_id>/

Idempotency: preprocess_document() short-circuits when all artifacts already
exist (unless force=True).  Per-stage caching (StageCache) skips individual
stages when their inputs have not changed.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from pathlib import Path

from config import Settings
from document_Process.cache import StageCache
from document_Process.stages.stage1_load_detect import LoadDetectStage
from document_Process.stages.stage2_reading_order import ReadingOrderStage
from document_Process.stages.stage3_visual import VisualUnderstandingStage
from document_Process.stages.stage4_hierarchy import HierarchyStage
from document_Process.stages.stage5_chunking import ChunkingStage
from document_Process.stages.stage6_export import ExportStage

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PreprocessingResult:
    document_id: str
    working_dir: Path
    document_json_path: Path
    structured_json_path: Path
    page_count: int
    chunk_count: int
    warnings: list[str]


class DocumentPreprocessingPipeline:
    """Wires the six preprocessing stages with per-stage caching."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.stage1 = LoadDetectStage(settings)
        self.stage2 = ReadingOrderStage(settings)
        self.stage3 = VisualUnderstandingStage(settings)
        self.stage4 = HierarchyStage()
        self.stage5 = ChunkingStage(settings)
        self.stage6 = ExportStage()

    def run(self, source_path: Path, *, document_id: str | None = None) -> PreprocessingResult:
        logger.info("Starting document preprocessing: %s", source_path)

        # Stage 1 (sync) — always runs first to establish working_dir and document_id
        s1 = self.stage1.run(source_path, document_id=document_id)
        cache = StageCache(s1.working_dir) if self.settings.stage_cache_enabled else _NoCache()

        # Stage 2 (sync)
        s2_key = self.stage2.cache_key(s1)
        if cache.is_hit(self.stage2.stage_name, s2_key):
            logger.info("Stage 2 cache hit — skipping reading order")
            # Re-run cheaply (no Paddle calls) to rebuild in-memory result
            s2 = self.stage2.run(s1)
        else:
            s2 = self.stage2.run(s1)
            cache.write(self.stage2.stage_name, s2_key)

        # Stage 3 (async)
        s3_key = self.stage3.cache_key(s1, s2)
        if cache.is_hit(self.stage3.stage_name, s3_key):
            logger.info("Stage 3 cache hit — skipping VLM")
            s3 = asyncio.run(self.stage3.run(s1, s2))
        else:
            s3 = asyncio.run(self.stage3.run(s1, s2))
            cache.write(self.stage3.stage_name, s3_key)

        # Stage 4 (sync)
        s4_key = self.stage4.cache_key(s1, s2, s3)
        if cache.is_hit(self.stage4.stage_name, s4_key):
            logger.info("Stage 4 cache hit — skipping hierarchy")
            s4 = self.stage4.run(s1, s2, s3)
        else:
            s4 = self.stage4.run(s1, s2, s3)
            cache.write(self.stage4.stage_name, s4_key)

        # Stage 5 (async)
        s5_key = self.stage5.cache_key(s1, s4)
        if cache.is_hit(self.stage5.stage_name, s5_key):
            logger.info("Stage 5 cache hit — skipping chunking/summarization")
            s5 = asyncio.run(self.stage5.run(s1, s4, visual_descriptions=s3.visual_descriptions))
        else:
            s5 = asyncio.run(self.stage5.run(s1, s4, visual_descriptions=s3.visual_descriptions))
            cache.write(self.stage5.stage_name, s5_key)

        # Stage 6 (sync)
        s6_key = self.stage6.cache_key(s1, s5)
        if cache.is_hit(self.stage6.stage_name, s6_key):
            logger.info("Stage 6 cache hit — skipping export")
            self.stage6.run(s1, s2, s3, s4, s5)
        else:
            self.stage6.run(s1, s2, s3, s4, s5)
            cache.write(self.stage6.stage_name, s6_key)

        warnings = [i.message for i in (s1.issues + s4.issues + s5.issues) if i.level == "warning"]
        logger.info(
            "Finished preprocessing %s: %s page(s), %s chunk(s)",
            s1.document_id,
            s1.page_count,
            len(s5.chunks),
        )
        return PreprocessingResult(
            document_id=s1.document_id,
            working_dir=s1.working_dir,
            document_json_path=s1.working_dir / "document.json",
            structured_json_path=s1.working_dir / "structured.json",
            page_count=s1.page_count,
            chunk_count=len(s5.chunks),
            warnings=warnings,
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

    # Compute document_id without loading Paddle (cheap SHA-256 hash)
    loader = LoadDetectStage(resolved_settings)
    resolved_id = document_id or loader._build_document_id(source_path)
    working_dir = resolved_settings.processed_documents_dir / resolved_id
    manifest_path = working_dir / "manifest.json"
    chunks_path = working_dir / "chunks.json"
    document_path = working_dir / "document.json"

    if not force and manifest_path.exists() and chunks_path.exists() and document_path.exists():
        logger.info("Reusing frozen preprocessing artifacts for %s", source_path)
        chunk_count = len(json.loads(chunks_path.read_text(encoding="utf-8")))
        manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        return PreprocessingResult(
            document_id=resolved_id,
            working_dir=working_dir,
            document_json_path=document_path,
            structured_json_path=working_dir / "structured.json",
            page_count=int(manifest_payload.get("page_count") or 0),
            chunk_count=chunk_count,
            warnings=[],
        )

    pipeline = DocumentPreprocessingPipeline(resolved_settings)
    return pipeline.run(source_path, document_id=resolved_id)


class _NoCache:
    """Drop-in replacement for StageCache when stage_cache_enabled=False."""

    def is_hit(self, stage_name: str, key: str) -> bool:
        return False

    def write(self, stage_name: str, key: str) -> None:
        pass
