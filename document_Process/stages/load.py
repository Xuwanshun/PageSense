"""Stage 1 — Load & Detect.

Accepts PDF or image. Assigns document_id (SHA-256). Renders pages to images.
Runs OCR (PaddleOCR PP-OCRv5) and layout detection (PP-DocLayout_plus-L) on
each page. Crops visual regions (table / figure) for downstream stages.
"""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
from functools import lru_cache
from pathlib import Path
from typing import Any

from config import Settings
from document_Process.models.base import BoundingBox, ProcessingIssue
from document_Process.models.internal import LoadResult, PageResult
from document_Process.models.legacy import (
    CroppedRegionAsset,
    LayoutRegion,
    OCRPageResult,
    OCRTextItem,
)

SUPPORTED_SUFFIXES = {".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
TEXT_BLOCK_LABELS = {
    "text",
    "title",
    "doc_title",
    "figure_title",
    "paragraph_title",
    "header",
    "footer",
    "reference",
    "caption",
    "list",
    "number",
    "formula_caption",
    "table_caption",
    "figure_caption",
    "aside_text",
}
FIGURE_LABELS = {"image", "figure", "chart", "graph"}

logger = logging.getLogger(__name__)


def _configure_paddle_env(cache_dir: Path) -> None:
    cache_home = cache_dir.resolve()
    cache_home.mkdir(parents=True, exist_ok=True)
    (cache_home / "temp").mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("PADDLE_PDX_CACHE_HOME", str(cache_home))
    os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")


@lru_cache(maxsize=1)
def _get_paddle_ocr() -> Any:
    from paddleocr import PaddleOCR  # type: ignore

    return PaddleOCR(
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        text_detection_model_name="PP-OCRv5_mobile_det",
        text_recognition_model_name="PP-OCRv5_mobile_rec",
    )


@lru_cache(maxsize=1)
def _get_paddle_layout_detector() -> Any:
    from paddleocr import LayoutDetection  # type: ignore

    return LayoutDetection()


class LoadStage:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        _configure_paddle_env(settings.paddle_cache_dir)

    def run(self, source_path: Path, *, document_id: str | None = None) -> LoadResult:
        logger.info("Stage 1 — Load & Detect: %s", source_path)
        if source_path.suffix.lower() not in SUPPORTED_SUFFIXES:
            raise ValueError(
                f"Unsupported document type: {source_path.suffix or 'no extension'}"
            )

        resolved_id = document_id or self._build_document_id(source_path)
        working_dir = self.settings.processed_documents_dir / resolved_id
        if working_dir.exists():
            shutil.rmtree(working_dir)
        source_dir = working_dir / "source"
        pages_dir = source_dir / "pages"
        source_dir.mkdir(parents=True, exist_ok=True)
        pages_dir.mkdir(parents=True, exist_ok=True)

        original_copy_path = source_dir / source_path.name
        if source_path.resolve() != original_copy_path.resolve():
            shutil.copy2(source_path, original_copy_path)

        if original_copy_path.suffix.lower() == ".pdf":
            pages = _load_pdf_pages(
                original_copy_path,
                pages_dir,
                render_scale=self.settings.pdf_render_scale,
            )
        else:
            pages = [_load_image_page(original_copy_path, page_number=1)]

        issues: list[ProcessingIssue] = []
        pdf_path = (
            original_copy_path if original_copy_path.suffix.lower() == ".pdf" else None
        )
        ocr_pages = self._run_ocr(pages, issues, pdf_path=pdf_path)
        regions = self._run_layout(pages, issues)
        self._crop_visual_regions(pages, regions, working_dir / "crops", issues)

        return LoadResult(
            document_id=resolved_id,
            source_filename=source_path.name,
            source_path=str(source_path),
            working_dir=working_dir,
            original_copy_path=original_copy_path,
            page_count=len(pages),
            pages=pages,
            ocr_pages=ocr_pages,
            regions=regions,
            issues=issues,
        )

    def _build_document_id(self, path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _run_ocr(
        self,
        pages: list[PageResult],
        issues: list[ProcessingIssue],
        pdf_path: Path | None = None,
    ) -> list[OCRPageResult]:
        logger.info("Running PaddleOCR on %s page(s)", len(pages))
        ocr = _get_paddle_ocr()
        results: list[OCRPageResult] = []
        for page in pages:
            # Try direct text extraction from the PDF text layer before falling back to OCR.
            # Machine-readable PDFs produce cleaner text; OCR introduces noise on them.
            if pdf_path is not None:
                try:
                    import pdfplumber  # type: ignore

                    scale = self.settings.pdf_render_scale
                    with pdfplumber.open(pdf_path) as plumb_pdf:
                        plumb_page = plumb_pdf.pages[page.page_number - 1]
                        words = plumb_page.extract_words() or []
                    nonws_chars = sum(len(w["text"].replace(" ", "")) for w in words)
                    if nonws_chars > 50:
                        pdf_items: list[OCRTextItem] = []
                        for idx, word in enumerate(words, start=1):
                            cleaned = str(word["text"]).strip()
                            if not cleaned:
                                continue
                            bbox = BoundingBox(
                                x0=float(word["x0"]) * scale,
                                y0=float(word["top"]) * scale,
                                x1=float(word["x1"]) * scale,
                                y1=float(word["bottom"]) * scale,
                            )
                            if not bbox.is_valid():
                                continue
                            pdf_items.append(
                                OCRTextItem(
                                    item_id=f"p{page.page_number}_pdf_{idx}",
                                    page_number=page.page_number,
                                    text=cleaned,
                                    bbox=bbox,
                                    confidence=1.0,
                                    source="pdfplumber",
                                )
                            )
                        if pdf_items:
                            results.append(
                                OCRPageResult(
                                    page_number=page.page_number,
                                    width=page.width,
                                    height=page.height,
                                    items=pdf_items,
                                    text_source="pdfplumber",
                                    page_image_path=str(page.page_image_path),
                                )
                            )
                            continue
                except Exception as exc:
                    logger.debug(
                        "pdfplumber text extraction failed for page %d, falling back to PaddleOCR: %s",
                        page.page_number,
                        exc,
                    )

            try:
                payload = ocr.predict(str(page.page_image_path))[0].json["res"]
            except Exception as exc:
                raise RuntimeError(
                    "PaddleOCR text extraction failed. Ensure paddlepaddle, paddleocr, and paddlex[ocr] are installed."
                ) from exc

            items: list[OCRTextItem] = []
            rec_texts = payload.get("rec_texts") or []
            rec_scores = payload.get("rec_scores") or []
            rec_boxes = payload.get("rec_boxes") or []
            dt_polys = payload.get("dt_polys") or []
            for index, text in enumerate(rec_texts, start=1):
                cleaned = str(text).strip()
                if not cleaned:
                    continue
                bbox = _bbox_from_ocr_payload(rec_boxes, dt_polys, index - 1)
                if bbox is None or not bbox.is_valid():
                    continue
                score = rec_scores[index - 1] if index - 1 < len(rec_scores) else None
                items.append(
                    OCRTextItem(
                        item_id=f"p{page.page_number}_ocr_{index}",
                        page_number=page.page_number,
                        text=cleaned,
                        bbox=bbox,
                        confidence=float(score) if score is not None else None,
                        source="paddleocr",
                    )
                )
            if not items:
                issues.append(
                    ProcessingIssue(
                        code="ocr_no_text",
                        message="PaddleOCR did not return any text for this page.",
                        level="warning",
                        page_number=page.page_number,
                    )
                )
            results.append(
                OCRPageResult(
                    page_number=page.page_number,
                    width=page.width,
                    height=page.height,
                    items=items,
                    text_source="paddleocr_ppocrv5_mobile",
                    page_image_path=str(page.page_image_path),
                )
            )
        return results

    def _run_layout(
        self, pages: list[PageResult], issues: list[ProcessingIssue]
    ) -> list[LayoutRegion]:
        logger.info("Running Paddle layout detection on %s page(s)", len(pages))
        detector = _get_paddle_layout_detector()
        regions: list[LayoutRegion] = []
        next_id = 1
        for page in pages:
            try:
                payload = detector.predict(str(page.page_image_path))[0].json["res"]
            except Exception as exc:
                raise RuntimeError(
                    "Paddle layout detection failed. Ensure paddlepaddle, paddleocr, and paddlex[ocr] are installed."
                ) from exc

            page_regions: list[LayoutRegion] = []
            for box in payload.get("boxes") or []:
                label = str(box.get("label") or "").strip().lower()
                region_type = _region_type_for_label(label)
                if region_type is None:
                    continue
                bbox = _bbox_from_layout_box(box.get("coordinate"))
                if bbox is None or not bbox.is_valid():
                    continue
                page_regions.append(
                    LayoutRegion(
                        region_id=f"region_{next_id}",
                        region_type=region_type,
                        page_number=page.page_number,
                        bbox=bbox,
                        confidence=float(box["score"])
                        if box.get("score") is not None
                        else None,
                        source="paddle_layout_detection",
                        metadata={"detector": "PP-DocLayout_plus-L", "label": label},
                    )
                )
                next_id += 1
            regions.extend(page_regions)

        if not regions:
            issues.append(
                ProcessingIssue(
                    code="layout_no_regions",
                    message="Paddle layout detection did not return any supported regions.",
                    level="warning",
                )
            )
        return _dedupe_regions(regions)

    def _crop_visual_regions(
        self,
        pages: list[PageResult],
        regions: list[LayoutRegion],
        crops_dir: Path,
        issues: list[ProcessingIssue],
    ) -> list[CroppedRegionAsset]:
        try:
            from PIL import Image  # type: ignore
        except Exception as exc:
            issues.append(
                ProcessingIssue(
                    code="crop_unavailable",
                    message="Pillow is required for region cropping.",
                    level="warning",
                    details={"error": str(exc)},
                )
            )
            return []

        page_lookup = {page.page_number: page for page in pages}
        for folder in ("tables", "figures"):
            (crops_dir / folder).mkdir(parents=True, exist_ok=True)

        assets: list[CroppedRegionAsset] = []
        for region in regions:
            if region.region_type not in {"table", "figure"}:
                continue
            page = page_lookup.get(region.page_number)
            if page is None or not page.page_image_path.exists():
                issues.append(
                    ProcessingIssue(
                        code="missing_page_image",
                        message="Skipping crop because the rendered page image is missing.",
                        level="warning",
                        page_number=region.page_number,
                        details={"region_id": region.region_id},
                    )
                )
                continue

            folder = "tables" if region.region_type == "table" else "figures"
            crop_path = crops_dir / folder / f"{region.region_id}.png"
            try:
                with Image.open(page.page_image_path) as image:
                    crop_box = _compute_crop_box(region, image.width, image.height)
                    if crop_box is None:
                        issues.append(
                            ProcessingIssue(
                                code="invalid_crop_bounds",
                                message="Skipping crop because the padded crop bounds are invalid.",
                                level="warning",
                                page_number=region.page_number,
                                details={"region_id": region.region_id},
                            )
                        )
                        continue
                    image.crop(crop_box).save(crop_path)
            except Exception as exc:
                issues.append(
                    ProcessingIssue(
                        code="crop_open_failed",
                        message="Skipping crop because the page image could not be opened.",
                        level="warning",
                        page_number=region.page_number,
                        details={"region_id": region.region_id, "error": str(exc)},
                    )
                )
                continue

            region.crop_path = str(crop_path)
            assets.append(
                CroppedRegionAsset(
                    asset_id=f"asset_{region.region_id}",
                    region_id=region.region_id,
                    page_number=region.page_number,
                    region_type=region.region_type,
                    crop_path=str(crop_path),
                    bbox=region.bbox,
                )
            )
        return assets


# ── Private helpers ────────────────────────────────────────────────────────────


def _load_pdf_pages(
    path: Path, pages_dir: Path, *, render_scale: float
) -> list[PageResult]:
    try:
        import pypdfium2 as pdfium  # type: ignore
    except Exception as exc:
        raise RuntimeError("PDF rendering requires pypdfium2.") from exc

    pdf = pdfium.PdfDocument(str(path))
    pages: list[PageResult] = []
    try:
        for page_index in range(len(pdf)):
            page_number = page_index + 1
            pdfium_page = pdf[page_index]
            bitmap = pdfium_page.render(scale=render_scale)
            image = bitmap.to_pil()
            image_path = pages_dir / f"page_{page_number}.png"
            image.save(image_path)
            width, height = image.size
            pages.append(
                PageResult(
                    page_number=page_number,
                    width=float(width),
                    height=float(height),
                    page_image_path=image_path,
                )
            )
    finally:
        pdf.close()
    return pages


def _load_image_page(path: Path, *, page_number: int) -> PageResult:
    try:
        from PIL import Image  # type: ignore
    except Exception as exc:
        raise RuntimeError("Image input requires Pillow.") from exc

    with Image.open(path) as image:
        width, height = image.size
    return PageResult(
        page_number=page_number,
        width=float(width),
        height=float(height),
        page_image_path=path,
    )


def _bbox_from_ocr_payload(
    rec_boxes: list[Any], dt_polys: list[Any], index: int
) -> BoundingBox | None:
    if index < len(rec_boxes):
        value = rec_boxes[index]
        if isinstance(value, list) and len(value) == 4:
            return BoundingBox.from_list([float(item) for item in value])
    if index < len(dt_polys):
        points = dt_polys[index]
        if isinstance(points, list) and points:
            xs = [float(point[0]) for point in points]
            ys = [float(point[1]) for point in points]
            return BoundingBox(x0=min(xs), y0=min(ys), x1=max(xs), y1=max(ys))
    return None


def _bbox_from_layout_box(value: Any) -> BoundingBox | None:
    if not isinstance(value, list) or len(value) != 4:
        return None
    return BoundingBox.from_list([float(item) for item in value])


def _region_type_for_label(label: str) -> str | None:
    if label == "table" or "table" in label:
        return "table"
    if label in TEXT_BLOCK_LABELS or label.endswith("_text"):
        return "text_block"
    if label in FIGURE_LABELS:
        return "figure"
    return None


def _dedupe_regions(regions: list[LayoutRegion]) -> list[LayoutRegion]:
    seen: set[tuple[int, str, tuple[float, ...]]] = set()
    deduped: list[LayoutRegion] = []
    for region in regions:
        key = (
            region.page_number,
            region.region_type,
            tuple(round(v, 1) for v in region.bbox.as_list()),
        )
        if key not in seen:
            seen.add(key)
            deduped.append(region)
    return deduped


def _compute_crop_box(
    region: LayoutRegion, image_width: int, image_height: int
) -> tuple[int, int, int, int] | None:
    width = region.bbox.x1 - region.bbox.x0
    height = region.bbox.y1 - region.bbox.y0
    if width <= 1 or height <= 1:
        return None

    if region.region_type == "table":
        pad_x = max(28, int(width * 0.05))
        pad_y = max(28, int(height * 0.08))
    else:
        pad_x = max(36, int(width * 0.08))
        pad_y = max(36, int(height * 0.12))

    left = max(0, int(region.bbox.x0 - pad_x))
    top = max(0, int(region.bbox.y0 - pad_y))
    right = min(image_width, int(region.bbox.x1 + pad_x))
    bottom = min(image_height, int(region.bbox.y1 + pad_y))
    if right - left < 48 or bottom - top < 48:
        return None
    return (left, top, right, bottom)
