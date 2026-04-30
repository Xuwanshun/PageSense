"""Stage 3 — Visual Understanding.

For each non-text region (table, figure, chart, image), crops from the page
image and calls a VLM (OpenAI vision API) to produce a structured description.
Inserts descriptions inline as <figure>...</figure> or <table>...</table> tags
in the reading-order text stream.

When fast_mode=True all VLM calls are skipped and every visual region gets a
"skipped" VisualDescription.  When a VLM call fails the region gets a
"fallback" description — the pipeline never fails the whole document for one
bad region.

VLM calls are made concurrently with asyncio.Semaphore(vlm_concurrency_limit).
"""

from __future__ import annotations

import asyncio
import base64
import logging
import re
from pathlib import Path
from typing import Any

from config import Settings
from document_Process.cache import StageCache
from document_Process.models import (
    LayoutRegion,
    Stage1Result,
    Stage2Result,
    Stage3Result,
)
from document_Process.models.stage3 import VisualDescription, VLMDetailLevel

logger = logging.getLogger(__name__)

_VISUAL_TYPES = {"table", "figure", "chart", "image"}

_INLINE_TAG: dict[str, str] = {
    "table": "table",
    "figure": "figure",
    "chart": "figure",
    "image": "figure",
}

_VLM_SYSTEM = "You are a document analysis assistant. Return only valid JSON."

_VLM_FEW_SHOT = (
    "Examples of expected output:\n"
    '1. Simple bar chart: {"type": "figure", "summary": "Bar chart showing quarterly revenue by region.", '
    '"key_finding": "North America leads with $4.2B.", "data_extracted": "Q1 NA=$1.1B EU=$0.8B", '
    '"confidence": "high", "retrieval_mode": "text_only"}\n'
    '2. Complex table: {"type": "table", "summary": "Financial summary table covering 2019-2023.", '
    '"key_finding": "Net income doubled from 2021 to 2023.", "data_extracted": "2023 net income $2.1B", '
    '"confidence": "medium", "retrieval_mode": "text_and_image"}\n'
)

_VLM_RULES = (
    "Rules:\n"
    "- type: table / figure / chart\n"
    "- summary: 1-2 sentence retrieval-quality description\n"
    "- key_finding: the single most important piece of information\n"
    "- data_extracted: key numeric values or labels\n"
    "- confidence: high / medium / low\n"
    "- retrieval_mode: text_only / text_and_image / image_only\n"
    "Return ONLY valid JSON with exactly these 6 fields."
)


class VisualUnderstandingStage:
    stage_name = "visual_understanding"
    stage_version = "1.0"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    async def run(self, s1: Stage1Result, s2: Stage2Result) -> Stage3Result:
        logger.info("Stage 3 — Visual Understanding: %s region(s)", len(s1.regions))
        visual_regions = [r for r in s1.regions if r.region_type in _VISUAL_TYPES]

        if self.settings.fast_mode or not self.settings.use_vlm_summaries:
            descriptions = [self._skipped_description(r) for r in visual_regions]
            enriched = self._build_enriched_text(s1, s2, descriptions)
            return Stage3Result(
                document_id=s1.document_id,
                visual_descriptions=descriptions,
                enriched_text_by_page=enriched,
                fast_mode_active=True,
            )

        if not self.settings.openai_api_key:
            logger.warning("Stage 3: OPENAI_API_KEY not set — skipping VLM calls")
            descriptions = [self._skipped_description(r) for r in visual_regions]
            enriched = self._build_enriched_text(s1, s2, descriptions)
            return Stage3Result(
                document_id=s1.document_id,
                visual_descriptions=descriptions,
                enriched_text_by_page=enriched,
                fast_mode_active=True,
            )

        semaphore = asyncio.Semaphore(self.settings.vlm_concurrency_limit)
        page_lookup = {page.page_number: page for page in s1.pages}
        ocr_text_by_page = self._build_ocr_text_by_page(s1)

        tasks = [self._process_region(region, page_lookup, ocr_text_by_page, semaphore) for region in visual_regions]
        descriptions: list[VisualDescription] = await asyncio.gather(*tasks)

        enriched = self._build_enriched_text(s1, s2, descriptions)
        return Stage3Result(
            document_id=s1.document_id,
            visual_descriptions=descriptions,
            enriched_text_by_page=enriched,
        )

    def cache_key(self, s1: Stage1Result, s2: Stage2Result) -> str:
        return StageCache.compute_key(
            s1.document_id,
            self.stage_name,
            self.stage_version,
            str(self.settings.fast_mode),
            self.settings.vlm_model,
            str(self.settings.use_vlm_summaries),
        )

    # ── Per-region VLM call ────────────────────────────────────────────────────

    async def _process_region(
        self,
        region: LayoutRegion,
        page_lookup: dict[int, Any],
        ocr_text_by_page: dict[int, str],
        semaphore: asyncio.Semaphore,
    ) -> VisualDescription:
        async with semaphore:
            try:
                return await asyncio.get_event_loop().run_in_executor(
                    None, self._call_vlm_sync, region, ocr_text_by_page.get(region.page_number, "")
                )
            except Exception as exc:
                logger.warning("VLM failed for %s: %s", region.region_id, exc)
                return VisualDescription(
                    description_id=f"desc_{region.region_id}",
                    region_id=region.region_id,
                    page_number=region.page_number,
                    region_type=region.region_type,
                    crop_path=region.crop_path,
                    detail_level="fallback",
                    inline_tag=_INLINE_TAG.get(region.region_type, "figure"),
                    inline_text="[Visual: could not process]",
                    is_meaningful=False,
                    vlm_error=str(exc),
                )

    def _call_vlm_sync(self, region: LayoutRegion, context_text: str) -> VisualDescription:
        from openai import OpenAI  # lazy import

        if not region.crop_path or not Path(region.crop_path).exists():
            return self._skipped_description(region)

        image_b64 = base64.b64encode(Path(region.crop_path).read_bytes()).decode()
        parent_title = region.metadata.get("parent_title", "untitled")

        user_content: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": (
                    f"Document section: {parent_title}\n"
                    f"Page: {region.page_number}\n"
                    f"Region type: {region.region_type}\n"
                    f"Surrounding text: {context_text[:300]}\n\n"
                    f"{_VLM_FEW_SHOT}\n{_VLM_RULES}"
                ),
            },
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{image_b64}", "detail": "high"},
            },
        ]
        client = OpenAI(api_key=self.settings.openai_api_key, base_url=self.settings.openai_base_url)
        response = client.chat.completions.create(
            model=self.settings.vlm_model,
            messages=[
                {"role": "system", "content": _VLM_SYSTEM},
                {"role": "user", "content": user_content},
            ],
            max_tokens=512,
            temperature=0,
        )
        raw = (response.choices[0].message.content or "").strip()
        result = _safe_parse_json(raw)

        summary = str(result.get("summary") or "")
        key_finding = result.get("key_finding")
        data_extracted = result.get("data_extracted")
        confidence = result.get("confidence")
        is_meaningful = summary.upper() != "SKIP"
        detail_level: VLMDetailLevel = "full" if is_meaningful else "caption"
        inline_tag = _INLINE_TAG.get(region.region_type, "figure")
        inline_text = self._build_inline_text(inline_tag, summary, key_finding)

        return VisualDescription(
            description_id=f"desc_{region.region_id}",
            region_id=region.region_id,
            page_number=region.page_number,
            region_type=region.region_type,
            crop_path=region.crop_path,
            detail_level=detail_level,
            inline_tag=inline_tag,
            inline_text=inline_text,
            summary=summary,
            key_finding=str(key_finding) if key_finding else None,
            data_extracted=str(data_extracted) if data_extracted else None,
            confidence=confidence if confidence in ("high", "medium", "low") else None,
            is_meaningful=is_meaningful,
        )

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _skipped_description(self, region: LayoutRegion) -> VisualDescription:
        inline_tag = _INLINE_TAG.get(region.region_type, "figure")
        placeholder = f"[{region.region_type.capitalize()} on page {region.page_number}]"
        return VisualDescription(
            description_id=f"desc_{region.region_id}",
            region_id=region.region_id,
            page_number=region.page_number,
            region_type=region.region_type,
            crop_path=region.crop_path,
            detail_level="skipped",
            inline_tag=inline_tag,
            inline_text=f"<{inline_tag}>{placeholder}</{inline_tag}>",
        )

    def _build_inline_text(self, tag: str, summary: str, key_finding: Any) -> str:
        parts = [summary]
        if key_finding:
            parts.append(str(key_finding))
        content = " ".join(filter(None, parts)).strip() or "[Visual content]"
        return f"<{tag}>{content}</{tag}>"

    def _build_ocr_text_by_page(self, s1: Stage1Result) -> dict[int, str]:
        return {page.page_number: " ".join(item.text for item in page.items).strip() for page in s1.ocr_pages}

    def _build_enriched_text(
        self,
        s1: Stage1Result,
        s2: Stage2Result,
        descriptions: list[VisualDescription],
    ) -> dict[int, str]:
        desc_by_region: dict[str, VisualDescription] = {d.region_id: d for d in descriptions}
        region_by_id: dict[str, Any] = {r.region_id: r for r in s1.regions}
        item_lookup: dict[str, Any] = {item.item_id: item for page in s1.ocr_pages for item in page.items}

        result: dict[int, str] = {}
        for page_order in s2.pages:
            parts: list[str] = []
            inserted_regions: set[str] = set()
            for item_id in page_order.ordered_item_ids:
                item = item_lookup.get(item_id)
                if item is None:
                    continue
                region_id = item.region_id
                if region_id and region_id not in inserted_regions:
                    region = region_by_id.get(region_id)
                    if region and region.region_type in _VISUAL_TYPES:
                        desc = desc_by_region.get(region_id)
                        if desc:
                            parts.append(desc.inline_text)
                            inserted_regions.add(region_id)
                            continue
                parts.append(item.text)
            result[page_order.page_number] = " ".join(parts).strip()
        return result


def _safe_parse_json(text: str) -> dict[str, Any]:
    stripped = re.sub(r"```(?:json)?\s*|\s*```", "", text).strip()
    import json

    try:
        r = json.loads(stripped)
        if isinstance(r, dict):
            return r
    except Exception:
        pass
    match = re.search(r"\{.*\}", stripped, re.DOTALL)
    if match:
        try:
            r = json.loads(match.group())
            if isinstance(r, dict):
                return r
        except Exception:
            pass
    return {}
