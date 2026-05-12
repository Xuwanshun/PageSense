"""Stage 3 — Visual Understanding.

For each non-text region (table, figure, chart, image), generates a natural
language description. Uses a placeholder by default; real VLM calls (OpenAI
vision API) are activated when use_vlm_summaries=True and an API key is set.

Placeholder format: "[Figure on page {n}: {region_type}]"
Real VLM: concurrent async calls with asyncio.Semaphore(vlm_concurrency_limit).
"""

from __future__ import annotations

import asyncio
import base64
import logging
import re
from pathlib import Path
from typing import Any

from config import Settings
from document_Process.models.internal import LoadResult, VisualRegion
from document_Process.models.legacy import LayoutRegion

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
    '"confidence": "high"}\n'
    '2. Complex table: {"type": "table", "summary": "Financial summary table covering 2019-2023.", '
    '"key_finding": "Net income doubled from 2021 to 2023.", "data_extracted": "2023 net income $2.1B", '
    '"confidence": "medium"}\n'
)

_VLM_RULES = (
    "Rules:\n"
    "- type: table / figure / chart\n"
    "- summary: 1-2 sentence retrieval-quality description\n"
    "- key_finding: the single most important piece of information\n"
    "- data_extracted: key numeric values or labels\n"
    "- confidence: high / medium / low\n"
    "Return ONLY valid JSON with exactly these 5 fields."
)


def _placeholder_description(region: LayoutRegion) -> VisualRegion:
    """Return a placeholder without any VLM call.

    Replace this function (or set use_vlm_summaries=True) to plug in real VLM.
    """
    inline_text = f"[Figure on page {region.page_number}: {region.region_type}]"
    return VisualRegion(
        region_id=region.region_id,
        page_number=region.page_number,
        region_type=region.region_type,
        crop_path=region.crop_path,
        inline_text=inline_text,
        summary=None,
        is_meaningful=False,
    )


class VisualStage:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def run(self, load_result: LoadResult) -> list[VisualRegion]:
        """Describe all visual regions. Uses placeholder unless use_vlm_summaries=True."""
        visual_regions = [
            r for r in load_result.regions if r.region_type in _VISUAL_TYPES
        ]
        logger.info("Stage 3 — Visual: %s visual region(s)", len(visual_regions))

        if not self.settings.use_vlm_summaries or not self.settings.openai_api_key:
            if self.settings.use_vlm_summaries and not self.settings.openai_api_key:
                logger.warning(
                    "Stage 3: OPENAI_API_KEY not set — using placeholder descriptions"
                )
            return [_placeholder_description(r) for r in visual_regions]

        return asyncio.run(self._run_vlm(visual_regions, load_result))

    async def _run_vlm(
        self, regions: list[LayoutRegion], load_result: LoadResult
    ) -> list[VisualRegion]:
        ocr_text_by_page = {
            page.page_number: " ".join(item.text for item in page.items).strip()
            for page in load_result.ocr_pages
        }
        semaphore = asyncio.Semaphore(self.settings.vlm_concurrency_limit)
        tasks = [self._process_region(r, ocr_text_by_page, semaphore) for r in regions]
        return list(await asyncio.gather(*tasks))

    async def _process_region(
        self,
        region: LayoutRegion,
        ocr_text_by_page: dict[int, str],
        semaphore: asyncio.Semaphore,
    ) -> VisualRegion:
        async with semaphore:
            return await self._call_vlm_with_retry(
                region, ocr_text_by_page.get(region.page_number, "")
            )

    async def _call_vlm_with_retry(
        self, region: LayoutRegion, context_text: str
    ) -> VisualRegion:
        max_retries = self.settings.vlm_retry_max
        for attempt in range(max_retries):
            try:
                return await asyncio.get_event_loop().run_in_executor(
                    None,
                    self._call_vlm_sync,
                    region,
                    context_text,
                )
            except Exception as exc:
                exc_str = str(exc)
                is_rate_limit = "429" in exc_str or "rate" in exc_str.lower()
                if is_rate_limit and attempt < max_retries - 1:
                    wait = 2**attempt
                    logger.warning(
                        "[Visual] VLM rate-limit on attempt %d for %s, retrying in %ds",
                        attempt + 1,
                        region.region_id,
                        wait,
                    )
                    await asyncio.sleep(wait)
                else:
                    if attempt == max_retries - 1:
                        logger.error(
                            "[Visual] VLM failed after %d retries for %s: %s",
                            max_retries,
                            region.region_id,
                            exc,
                        )
                    else:
                        logger.warning("VLM failed for %s: %s", region.region_id, exc)
                    return VisualRegion(
                        region_id=region.region_id,
                        page_number=region.page_number,
                        region_type=region.region_type,
                        crop_path=region.crop_path,
                        inline_text="[Visual: could not process after retries]",
                        is_meaningful=False,
                    )
        # Should never reach here
        return VisualRegion(
            region_id=region.region_id,
            page_number=region.page_number,
            region_type=region.region_type,
            crop_path=region.crop_path,
            inline_text="[Visual: could not process after retries]",
            is_meaningful=False,
        )

    def _call_vlm_sync(self, region: LayoutRegion, context_text: str) -> VisualRegion:
        from openai import OpenAI  # lazy import

        if not region.crop_path or not Path(region.crop_path).exists():
            return _placeholder_description(region)

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
                "image_url": {
                    "url": f"data:image/png;base64,{image_b64}",
                    "detail": "high",
                },
            },
        ]
        client = OpenAI(
            api_key=self.settings.openai_api_key, base_url=self.settings.openai_base_url
        )
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
        is_meaningful = bool(summary) and summary.upper() != "SKIP"

        inline_tag = _INLINE_TAG.get(region.region_type, "figure")
        parts = [summary]
        if key_finding:
            parts.append(str(key_finding))
        content = " ".join(filter(None, parts)).strip() or "[Visual content]"
        inline_text = f"<{inline_tag}>{content}</{inline_tag}>"

        return VisualRegion(
            region_id=region.region_id,
            page_number=region.page_number,
            region_type=region.region_type,
            crop_path=region.crop_path,
            inline_text=inline_text,
            summary=summary or None,
            is_meaningful=is_meaningful,
        )


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
