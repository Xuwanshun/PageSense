"""
VLM (vision-language model) enrichment for cropped region summaries.

WHY THIS EXISTS
---------------
The OCR pipeline extracts text from pages and assigns nearby text to each
detected table or figure as its "summary". This works when a table has a
clear caption, but fails for:

  • Charts / graphs — pixels only, no extractable text describing the trend
  • Complex tables  — OCR text is garbled without column/row context
  • Diagrams        — structural meaning is entirely visual

A vision-language model (GPT-4o) can look at the saved crop image and
produce a natural-language description that is:
  1. Semantically accurate (it understands what the chart shows)
  2. Embeddable (dense enough to match user queries like "revenue growth")
  3. Grounded (it sees the actual data, not just nearby labels)

USAGE
-----
Enable in config:  USE_VLM_SUMMARIES=true  (disabled by default — costs money)
Model:             VLM_MODEL=gpt-4o        (default)

The function `enrich_summaries_with_vlm` is called from the preprocessing
pipeline after `build_visual_summaries`. Each summary with a saved crop
image gets a VLM-generated description. Failures are non-fatal — the
original OCR-text summary is kept on error.

PROMPT DESIGN FOR RAG
----------------------
The prompts are written to produce descriptions that match the kinds of
questions users actually ask. Instead of "this is a bar chart", the model
is instructed to mention the topic, specific values, trends, and conclusions
— the things a user would type into a search box.
"""
from __future__ import annotations

import base64
import logging
from pathlib import Path
from typing import Any

from config import Settings
from document_Process.models import VisualRegionSummary

logger = logging.getLogger(__name__)

# Prompts tuned for RAG retrieval quality: we want descriptions that match
# the vocabulary of user questions, not generic chart-type labels.
#
# Each prompt instructs the model to return SKIP when the image has no
# meaningful content for retrieval (logos, icons, decorative elements).
# This sentinel is parsed in _describe_crop to set is_meaningful=False.
_SKIP_SENTINEL = "SKIP"

_MEANINGLESS_INSTRUCTION = (
    "IMPORTANT: If the image is a logo, icon, product photo, watermark, decorative "
    "element, or any purely visual brand mark that contains no information useful for "
    "answering questions (e.g., a company logo, chip photograph, geometric decoration, "
    "UI button), respond with exactly the word SKIP and nothing else."
)

_SYSTEM_PROMPTS: dict[str, str] = {
    "table": (
        "You are a document analysis assistant preparing table descriptions for a "
        "retrieval-augmented generation (RAG) system. "
        "Describe the table concisely but completely: its topic, column and row headers, "
        "key data values, units of measurement, time periods covered, and any notable "
        "trends or conclusions visible in the data. "
        "Write in plain prose. Do not use markdown. "
        "Focus on information users would search for. "
        + _MEANINGLESS_INSTRUCTION
    ),
    "figure": (
        "You are a document analysis assistant preparing figure descriptions for a "
        "retrieval-augmented generation (RAG) system. "
        "Describe the figure concisely but completely: the type of visualisation "
        "(line chart, bar chart, diagram, photograph, etc.), its topic, axis labels and "
        "units if visible, key data points or trends, and any conclusions the figure "
        "supports. "
        "Write in plain prose. Do not use markdown. "
        "Focus on information users would search for. "
        + _MEANINGLESS_INSTRUCTION
    ),
}


def enrich_summaries_with_vlm(
    summaries: list[VisualRegionSummary],
    *,
    settings: Settings,
) -> list[VisualRegionSummary]:
    """
    Replace text-fallback summaries with VLM-generated descriptions.

    Only processes summaries that have a saved crop image on disk.
    Failures are logged as warnings and the original summary is kept.
    Returns a new list — the input summaries are not mutated.
    """
    enriched: list[VisualRegionSummary] = []
    for summary in summaries:
        crop = Path(summary.crop_path) if summary.crop_path else None
        if crop and crop.exists():
            try:
                description, is_meaningful = _describe_crop(
                    crop_path=crop,
                    region_type=summary.region_type,
                    context_text=summary.summary_text,
                    settings=settings,
                )
                if is_meaningful:
                    summary = summary.model_copy(update={"summary_text": description})
                    logger.info(
                        "VLM description generated for %s (%s, %d chars)",
                        summary.region_id,
                        summary.region_type,
                        len(description),
                    )
                else:
                    summary = summary.model_copy(update={"is_meaningful": False})
                    logger.info(
                        "VLM marked %s as not meaningful (logo/icon) — excluded from RAG",
                        summary.region_id,
                    )
            except Exception as exc:
                logger.warning(
                    "VLM description failed for %s — keeping OCR-text fallback: %s",
                    summary.region_id,
                    exc,
                )
        else:
            logger.debug(
                "Skipping VLM for %s — no crop image available",
                summary.region_id,
            )
        enriched.append(summary)
    return enriched


def _describe_crop(
    crop_path: Path,
    region_type: str,
    context_text: str,
    settings: Settings,
) -> tuple[str, bool]:
    """
    Call the OpenAI vision API with the cropped image.

    Returns (description, is_meaningful). When the VLM determines the image
    is a logo, icon, or decorative element with no retrieval value, it responds
    with the SKIP sentinel and is_meaningful is False.

    The surrounding OCR text (context_text) is included as a hint when it
    contains real content, so the model can anchor its description to the
    document's own language.
    """
    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is required for VLM summaries.")

    # Lazy import so the rest of the app does not depend on openai at import time
    from openai import OpenAI  # type: ignore

    client = OpenAI(api_key=settings.openai_api_key, base_url=settings.openai_base_url)
    image_b64 = base64.b64encode(crop_path.read_bytes()).decode()

    system_prompt = _SYSTEM_PROMPTS.get(region_type, _SYSTEM_PROMPTS["figure"])
    user_content: list[dict[str, Any]] = []

    # Only pass context when it is real text, not the placeholder fallback.
    if context_text and not context_text.startswith("Detected "):
        user_content.append({
            "type": "text",
            "text": f"Surrounding document text for context:\n{context_text[:400]}",
        })

    user_content.append({
        "type": "image_url",
        "image_url": {
            "url": f"data:image/png;base64,{image_b64}",
            "detail": "high",  # use high-detail mode for tables with small text
        },
    })

    response = client.chat.completions.create(
        model=settings.vlm_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        max_tokens=512,
        temperature=0,
    )
    raw = (response.choices[0].message.content or "").strip()
    if raw.upper() == _SKIP_SENTINEL:
        return ("", False)
    return (raw, True)
