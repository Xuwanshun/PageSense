"""Backward-compat shim — preserves document_Process.vlm import path for tests.

Real VLM logic lives in document_Process/stages/stage3_visual.py.
"""

from __future__ import annotations

import base64
import logging
from pathlib import Path

from config import Settings
from document_Process.models import VisualRegionSummary

logger = logging.getLogger(__name__)

_PLACEHOLDER_PREFIXES = ("Detected ", "detected ")

_SYSTEM_PROMPTS: dict[str, str] = {
    "table": (
        "You are an expert data analyst. Describe this table image concisely: "
        "what data it contains, its structure, and key values. "
        "If the image is decorative or not a real table, respond with exactly: SKIP"
    ),
    "figure": (
        "You are an expert visual analyst. Describe this figure or chart image concisely: "
        "what it shows, the type of chart, and key insights. "
        "If the image is decorative or not meaningful, respond with exactly: SKIP"
    ),
    "chart": (
        "You are an expert data analyst. Describe this chart image concisely: "
        "the chart type, data shown, trends, and key findings. "
        "If the image is decorative or not meaningful, respond with exactly: SKIP"
    ),
    "image": (
        "You are an expert visual analyst. Describe what this image shows concisely. "
        "If the image is decorative, a logo, or not meaningful for document understanding, "
        "respond with exactly: SKIP"
    ),
}
_DEFAULT_SYSTEM_PROMPT = _SYSTEM_PROMPTS["image"]


def _describe_crop(
    crop_path: str | Path,
    region_type: str,
    context_text: str,
    settings: Settings,
) -> tuple[str, bool]:
    """Call the VLM to get a description of a cropped region image.

    Returns (description, is_meaningful). If the model returns "SKIP",
    is_meaningful is False and description is "".
    """
    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is required for VLM descriptions")

    import openai

    crop_path = Path(crop_path)
    image_bytes = crop_path.read_bytes()
    b64 = base64.b64encode(image_bytes).decode()
    data_url = f"data:image/png;base64,{b64}"

    system_prompt = _SYSTEM_PROMPTS.get(region_type, _DEFAULT_SYSTEM_PROMPT)

    user_content: list[dict] = []
    is_placeholder = any(context_text.startswith(p) for p in _PLACEHOLDER_PREFIXES)
    if context_text.strip() and not is_placeholder:
        user_content.append({"type": "text", "text": f"Context text near this region:\n{context_text}"})
    user_content.append({"type": "image_url", "image_url": {"url": data_url}})

    client = openai.OpenAI(
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url or None,
    )
    response = client.chat.completions.create(
        model=settings.vlm_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        max_tokens=512,
    )
    text = (response.choices[0].message.content or "").strip()
    if text.strip().upper() == "SKIP":
        return "", False
    return text, True


def enrich_summaries_with_vlm(
    summaries: list[VisualRegionSummary],
    *,
    settings: Settings,
) -> list[VisualRegionSummary]:
    """Enrich VisualRegionSummary objects with VLM descriptions where crops exist.

    Returns a new list; input summaries are not mutated.
    """
    results: list[VisualRegionSummary] = []
    for summary in summaries:
        if not summary.crop_path or not Path(summary.crop_path).exists():
            results.append(summary.model_copy())
            continue

        try:
            description, is_meaningful = _describe_crop(
                summary.crop_path,
                summary.region_type,
                summary.summary_text,
                settings,
            )
        except Exception as exc:
            logger.warning("VLM description failed for %s: %s", summary.region_id, exc)
            results.append(summary.model_copy())
            continue

        updated = summary.model_copy(
            update={
                "is_meaningful": is_meaningful,
                "summary_text": description if is_meaningful else summary.summary_text,
            }
        )
        results.append(updated)

    return results
