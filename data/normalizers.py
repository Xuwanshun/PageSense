"""One normalizer per dataset → unified schema.

Every normalizer returns List[dict].  Empty list means "skip this sample."

Unified schema
--------------
    image    : PIL.Image.Image | None   (None for text-only datasets, e.g. GovReport)
    prompt   : str
    response : str
    source   : str
    metadata : dict
"""
from __future__ import annotations

import io
import logging
from collections import Counter
from typing import Callable, List, Optional

from PIL import Image

from .config import DOCLAYNET_CATEGORIES, PROMPTS

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pil(obj) -> Optional[Image.Image]:
    """Convert whatever HF gives us (PIL, bytes, {"bytes":…, "path":…}) to PIL."""
    if obj is None:
        return None
    if isinstance(obj, Image.Image):
        return obj.convert("RGB")
    if isinstance(obj, dict):
        if obj.get("bytes"):
            return Image.open(io.BytesIO(obj["bytes"])).convert("RGB")
        if obj.get("path"):
            return Image.open(obj["path"]).convert("RGB")
    if isinstance(obj, (bytes, bytearray)):
        return Image.open(io.BytesIO(obj)).convert("RGB")
    log.warning("_pil: unrecognised image type %s", type(obj))
    return None


def _out(image, prompt, response, source, metadata) -> dict:
    return {"image": image, "prompt": prompt, "response": response,
            "source": source, "metadata": metadata}


# ---------------------------------------------------------------------------
# 1. ChartCap  —  junyoung-00/ChartCap
# ---------------------------------------------------------------------------
# Fields: image (PIL), image_filename, chart_info, caption
# Splits: train 508 783 / test 56 486

def normalize_chartcap(sample: dict) -> List[dict]:
    image   = _pil(sample.get("image"))
    caption = (sample.get("caption") or "").strip()
    if not image or not caption:
        return []
    return [_out(image, PROMPTS["chartcap"], caption, "chartcap",
                 {"image_filename": sample.get("image_filename", "")})]


# ---------------------------------------------------------------------------
# 2. ChartSumm  —  Google Drive → S3 cache
# ---------------------------------------------------------------------------
# S3 record fields: image_b64, prompt, response, source, metadata

def normalize_chartsumm(sample: dict) -> List[dict]:
    from .s3_cache import b64_to_pil
    if "image_b64" not in sample:
        return []
    try:
        image = b64_to_pil(sample["image_b64"])
    except Exception as exc:
        log.warning("normalize_chartsumm: decode error: %s", exc)
        return []
    response = (sample.get("response") or "").strip()
    if not response:
        return []
    return [_out(image, sample.get("prompt", PROMPTS["chartsumm"]),
                 response, "chartsumm", sample.get("metadata", {}))]


# ---------------------------------------------------------------------------
# 3. ArXivCap  —  MMInstruction/ArxivCap
# ---------------------------------------------------------------------------
# Fields: arxiv_id, title, abstract, caption_images[].cil_pairs[].{image, caption, sub_caption}
# Splits: train 573 k papers → ~6.4 M images
# 1 paper → N samples (one per figure)

def normalize_arxivcap(sample: dict) -> List[dict]:
    arxiv_id = sample.get("arxiv_id", "")
    title    = sample.get("title", "")
    results  = []
    for group in (sample.get("caption_images") or []):
        main = (group.get("caption") or "").strip()
        for pair in (group.get("cil_pairs") or []):
            image = _pil(pair.get("image"))
            if image is None:
                continue
            sub     = (pair.get("sub_caption") or "").strip()
            caption = f"{main} {sub}".strip() if sub else main
            if not caption:
                continue
            results.append(_out(image, PROMPTS["arxivcap"], caption, "arxivcap",
                                {"arxiv_id": arxiv_id, "title": title}))
    return results


# ---------------------------------------------------------------------------
# 4. SciCap  —  CrowdAILab/scicap
# ---------------------------------------------------------------------------
# Fields: images.{figure_type, file_name, id, ocr}, annotations.{caption, caption_no_index, …}
# Splits: train / validation / test  (~400 k figures)
# ⚠ Known schema inconsistency across HF shards — normalizer is defensive.

def normalize_scicap(sample: dict) -> List[dict]:
    images_meta = sample.get("images") or {}
    annotations = sample.get("annotations") or {}
    caption = (annotations.get("caption_no_index") or annotations.get("caption") or "").strip()
    if not caption:
        return []
    image = (
        _pil(sample.get("image"))
        or _pil(images_meta.get("image"))
        or _pil(sample.get("png"))
    )
    if image is None:
        return []
    return [_out(image, PROMPTS["scicap"], caption, "scicap", {
        "figure_type": images_meta.get("figure_type", ""),
        "file_name":   images_meta.get("file_name", ""),
        "image_id":    str(images_meta.get("id", "")),
    })]


# ---------------------------------------------------------------------------
# 5. DocVQA (shared logic)  —  lmms-lab/DocVQA  +  HuggingFaceM4/DocumentVQA
# ---------------------------------------------------------------------------
# Both datasets expose the same schema:
#   questionId, question, image (PIL), answers (list[str]), docId
#
# lmms-lab/DocVQA  → validation split only (5 350 samples with answers)
# HuggingFaceM4/DocumentVQA → full train split (39 463 samples)

def _normalize_docvqa_common(sample: dict, source: str) -> List[dict]:
    image    = _pil(sample.get("image"))
    question = (sample.get("question") or "").strip()
    answers  = sample.get("answers") or []
    answer   = answers[0].strip() if answers else ""
    if not image or not question or not answer:
        return []
    return [_out(image, PROMPTS["docvqa"].format(question=question), answer, source, {
        "question_id":    sample.get("questionId", ""),
        "question_types": sample.get("question_types") or [],
        "all_answers":    answers,
    })]


def normalize_docvqa(sample: dict) -> List[dict]:
    """lmms-lab/DocVQA — validation split (5 350 samples)."""
    return _normalize_docvqa_common(sample, "docvqa")


def normalize_docvqa_full(sample: dict) -> List[dict]:
    """HuggingFaceM4/DocumentVQA — full train split (39 463 samples)."""
    return _normalize_docvqa_common(sample, "docvqa_full")


# ---------------------------------------------------------------------------
# 6. VisText  —  MIT GitHub → S3 cache
# ---------------------------------------------------------------------------
# S3 record fields: image_b64, prompt, response (L2L3 > L2 > L1), metadata

def normalize_vistext(sample: dict) -> List[dict]:
    from .s3_cache import b64_to_pil
    if "image_b64" not in sample:
        return []
    try:
        image = b64_to_pil(sample["image_b64"])
    except Exception as exc:
        log.warning("normalize_vistext: decode error: %s", exc)
        return []
    response = (sample.get("response") or "").strip()
    if not response:
        return []
    return [_out(image, sample.get("prompt", PROMPTS["vistext"]),
                 response, "vistext", sample.get("metadata", {}))]


# ---------------------------------------------------------------------------
# 7. ShareGPT4V  —  Lin-Chen/ShareGPT4V
# ---------------------------------------------------------------------------
# Fields: id, image (path string, NOT PIL), conversations [{from, value}]
# Images must be pre-uploaded to S3 (COCO, SAM, LLaVA-CC3M, …).
# image_loader is provided by S3ImageResolver in pipeline.py.

def normalize_sharegpt4v(sample: dict, image_loader=None) -> List[dict]:
    conversations = sample.get("conversations") or []
    human = next((c for c in conversations if c.get("from") == "human"), None)
    gpt   = next((c for c in conversations if c.get("from") == "gpt"),   None)
    if not human or not gpt:
        return []
    response = (gpt.get("value") or "").strip()
    if not response:
        return []

    question = (human.get("value") or "").replace("<image>", "").strip()
    prompt   = PROMPTS["sharegpt4v"].format(question=question) if question else PROMPTS["chartcap"]

    image_field = sample.get("image")
    image: Optional[Image.Image] = None
    if image_loader and isinstance(image_field, str):
        image = image_loader(image_field)
    elif not isinstance(image_field, str):
        image = _pil(image_field)

    if image is None:
        return []
    return [_out(image, prompt, response, "sharegpt4v",
                 {"id": sample.get("id", ""), "image_path": str(image_field)})]


# ---------------------------------------------------------------------------
# 8. DocLayNet v1.2  —  ds4sd/DocLayNet
# ---------------------------------------------------------------------------
# Fields: image (PIL), labels_block (list[int]), bboxes_block (list[[x0,y0,x1,y1]])
# Category IDs → DOCLAYNET_CATEGORIES (0=Caption … 10=Title)
# Splits: train 69 103 / validation 6 489 / test 4 993 pages
#
# Task: given a document page image, describe its layout structure.
# Response: structured count of each element type present.

def normalize_doclaynet(sample: dict) -> List[dict]:
    image = _pil(sample.get("image"))
    if image is None:
        return []

    # Accept both "labels_block" (v1.2 schema) and "labels" (older schema)
    raw_labels = sample.get("labels_block") or sample.get("labels") or []
    if not raw_labels:
        return []

    label_names: List[str] = []
    for lbl in raw_labels:
        if isinstance(lbl, int) and 0 <= lbl < len(DOCLAYNET_CATEGORIES):
            label_names.append(DOCLAYNET_CATEGORIES[lbl])
        elif isinstance(lbl, str) and lbl in DOCLAYNET_CATEGORIES:
            label_names.append(lbl)

    if not label_names:
        return []

    counts = Counter(label_names)

    # Build a natural-language layout description ordered by DOCLAYNET_CATEGORIES
    lines = ["This document page contains the following layout elements:"]
    for cat in DOCLAYNET_CATEGORIES:
        if cat in counts:
            n = counts[cat]
            plural = "s" if n > 1 else ""
            lines.append(f"- {n} {cat}{plural}")

    response = "\n".join(lines)

    return [_out(image, PROMPTS["doclaynet"], response, "doclaynet", {
        "num_elements": len(label_names),
        "element_counts": dict(counts),
    })]


# ---------------------------------------------------------------------------
# 9. DUDE  —  lmms-lab/DUDE
# ---------------------------------------------------------------------------
# Document Understanding Dataset and Evaluation (ICDAR 2023)
# ~27 k documents, ~100 k QA pairs across diverse document types.
#
# Fields (defensive — schema may vary by HF mirror):
#   questionId / id, question, image (PIL | list[PIL]), answers / answer
#   answer_type: extractive | abstractive | not-answerable
#
# Multi-page: when "images" is a list we use only the first page image.
# "not-answerable" samples are skipped (no useful supervision signal).

def normalize_dude(sample: dict) -> List[dict]:
    # Resolve image — single image or first page of multi-page doc
    image = _pil(sample.get("image"))
    if image is None:
        pages = sample.get("images") or []
        if pages:
            image = _pil(pages[0])
    if image is None:
        return []

    question = (sample.get("question") or "").strip()
    if not question:
        return []

    # Normalise answers — may be a list, a single string, or absent
    raw_answers = sample.get("answers") or sample.get("answer") or []
    if isinstance(raw_answers, str):
        raw_answers = [raw_answers]
    answers = [a.strip() for a in raw_answers if a and a.strip()]

    # Skip unanswerable samples — they have no training signal for generation
    answer_type = (sample.get("answer_type") or "").lower()
    if answer_type == "not-answerable" or not answers:
        return []

    answer = answers[0]

    return [_out(image, PROMPTS["dude"].format(question=question), answer, "dude", {
        "question_id":  sample.get("questionId", sample.get("id", "")),
        "answer_type":  answer_type,
        "all_answers":  answers,
    })]


# ---------------------------------------------------------------------------
# 10. GovReport  —  ccdv/govreport-summarization  (text-only)
# ---------------------------------------------------------------------------
# US government report summaries — long-document summarization task.
# Fields: report (str), summary (str)
# Splits: train 17 517 / validation 973 / test 973
#
# image is None — train.py VLMDataset handles text-only samples
# (the "image" key is omitted from the JSONL record when absent).
#
# Very long reports are truncated to MAX_REPORT_CHARS to stay within
# model_max_length (4096 tokens ≈ ~16 k characters with Qwen3 tokenizer).

_MAX_REPORT_CHARS = 12_000


def normalize_govreport(sample: dict) -> List[dict]:
    report  = (sample.get("report")  or "").strip()
    summary = (sample.get("summary") or "").strip()
    if not report or not summary:
        return []

    # Hard-truncate the report to avoid exceeding model_max_length
    truncated = len(report) > _MAX_REPORT_CHARS
    if truncated:
        report = report[:_MAX_REPORT_CHARS].rsplit(" ", 1)[0] + " [...]"

    prompt = PROMPTS["govreport"].format(report=report)

    return [_out(None, prompt, summary, "govreport", {
        "report_len":   len(report),
        "summary_len":  len(summary),
        "truncated":    truncated,
    })]
