"""Stage 3 — Query pipeline: section pre-filter → block retrieval → synthesis."""

from __future__ import annotations

import base64
import json
import logging
import re
from dataclasses import dataclass, field
from io import BytesIO

from PIL import Image

from config import Settings
from document_Process.clients import OpenAIJSONModelClient
from rag.chunk import RetrievedChunk
from rag.retrieve import (
    DocumentRetriever,
    _apply_query_boosts,
    _sort_by_reading_order,
)

logger = logging.getLogger(__name__)

_NEIGHBOR_WINDOW = 2

_METRIC_TERMS = frozenset(
    [
        "bleu",
        "score",
        "accuracy",
        "performance",
        "result",
        "results",
        "benchmark",
        "metric",
        "metrics",
        "percentage",
        "wer",
        "rouge",
        "f1",
        "precision",
        "recall",
        "table",
        "comparison",
        "versus",
        "vs",
    ]
)
_METRIC_TERM_MIN = 2  # how many terms must appear to trigger lower threshold

_SYSTEM_PROMPT = (
    "You are a document QA assistant. Answer using only the retrieved blocks below.\n"
    "Cite evidence as (Section: <title>, Page <n>) for every claim.\n"
    "If the blocks do not contain enough information, say so explicitly — do not speculate."
)


# ── Response types ─────────────────────────────────────────────────────────────


@dataclass
class SourceRef:
    block_id: str
    section_title: str
    page: int
    score: float
    block_type: str
    crop_paths: list[str] = field(default_factory=list)
    doc_id: str = ""
    source_filename: str = ""


@dataclass
class BlockWindow:
    anchor_block_id: str
    section_title: str
    page: int
    blocks: list[RetrievedChunk]  # anchor ± neighbors, reading order preserved
    score: float


@dataclass
class QAResponse:
    question: str
    answer: str
    sources: list[SourceRef]
    faithfulness: str | None  # "FAITHFUL" | "UNSUPPORTED_REMOVED" | None


# ── Public entry point ─────────────────────────────────────────────────────────


def answer_question(
    question: str,
    *,
    settings: Settings | None = None,
    top_k: int | None = None,
) -> QAResponse:
    """Answer a question from the indexed corpus.

    Pipeline:
    1.   Metric query detection  — no embedding, no LLM.
    1.5. Document pre-filter     — Pool 0 cosine search → candidate doc IDs.
    2.   Section pre-filter      — Pool A cosine search scoped to candidate docs.
    3.   Block retrieval         — Pool B scoped to candidate sections/docs.
    4.   Synthesis               — one LLM call.
    5.   Faithfulness gate       — optional second LLM call.
    """
    resolved = settings or Settings()
    retriever = DocumentRetriever(resolved)
    k = top_k or resolved.default_top_k

    query_embedding = retriever.embedding_backend.embed_texts([question])[0]

    is_metric_query = _detect_metric_query(question)

    # Step 1.5 — Document pre-filter (Pool 0)
    candidate_doc_ids = _document_prefilter(query_embedding, retriever, resolved)

    candidate_sections = _section_prefilter(
        question,
        query_embedding,
        retriever,
        resolved,
        is_metric_query,
        candidate_doc_ids,
    )
    raw_blocks = _retrieve_blocks(
        question,
        query_embedding,
        candidate_sections,
        retriever,
        k,
        is_metric_query,
        candidate_doc_ids,
    )

    if not raw_blocks:
        return QAResponse(
            question=question,
            answer="I cannot answer from the indexed documents because no relevant context was retrieved.",
            sources=[],
            faithfulness=None,
        )

    neighbor_window = 1 if resolved.fast_query_mode else _NEIGHBOR_WINDOW
    windows = _build_windows(
        raw_blocks[:k], retriever.block_store.get_all_chunks(), window=neighbor_window
    )
    answer = _synthesize(question, windows, resolved)
    answer, faithfulness_label = _run_faithfulness_gate(
        question, answer, windows, resolved
    )

    sources = _collect_sources(windows, resolved)
    return QAResponse(
        question=question,
        answer=answer,
        sources=sources,
        faithfulness=faithfulness_label,
    )


# ── Step 1: Metric detection ───────────────────────────────────────────────────


def _detect_metric_query(question: str) -> bool:
    tokens = set(question.lower().split())
    if sum(1 for t in tokens if t in _METRIC_TERMS) >= _METRIC_TERM_MIN:
        return True
    # Decimal numbers (e.g. "93.8", "0.85") or percentage values signal metric intent.
    return bool(re.search(r"\d+\.\d+", question) or re.search(r"\d+%", question))


# ── Step 1.5: Document pre-filter (Pool 0) ────────────────────────────────────


def _document_prefilter(
    query_embedding: list[float],
    retriever: DocumentRetriever,
    settings: Settings,
) -> set[str]:
    """Return set of doc_ids that pass Pool 0. Empty set means search all docs."""
    if not settings.use_document_prefilter:
        return set()

    top_k = settings.document_filter_top_k
    results = retriever.document_store.query(query_embedding, top_k * 2)
    passing = [r for r in results if r.score >= settings.document_filter_threshold][
        :top_k
    ]

    if passing:
        candidate_doc_ids = {str(r.metadata.get("doc_id") or "") for r in passing}
        logger.info(
            "[QA] document pre-filter: %d doc(s) pass threshold %.2f — %s",
            len(candidate_doc_ids),
            settings.document_filter_threshold,
            [
                r.metadata.get("source_filename", r.metadata.get("doc_id", "?")[:8])
                for r in passing
            ],
        )
        return candidate_doc_ids
    else:
        logger.info(
            "[QA] document pre-filter: no documents pass threshold %.2f — global fallback",
            settings.document_filter_threshold,
        )
        return set()


# ── Step 2: Section pre-filter ────────────────────────────────────────────────


def _section_prefilter(
    question: str,
    query_embedding: list[float],
    retriever: DocumentRetriever,
    settings: Settings,
    is_metric_query: bool = False,
    candidate_doc_ids: set[str] | None = None,
) -> set[str]:
    threshold = settings.section_filter_threshold
    if is_metric_query:
        threshold = min(threshold, settings.metric_query_threshold)
        logger.debug(
            "[QA] metric query detected — lowering section threshold to %.2f", threshold
        )

    filter_ids = candidate_doc_ids if candidate_doc_ids else None
    all_results = retriever.section_store.query(
        query_embedding, settings.section_filter_max * 4, filter_doc_ids=filter_ids
    )
    above_threshold = [r for r in all_results if r.score >= threshold]
    if above_threshold:
        top_score = above_threshold[0].score
        candidate_sections = [
            r for r in above_threshold if r.score >= top_score - 0.12
        ][: settings.section_filter_max]
    else:
        candidate_sections = []

    # Name-match injection: if the query contains words that appear in a section
    # title, inject that section regardless of cosine score.
    query_words = {w.lower() for w in re.findall(r"\b\w{5,}\b", question)}
    if query_words:
        all_section_records = retriever.section_store.get_all_chunks()
        if candidate_doc_ids:
            all_section_records = [
                r
                for r in all_section_records
                if str(r.metadata.get("doc_id") or "") in candidate_doc_ids
            ]
        already = {str(r.metadata.get("section_id") or "") for r in candidate_sections}
        for record in all_section_records:
            if len(candidate_sections) >= settings.section_filter_max:
                break
            if str(record.metadata.get("section_id") or "") in already:
                continue
            title_words = {
                w.lower()
                for w in re.findall(
                    r"\b\w{5,}\b",
                    record.metadata.get("section_title", "") + " " + record.text,
                )
            }
            if query_words & title_words:
                candidate_sections.append(record)
                already.add(str(record.metadata.get("section_id") or ""))
                logger.debug(
                    "[QA] name-match inject: section %r (doc %s)",
                    record.metadata.get("section_title", ""),
                    str(record.metadata.get("doc_id") or "")[:8],
                )

    # Fix I — multi-document diversity injection
    # If Pool 0 identified multiple candidate docs but Pool A only selected sections
    # from a subset of them, force-include the best section from missing docs.
    if candidate_doc_ids and len(candidate_doc_ids) > 1:
        represented_docs = {
            str(r.metadata.get("doc_id") or "") for r in candidate_sections
        }
        missing_docs = candidate_doc_ids - represented_docs
        for missing_doc_id in missing_docs:
            best = next(
                (
                    r
                    for r in all_results
                    if str(r.metadata.get("doc_id") or "") == missing_doc_id
                ),
                None,
            )
            if best:
                candidate_sections.append(best)
                logger.info(
                    "[QA] diversity inject: added section from %s (score %.3f) for multi-doc coverage",
                    missing_doc_id[:8],
                    best.score,
                )

    return {str(r.metadata.get("section_id") or "") for r in candidate_sections}


# ── Step 3: Block retrieval + boost ────────────────────────────────────────────


def _retrieve_blocks(
    question: str,
    query_embedding: list[float],
    candidate_sections: set[str],
    retriever: DocumentRetriever,
    top_k: int,
    is_metric_query: bool = False,
    candidate_doc_ids: set[str] | None = None,
) -> list[RetrievedChunk]:
    fetch_k = top_k * 2
    filter_ids = candidate_doc_ids if candidate_doc_ids else None
    raw = retriever.block_store.query(
        query_embedding, fetch_k * 3, filter_doc_ids=filter_ids
    )

    if candidate_sections:
        scoped = [
            b
            for b in raw
            if str(b.metadata.get("section_id") or "") in candidate_sections
        ]
        if scoped:
            candidate_blocks = scoped[:fetch_k]
        else:
            candidate_blocks = raw[:fetch_k]

        # For metric queries, supplement with top 2 globally-scored blocks outside
        # candidate sections (scoped to candidate_doc_ids to avoid irrelevant docs).
        if is_metric_query:
            seen_ids = {b.chunk_id for b in candidate_blocks}
            supplement_raw = retriever.block_store.query(
                query_embedding, 4, filter_doc_ids=filter_ids
            )
            global_extra = [b for b in supplement_raw if b.chunk_id not in seen_ids][:2]
            candidate_blocks = candidate_blocks + global_extra
    else:
        candidate_blocks = raw[:fetch_k]

    return _apply_query_boosts(candidate_blocks, question)


# ── Step 2b: Build BlockWindows ────────────────────────────────────────────────


def _build_windows(
    anchors: list[RetrievedChunk],
    all_blocks: list[RetrievedChunk],
    *,
    window: int = _NEIGHBOR_WINDOW,
) -> list[BlockWindow]:
    section_map: dict[tuple[str, str], list[RetrievedChunk]] = {}
    for block in all_blocks:
        key = (
            str(block.metadata.get("doc_id") or ""),
            str(block.metadata.get("section_id") or ""),
        )
        section_map.setdefault(key, []).append(block)
    for blocks in section_map.values():
        blocks.sort(key=lambda b: b.metadata.get("block_index", 0))

    position: dict[str, tuple[tuple[str, str], int]] = {}
    for key, blocks in section_map.items():
        for pos, block in enumerate(blocks):
            position[block.chunk_id] = (key, pos)

    seen_anchors: set[str] = set()
    windows: list[BlockWindow] = []
    for anchor in anchors:
        if anchor.chunk_id in seen_anchors:
            continue
        seen_anchors.add(anchor.chunk_id)

        loc = position.get(anchor.chunk_id)
        if loc is not None:
            key, pos = loc
            section = section_map[key]
            neighbor_blocks = section[max(0, pos - window) : pos + window + 1]
        else:
            neighbor_blocks = [anchor]

        ordered = _sort_by_reading_order(neighbor_blocks)
        seen_block_ids: set[str] = set()
        deduped: list[RetrievedChunk] = []
        for b in ordered:
            if b.chunk_id not in seen_block_ids:
                seen_block_ids.add(b.chunk_id)
                deduped.append(b)

        windows.append(
            BlockWindow(
                anchor_block_id=anchor.chunk_id,
                section_title=anchor.metadata.get("section_title", ""),
                page=int(
                    anchor.metadata.get("page")
                    or anchor.metadata.get("page_number")
                    or 0
                ),
                blocks=deduped,
                score=anchor.score,
            )
        )

    return windows


# ── Step 3: Synthesis ──────────────────────────────────────────────────────────


def _synthesize(question: str, windows: list[BlockWindow], settings: Settings) -> str:
    context = _build_context(windows, settings)
    user_text = f"Question: {question}\n\nRetrieved blocks:\n{context}"
    client = _synthesis_client(settings)

    if settings.use_vlm_summaries:
        seen_crops: set[str] = set()
        image_parts: list[dict] = []
        for window in windows:
            for block in window.blocks:
                crops = block.metadata.get("crop_references") or []
                if crops and crops[0] not in seen_crops:
                    seen_crops.add(crops[0])
                    try:
                        img = Image.open(crops[0]).convert("RGB")
                        max_px = settings.vision_max_image_px
                        if max(img.width, img.height) > max_px:
                            scale = max_px / max(img.width, img.height)
                            img = img.resize(
                                (int(img.width * scale), int(img.height * scale)),
                                Image.LANCZOS,
                            )
                        buf = BytesIO()
                        img.save(buf, format="JPEG", quality=85)
                        b64 = base64.b64encode(buf.getvalue()).decode()
                        image_parts.append(
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                            }
                        )
                    except Exception:
                        pass
        if image_parts:
            response = client.client.chat.completions.create(
                model=settings.vision_synthesis_model,
                temperature=0,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": user_text}] + image_parts,
                    },
                ],
            )
            return str(response.choices[0].message.content or "").strip()

    return client.generate_text(
        system_prompt=_SYSTEM_PROMPT, user_prompt=user_text
    ).strip()


def _build_context(windows: list[BlockWindow], settings: Settings) -> str:
    parts: list[str] = []
    for window in windows:
        for block in window.blocks:
            meta = block.metadata
            section_title = meta.get("section_title", "")
            page = meta.get("page") or meta.get("page_number", "")
            block_type = meta.get("block_type", "paragraph")
            content = meta.get("content") or block.text.strip()
            header = f"[Section: {section_title} | Page {page} | Type: {block_type}]"
            body = content
            crop_refs = meta.get("crop_references") or []
            if crop_refs and not settings.use_vlm_summaries:
                body += f"\n[Visual asset available: {crop_refs[0]}]"
            parts.append(f"{header}\n{body}")
    return "\n\n".join(p for p in parts if p.strip())


# ── Step 4: Faithfulness gate ──────────────────────────────────────────────────


def _run_faithfulness_gate(
    question: str,
    answer: str,
    windows: list[BlockWindow],
    settings: Settings,
) -> tuple[str, str | None]:
    if settings.fast_query_mode or not settings.use_faithfulness_check:
        return answer, None

    flat_blocks = [b for w in windows for b in w.blocks]
    try:
        from rag.faithfulness import FaithfulnessChecker

        checker = FaithfulnessChecker(settings)
        result = checker.check(question, flat_blocks, answer)
        if result.recommended_action != "return_as_is":
            corrected = checker.correct(question, answer, result, flat_blocks)
            return corrected, "UNSUPPORTED_REMOVED"
        return answer, "FAITHFUL"
    except Exception as exc:
        logger.warning("Faithfulness gate failed, returning original answer: %s", exc)
        return answer, None


# ── Internal helpers ───────────────────────────────────────────────────────────


def _collect_sources(windows: list[BlockWindow], settings: Settings) -> list[SourceRef]:
    doc_ids = {
        str(block.metadata.get("doc_id") or "")
        for window in windows
        for block in window.blocks
        if block.metadata.get("doc_id")
    }
    filename_cache = _load_manifest_filenames(doc_ids, settings)

    seen: set[str] = set()
    sources: list[SourceRef] = []
    for window in windows:
        for block in window.blocks:
            if block.chunk_id in seen:
                continue
            seen.add(block.chunk_id)
            meta = block.metadata
            # Neighbour blocks from get_all_chunks() have score=0.0 (no similarity
            # was computed for them). Use the parent window's anchor score so that
            # all sources in the same window report a meaningful similarity value.
            score = block.score if block.score > 0.0 else window.score
            doc_id = str(meta.get("doc_id") or "")
            raw_crops = list(meta.get("crop_references") or [])
            seen_crops: set[str] = set()
            crop_paths: list[str] = []
            for p in raw_crops:
                if p not in seen_crops:
                    seen_crops.add(p)
                    crop_paths.append(p)
            sources.append(
                SourceRef(
                    block_id=block.chunk_id,
                    section_title=meta.get("section_title", ""),
                    page=int(meta.get("page") or meta.get("page_number") or 0),
                    score=round(score, 4),
                    block_type=meta.get("block_type", "paragraph"),
                    crop_paths=crop_paths,
                    doc_id=doc_id,
                    source_filename=filename_cache.get(
                        doc_id, doc_id[:8] if doc_id else ""
                    ),
                )
            )
    return sources


def _load_manifest_filenames(doc_ids: set[str], settings: Settings) -> dict[str, str]:
    result: dict[str, str] = {}
    for doc_id in doc_ids:
        if not doc_id:
            continue
        manifest_path = settings.processed_documents_dir / doc_id / "manifest.json"
        try:
            data = json.loads(manifest_path.read_text(encoding="utf-8"))
            result[doc_id] = str(data.get("source_filename") or doc_id[:8])
        except Exception:
            result[doc_id] = doc_id[:8]
    return result


def _synthesis_client(settings: Settings) -> OpenAIJSONModelClient:
    if not settings.openai_api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Add it to your .env file or set it as an environment variable."
        )
    return OpenAIJSONModelClient(
        model=settings.synthesis_model,
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
    )
