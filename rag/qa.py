"""Stage 3 — Query pipeline: section pre-filter → block retrieval → synthesis."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from config import Settings
from document_Process.clients import OpenAIJSONModelClient
from rag.chunk import RetrievedChunk
from rag.retrieve import (
    _TOP_SECTIONS,
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
    1. Section pre-filter  — cosine search Pool A → candidate section IDs.
    2. Block retrieval     — search Pool B, apply boosts, return top candidates.
    3. Synthesis           — build context, one LLM call, cite section + page.
    4. Faithfulness gate   — optional second LLM call (USE_FAITHFULNESS_CHECK).
    """
    resolved = settings or Settings()
    retriever = DocumentRetriever(resolved)
    k = top_k or resolved.default_top_k

    query_embedding = retriever.embedding_backend.embed_texts([question])[0]

    is_metric_query = _detect_metric_query(question)
    candidate_sections = _section_prefilter(query_embedding, retriever, resolved, is_metric_query)
    raw_blocks = _retrieve_blocks(question, query_embedding, candidate_sections, retriever, k, is_metric_query)

    if not raw_blocks:
        return QAResponse(
            question=question,
            answer="I cannot answer from the indexed documents because no relevant context was retrieved.",
            sources=[],
            faithfulness=None,
        )

    windows = _build_windows(raw_blocks[:k], retriever.block_store.get_all_chunks())
    answer = _synthesize(question, windows, resolved)
    answer, faithfulness_label = _run_faithfulness_gate(question, answer, windows, resolved)

    sources = _collect_sources(windows)
    return QAResponse(
        question=question,
        answer=answer,
        sources=sources,
        faithfulness=faithfulness_label,
    )


# ── Step 1: Section pre-filter ─────────────────────────────────────────────────


def _detect_metric_query(question: str) -> bool:
    tokens = set(question.lower().split())
    return sum(1 for t in tokens if t in _METRIC_TERMS) >= _METRIC_TERM_MIN


def _section_prefilter(
    query_embedding: list[float],
    retriever: DocumentRetriever,
    settings: Settings,
    is_metric_query: bool = False,
) -> set[str]:
    threshold = settings.section_filter_threshold
    if is_metric_query:
        threshold = min(threshold, settings.metric_query_threshold)
        logger.debug("[QA] metric query detected — lowering section threshold to %.2f", threshold)
    results = retriever.section_store.query(query_embedding, _TOP_SECTIONS * 2)
    above = [r for r in results if r.score >= threshold][:_TOP_SECTIONS]
    return {str(r.metadata.get("section_id") or "") for r in above}


# ── Step 2: Block retrieval + boost ────────────────────────────────────────────


def _retrieve_blocks(
    question: str,
    query_embedding: list[float],
    candidate_sections: set[str],
    retriever: DocumentRetriever,
    top_k: int,
    is_metric_query: bool = False,
) -> list[RetrievedChunk]:
    fetch_k = top_k * 2
    raw = retriever.block_store.query(query_embedding, fetch_k * 3)

    if candidate_sections:
        scoped = [b for b in raw if str(b.metadata.get("section_id") or "") in candidate_sections]
        if scoped:
            candidate_blocks = scoped[:fetch_k]
        else:
            candidate_blocks = raw[:fetch_k]

        # A2: for metric queries, always supplement with the top 2 globally-scored
        # blocks that fall outside the candidate sections. This catches blocks whose
        # content is about results/metrics but whose *section* was misdetected by the
        # layout model (e.g. "6 Results" classified as text_block → absorbed into
        # "5.4 Regularization" which scores ~0.16 at Pool A level and is never a
        # candidate section). Pool B embeddings still carry the real block text, so
        # cosine search there finds the right block even though Pool A missed it.
        if is_metric_query:
            seen_ids = {b.chunk_id for b in candidate_blocks}
            global_extra = [b for b in raw if b.chunk_id not in seen_ids][:2]
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
                page=int(anchor.metadata.get("page") or anchor.metadata.get("page_number") or 0),
                blocks=deduped,
                score=anchor.score,
            )
        )

    return windows


# ── Step 3: Synthesis ──────────────────────────────────────────────────────────


def _synthesize(question: str, windows: list[BlockWindow], settings: Settings) -> str:
    context = _build_context(windows, settings)
    user_prompt = f"Question: {question}\n\nRetrieved blocks:\n{context}"
    client = _synthesis_client(settings)
    return client.generate_text(system_prompt=_SYSTEM_PROMPT, user_prompt=user_prompt).strip()


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
                body += (
                    f"\n[Note: this block has an associated figure/table image at {crop_refs[0]}"
                    " — enable USE_VLM_SUMMARIES=true to include visual descriptions]"
                )
            parts.append(f"{header}\n{body}")
    return "\n\n".join(p for p in parts if p.strip())


# ── Step 4: Faithfulness gate ──────────────────────────────────────────────────


def _run_faithfulness_gate(
    question: str,
    answer: str,
    windows: list[BlockWindow],
    settings: Settings,
) -> tuple[str, str | None]:
    if not settings.use_faithfulness_check:
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


def _collect_sources(windows: list[BlockWindow]) -> list[SourceRef]:
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
            sources.append(
                SourceRef(
                    block_id=block.chunk_id,
                    section_title=meta.get("section_title", ""),
                    page=int(meta.get("page") or meta.get("page_number") or 0),
                    score=round(score, 4),
                    block_type=meta.get("block_type", "paragraph"),
                )
            )
    return sources


def _synthesis_client(settings: Settings) -> OpenAIJSONModelClient:
    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is not set. Add it to your .env file or set it as an environment variable.")
    return OpenAIJSONModelClient(
        model=settings.synthesis_model,
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
    )
