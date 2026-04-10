"""
Context compression for retrieved chunks.

Strips PDF OCR artifacts, boilerplate, and query-irrelevant sentences from
retrieved passages before they are fed to the synthesis agent.  This reduces
prompt token usage and improves answer quality by removing noise.

Controlled by ``use_context_compression`` in Settings.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from config import Settings
from document_Process.clients import build_openai_client
from rag.chunk import RetrievedChunk

logger = logging.getLogger(__name__)

CONTEXT_COMPRESSION_PROMPT = """You are a context distiller for a document QA system that processes OCR-parsed PDFs.

PDF parsing artifacts to watch for and remove:
- Repeated headers/footers (page numbers, document titles appearing mid-chunk)
- OCR noise (garbled characters, broken hyphenation, isolated punctuation)
- Navigation text (e.g. "See section 3.2", "continued on next page")
- Boilerplate (copyright notices, revision history lines)
- Sentences that are topically in the right section but do not contribute \
any information relevant to the query

Query: {query}

Retrieved context passages (in retrieval rank order):
{retrieved_passages}

Instructions:
1. For each passage, keep sentences that directly contribute evidence toward answering the query
2. Remove OCR artifacts and boilerplate silently (do not mention removal)
3. For TABLE_DESC chunks: keep the full description if the table is relevant; \
these are already compressed — do not truncate further
4. For FIGURE_DESC chunks: keep the full description; same rule as tables
5. For PROSE chunks: extract only the sentences bearing on the query; \
preserve exact wording — do not paraphrase
6. Preserve chunk attribution tags so the synthesis agent can cite sources
7. If a passage scores < {compression_threshold} AND contains no unique information \
not covered by higher-scoring passages, drop it entirely

Output the compressed context maintaining this structure:
[CHUNK {{chunk_id}} | {{region_type}} | {{parent_title}} > {{parent_subtitle}}]
{{compressed_content}}

Target: reduce total tokens by ≥40% while retaining all answer-bearing content."""

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _primary_region_type(chunk: RetrievedChunk) -> str:
    types: list[str] = chunk.metadata.get("region_types") or []
    if "table" in types:
        return "TABLE_DESC"
    if "figure" in types:
        return "FIGURE_DESC"
    if len(chunk.text.strip()) < 80:
        return "SECTION_HEADER"
    return "PROSE"


def _best_content(chunk: RetrievedChunk, visual_summaries: dict[str, Any]) -> str:
    region_ids: list[str] = (
        chunk.metadata.get("source_region_ids") or chunk.metadata.get("region_ids") or []
    )
    for rid in region_ids:
        summary = visual_summaries.get(str(rid))
        if summary and summary.get("summary_text"):
            return summary["summary_text"]
    return chunk.text


def _format_passages(chunks: list[RetrievedChunk], visual_summaries: dict[str, Any]) -> str:
    """Format retrieved chunks into the tagged passage block expected by the prompt."""
    lines: list[str] = []
    for chunk in chunks:
        region_type = _primary_region_type(chunk)
        parent_title = chunk.metadata.get("parent_title") or ""
        parent_subtitle = chunk.metadata.get("parent_subtitle") or ""
        rerank_score = round(chunk.score, 4)
        content = _best_content(chunk, visual_summaries)
        lines.append(
            f"[CHUNK {chunk.chunk_id} | {region_type} | {parent_title} > {parent_subtitle} | score: {rerank_score}]\n"
            f"{content}"
        )
    return "\n\n".join(lines)


class ContextCompressor:
    """
    Compresses retrieved passages with an LLM before synthesis.

    Falls back to the uncompressed passage text on LLM failure so that the
    QA pipeline is never blocked by a compression error.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def compress(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        *,
        visual_summaries: dict[str, Any] | None = None,
        compression_threshold: float = 0.5,
    ) -> str:
        """
        Return a compressed, query-focused version of the retrieved passages.

        The returned string preserves the
        ``[CHUNK id | type | title > subtitle]`` tag format so the synthesis
        agent can still cite sources by chunk id.
        """
        if not chunks:
            return ""

        summaries = visual_summaries or {}
        passages = _format_passages(chunks, summaries)

        user_prompt = CONTEXT_COMPRESSION_PROMPT.format(
            query=query,
            retrieved_passages=passages,
            compression_threshold=compression_threshold,
        )

        try:
            client = build_openai_client(self.settings)
            return client.generate_text(
                system_prompt=(
                    "You are a context distiller. "
                    "Respond only with the compressed passages in the specified format."
                ),
                user_prompt=user_prompt,
            ).strip()
        except Exception as exc:
            logger.warning("Context compression failed, using uncompressed passages: %s", exc)
            return passages
