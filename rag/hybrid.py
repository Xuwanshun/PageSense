"""
Hybrid retrieval utilities: BM25 sparse index, Reciprocal Rank Fusion,
parent-context expansion, and region-type boosting.

Usage pattern (handled by DocumentRetriever.hybrid_retrieve):
    1. Run dense vector query → dense_results (list[RetrievedChunk])
    2. Run BM25 sparse query  → sparse_results (list[RetrievedChunk])
    3. rrf_fuse(dense_results, sparse_results) → fused (list[RetrievedChunk])
    4. apply_region_boost(fused, query) → re-scored
    5. expand_to_parent_context(top_k_chunks, all_chunks, threshold) → final
"""

from __future__ import annotations

import math
import re
from typing import Any

from rag.chunk import RetrievedChunk

# ── Tokeniser ─────────────────────────────────────────────────────────────────

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Terms that indicate data-seeking / numeric intent in a query.
_DATA_INTENT_TERMS = frozenset(
    [
        "how many",
        "how much",
        "total",
        "average",
        "mean",
        "percent",
        "percentage",
        "%",
        "rate",
        "ratio",
        "number",
        "count",
        "compare",
        "comparison",
        "versus",
        "vs",
        "list",
        "top",
        "rank",
        "ranking",
        "highest",
        "lowest",
        "maximum",
        "minimum",
        "sum",
        "breakdown",
        "distribution",
    ]
)

_DIGIT_RE = re.compile(r"\d")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


# ── BM25 index ────────────────────────────────────────────────────────────────


class BM25Index:
    """
    In-memory BM25 (Okapi BM25) index built over a flat list of chunk records.

    Call ``build()`` once with the full corpus, then ``query()`` as needed.
    Parameters k1=1.5 and b=0.75 are the standard BM25 defaults.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._chunk_ids: list[str] = []
        self._metadatas: list[dict[str, Any]] = []
        self._texts: list[str] = []
        self._token_lists: list[list[str]] = []
        self._doc_freqs: dict[str, int] = {}
        self._avg_dl: float = 1.0

    def build(
        self,
        chunk_ids: list[str],
        texts: list[str],
        metadatas: list[dict[str, Any]],
    ) -> None:
        """(Re-)build the index from the given parallel lists."""
        self._chunk_ids = list(chunk_ids)
        self._texts = list(texts)
        self._metadatas = list(metadatas)
        self._token_lists = [_tokenize(t) for t in texts]

        # Document-frequency table
        self._doc_freqs = {}
        for tokens in self._token_lists:
            for token in set(tokens):
                self._doc_freqs[token] = self._doc_freqs.get(token, 0) + 1

        # Average document length (in tokens)
        total = sum(len(t) for t in self._token_lists)
        self._avg_dl = (total / len(self._token_lists)) if self._token_lists else 1.0

    def query(
        self,
        query: str,
        top_k: int,
        *,
        doc_filter: list[str] | None = None,
    ) -> list[RetrievedChunk]:
        """Return the top-K chunks ranked by BM25 score."""
        if not self._chunk_ids:
            return []

        query_tokens = _tokenize(query)
        if not query_tokens:
            return []

        n = len(self._token_lists)
        filter_set = set(doc_filter) if doc_filter else None

        scored: list[tuple[float, int]] = []
        for idx, tokens in enumerate(self._token_lists):
            if filter_set is not None:
                doc_id = self._metadatas[idx].get("document_id")
                if doc_id not in filter_set:
                    continue

            doc_len = len(tokens)
            tf_map: dict[str, int] = {}
            for token in tokens:
                tf_map[token] = tf_map.get(token, 0) + 1

            score = 0.0
            for token in query_tokens:
                df = self._doc_freqs.get(token, 0)
                if df == 0:
                    continue
                idf = math.log((n - df + 0.5) / (df + 0.5) + 1.0)
                tf = tf_map.get(token, 0)
                numerator = tf * (self.k1 + 1.0)
                denominator = tf + self.k1 * (
                    1.0 - self.b + self.b * doc_len / self._avg_dl
                )
                score += idf * numerator / denominator

            scored.append((score, idx))

        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [
            RetrievedChunk(
                chunk_id=self._chunk_ids[idx],
                text=self._texts[idx],
                metadata=self._metadatas[idx],
                score=bm25_score,
            )
            for bm25_score, idx in scored[:top_k]
        ]


# ── Reciprocal Rank Fusion ────────────────────────────────────────────────────

_RRF_K = 60  # standard constant that dampens top-rank advantage


def rrf_fuse(
    dense_results: list[RetrievedChunk],
    sparse_results: list[RetrievedChunk],
    *,
    rrf_k: int = _RRF_K,
) -> list[RetrievedChunk]:
    """
    Merge two ranked lists with Reciprocal Rank Fusion.

    score(chunk) = 1/(rrf_k + dense_rank) + 1/(rrf_k + sparse_rank)

    Chunks that appear in only one list get 0 for the missing rank component.
    The returned list is ordered by descending RRF score.
    """
    # Build rank maps (1-based)
    dense_rank: dict[str, int] = {c.chunk_id: i + 1 for i, c in enumerate(dense_results)}
    sparse_rank: dict[str, int] = {c.chunk_id: i + 1 for i, c in enumerate(sparse_results)}

    # Merge all unique chunk ids
    all_ids: dict[str, RetrievedChunk] = {}
    for chunk in dense_results:
        all_ids[chunk.chunk_id] = chunk
    for chunk in sparse_results:
        all_ids.setdefault(chunk.chunk_id, chunk)

    fused: list[RetrievedChunk] = []
    for chunk_id, chunk in all_ids.items():
        dr = dense_rank.get(chunk_id)
        sr = sparse_rank.get(chunk_id)
        rrf_score = (1.0 / (rrf_k + dr) if dr else 0.0) + (
            1.0 / (rrf_k + sr) if sr else 0.0
        )
        fused.append(
            RetrievedChunk(
                chunk_id=chunk.chunk_id,
                text=chunk.text,
                metadata=chunk.metadata,
                score=rrf_score,
            )
        )

    return sorted(fused, key=lambda c: c.score, reverse=True)


# ── Query intent detection ────────────────────────────────────────────────────


def has_data_seeking_intent(query: str) -> bool:
    """
    Return True when the query contains numeric/comparative/list-seeking intent,
    which triggers a 1.3× boost on table/figure chunks.
    """
    lowered = query.lower()
    if _DIGIT_RE.search(lowered):
        return True
    for term in _DATA_INTENT_TERMS:
        if term in lowered:
            return True
    return False


def detect_query_type(query: str) -> str:
    """
    Lightweight heuristic query classifier.

    Returns one of: 'prose' | 'table_lookup' | 'figure_lookup' | 'multi_hop'
    """
    lowered = query.lower()
    figure_terms = {"figure", "chart", "image", "diagram", "graph", "plot", "shown"}
    table_terms = {"table", "row", "column", "cell", "entry"}

    has_figure = any(t in lowered for t in figure_terms)
    has_table = any(t in lowered for t in table_terms) or has_data_seeking_intent(query)
    # Multi-hop heuristic: query contains multiple clauses separated by "and"/"also"
    is_multi = bool(re.search(r"\band\b|\balso\b|\bas well\b", lowered))

    if is_multi:
        return "multi_hop"
    if has_figure and not has_table:
        return "figure_lookup"
    if has_table:
        return "table_lookup"
    return "prose"


# ── Region-type boost ─────────────────────────────────────────────────────────

_REGION_BOOST = 1.3
_BOOSTED_TYPES = frozenset(["table", "figure"])


def apply_region_boost(chunks: list[RetrievedChunk], query: str) -> list[RetrievedChunk]:
    """
    Apply a 1.3× score boost to table/figure chunks when the query has
    data-seeking intent (numbers, comparisons, lists).  Re-sorts by score.
    """
    if not has_data_seeking_intent(query):
        return chunks

    boosted: list[RetrievedChunk] = []
    for chunk in chunks:
        region_types: list[str] = chunk.metadata.get("region_types") or []
        if any(rt in _BOOSTED_TYPES for rt in region_types):
            boosted.append(
                RetrievedChunk(
                    chunk_id=chunk.chunk_id,
                    text=chunk.text,
                    metadata=chunk.metadata,
                    score=chunk.score * _REGION_BOOST,
                )
            )
        else:
            boosted.append(chunk)

    return sorted(boosted, key=lambda c: c.score, reverse=True)


# ── Parent-context expansion ──────────────────────────────────────────────────


def _parent_key(chunk: RetrievedChunk) -> tuple[str | None, str | None, str | None]:
    """Return (document_id, parent_title, parent_subtitle) as a grouping key."""
    meta = chunk.metadata
    return (
        meta.get("document_id"),
        meta.get("parent_title"),
        meta.get("parent_subtitle"),
    )


def expand_to_parent_context(
    top_chunks: list[RetrievedChunk],
    all_chunks: list[RetrievedChunk],
    *,
    sibling_threshold: int = 2,
) -> list[RetrievedChunk]:
    """
    For each parent section that has ≥ ``sibling_threshold`` chunks in
    ``top_chunks``, replace those individual chunks with the *full* parent
    section (all sibling chunks from ``all_chunks`` sharing the same
    document_id + parent_title + parent_subtitle).

    Chunks without a parent_title are returned as-is.
    The returned list preserves a stable reading order (original top-K order
    for non-expanded chunks; section chunks inserted at the position of the
    first sibling).
    """
    if not top_chunks or not all_chunks:
        return top_chunks

    # Count how many top-K chunks belong to each parent section
    section_hits: dict[tuple, list[RetrievedChunk]] = {}
    for chunk in top_chunks:
        key = _parent_key(chunk)
        if key[1] is None:  # no parent_title → skip grouping
            continue
        section_hits.setdefault(key, []).append(chunk)

    # Sections that meet the threshold → fetch all siblings from all_chunks
    expand_keys: set[tuple] = {
        key for key, hits in section_hits.items() if len(hits) >= sibling_threshold
    }

    if not expand_keys:
        return top_chunks

    # Build full-section maps keyed by parent
    full_sections: dict[tuple, list[RetrievedChunk]] = {}
    for chunk in all_chunks:
        key = _parent_key(chunk)
        if key in expand_keys:
            full_sections.setdefault(key, []).append(chunk)

    # Rebuild the output list, substituting expansions at the first sibling pos
    seen_expanded: set[tuple] = set()
    seen_ids: set[str] = set()
    result: list[RetrievedChunk] = []

    for chunk in top_chunks:
        key = _parent_key(chunk)
        if key in expand_keys:
            if key not in seen_expanded:
                seen_expanded.add(key)
                for sibling in full_sections.get(key, []):
                    if sibling.chunk_id not in seen_ids:
                        seen_ids.add(sibling.chunk_id)
                        result.append(sibling)
            # Skip duplicate siblings already added
        else:
            if chunk.chunk_id not in seen_ids:
                seen_ids.add(chunk.chunk_id)
                result.append(chunk)

    return result
