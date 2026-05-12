# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment Setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then fill in OPENAI_API_KEY
```

Required env vars: `OPENAI_API_KEY`. Optional: `OPENAI_BASE_URL`.

## Common Commands

```bash
# CLI pipeline
python main.py --preprocess            # OCR + freeze artifacts from data/raw/
python main.py --pdf PATH              # preprocess a specific PDF file (any path)
python main.py --index                 # build vector index from frozen artifacts
python main.py --ask "your question"   # query against the index
python main.py --force-preprocess      # re-run preprocessing even if artifacts exist

# Lint
ruff check .
ruff format .
```

## Architecture

Single CLI entry point (`python main.py --preprocess/--pdf/--index/--ask`). No API server on this branch.

**Pipeline flow (3 stages):**

### Stage 1 — `document_Process/` · PDF → frozen artifacts

`pipeline.py` orchestrates five sequential stages:

1. **`LoadStage`** (`stages/load.py`) — SHA-256 document ID, PDF render, PaddleOCR, PP-DocLayout_plus-L layout detection, region cropping. Outputs `LoadResult`.
2. **`OrderStage`** (`stages/order.py`) — LTR/RTL multi-column reading order sort. Outputs `OrderResult`.
3. **`VisualStage`** (`stages/visual.py`) — VLM figure/table descriptions (placeholder by default; enabled via `USE_VLM_SUMMARIES=true`, uses `vlm_model`). Outputs `list[VisualRegion]`.
4. **`HierarchyStage`** (`stages/hierarchy.py`) — `_build_blocks()` first (sets `item.region_id`), then `_assign_titles()` (propagates parent title/subtitle), then `_build_hierarchy()` (Document→Section→Block tree). Outputs `HierarchyResult`.
5. **`SummarizeStage`** (`stages/summarize.py`) — Character-budget chunking (1800 chars, 200 overlap) + optional async LLM section summaries. Section summaries are propagated back into each chunk's `metadata["section_summary"]` after generation. Outputs `SummarizeResult`.

Artifacts written to `data/processed/<document_id>/`:
- `document.json` — `ProcessedDocument` (processing metadata)
- `chunks.json` — `list[ProcessedChunk]`
- `manifest.json` — lightweight metadata

### Stage 2 — `rag/index.py` · frozen artifacts → vector index

Reads `chunks.json` → builds three embedding pools via `rag/chunk.py`:

- **Pool 0** (`documents.json`) — one record per document, embed text: `"{source_filename}: {document_summary}"` (falls back to aggregated section summary sentences when empty; minimum is filename alone)
- **Pool A** (`sections.json`) — one record per unique section, embed text: `"{section_title} — {parent_subtitle}: {section_summary}"` (subtitle/summary included when present)
- **Pool B** (`blocks.json`) — one record per block, embed text: `"[{section_title}] > {content}"` (+ adjacent figure description if present; + `[has visual crop: path]` for table/figure blocks with crop images). `section_summary` is NOT stored in Pool B records (it lives in Pool A). Neighbor expansion is section-boundary-capped: only blocks with the same `section_id` as the anchor are included.

All pools stored in `data/embedded/` as JSON (`JsonVectorStore`). `JsonVectorStore` caches rows in memory keyed by file `mtime` — both `query()` and `get_all_chunks()` share the cache, avoiding a double disk read per `--ask` call. `query()` accepts `filter_doc_ids: set[str] | None` for Pool-0-scoped retrieval.

### Stage 3 — `rag/qa.py` · question → answer

Five steps, no LLM until Step 4:

1. **Metric query detection** — if ≥2 metric/result terms appear in the query (e.g. "bleu", "score", "results", "table", "benchmark"), `is_metric_query=True` and the section pre-filter threshold is lowered from `SECTION_FILTER_THRESHOLD` to `METRIC_QUERY_THRESHOLD` (0.35) for this query only.
1.5. **Document pre-filter** (Pool 0) — cosine search `documents.json` → keep docs with score ≥ `DOCUMENT_FILTER_THRESHOLD` (0.45), up to `DOCUMENT_FILTER_TOP_K` (3). Empty result = global fallback. Controlled by `USE_DOCUMENT_PREFILTER=true`.
2. **Section pre-filter** — cosine search Pool A scoped to `candidate_doc_ids` → keep top 3 sections ≥ threshold. Name-match injection: any section whose title shares a ≥5-char word with the query is injected regardless of cosine score, up to `SECTION_FILTER_MAX`. Multi-doc diversity: if Pool 0 passed >1 doc, inject the best Pool A section from each under-represented doc. If no sections pass → global Pool B fallback.
3. **Block retrieval + boost** — search Pool B scoped to `candidate_doc_ids` and candidate sections → additive boosts (figure +0.15, table +0.10, adjacent +0.05, token overlap +0.01×count) → ±2 neighbor expansion → `list[BlockWindow]`.
4. **Synthesis** — one LLM call, context formatted as `[Section | Page | Type]`. If `USE_VLM_SUMMARIES=true` and blocks have crop images, crop images are resized to `VISION_MAX_IMAGE_PX` (LANCZOS JPEG) and sent as multimodal content; `VISION_SYNTHESIS_MODEL` is used for the model. Without vision blocks, `SYNTHESIS_MODEL` is used. If `USE_VLM_SUMMARIES=false` and a block has crop images, a note is appended to the context advising the LLM that a visual asset exists.
5. **Faithfulness gate** — optional second LLM call (`USE_FAITHFULNESS_CHECK=true`): verifies claims, strips unsupported ones. Source scores use the parent `BlockWindow.score` for neighbour blocks (which have no individual cosine score).

**Configuration** (`config.py`): `Settings` (pydantic-settings) reads all config from env vars / `.env`. Never call `os.getenv()` — always use `Settings`. `ensure_data_dirs(settings)` is called at startup, not inside `Settings.__init__`, so `Settings()` is safe to construct in tests without side effects.

## Key modules

| File | Role |
|------|------|
| `document_Process/pipeline.py` | Orchestrates the 5-stage preprocessing pipeline |
| `document_Process/models/legacy.py` | Frozen Pydantic models (`ProcessedChunk`, `ProcessedDocument`) — do not modify |
| `document_Process/models/internal.py` | Pipeline-internal dataclasses (`LoadResult`, `HierarchyResult`, etc.) |
| `rag/index.py` | `index_document()`, `index_all_documents()`, `_index_pool0()` |
| `rag/chunk.py` | `section_records_from_processed_chunks()`, `block_records_from_processed_chunks()`, `document_record_from_summary()` |
| `rag/retrieve.py` | `JsonVectorStore` (with mtime cache, filter_doc_ids), `DocumentRetriever` (document_store + section_store + block_store) |
| `rag/qa.py` | `answer_question()`, `QAResponse`, `SourceRef`, `BlockWindow`, metric/document pre-filter |
| `rag/embed.py` | `EmbeddingBackend` protocol + `OpenAIEmbeddingBackend` |
| `rag/faithfulness.py` | `FaithfulnessChecker` (optional gate) |
| `config.py` | `Settings` — all configuration lives here |

## Feature flags

| Variable | Default | Effect |
|----------|---------|--------|
| `USE_DOCUMENT_PREFILTER` | `true` | Pool 0 document-level pre-filter before Pool A |
| `DOCUMENT_FILTER_THRESHOLD` | `0.45` | Min cosine similarity for Pool 0 |
| `DOCUMENT_FILTER_TOP_K` | `3` | Max documents to pass Pool 0 |
| `USE_DOCUMENT_SUMMARY` | `true` | Generate document_summary during preprocessing (Pool 0 embed source) |
| `SECTION_FILTER_THRESHOLD` | `0.55` | Min cosine similarity for section pre-filter |
| `METRIC_QUERY_THRESHOLD` | `0.35` | Lowered threshold for numeric/metric queries |
| `DEFAULT_TOP_K` | `4` | Block windows returned to synthesis |
| `USE_FAITHFULNESS_CHECK` | `true` | Step 5 faithfulness gate |
| `FAST_QUERY_MODE` | `false` | Skip faithfulness gate + reduce neighbor expansion (±1) for faster queries |
| `SECTION_FILTER_MAX` | `8` | Max candidate sections after score-gap filter (within 0.12 of top score) |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding backend |
| `SYNTHESIS_MODEL` | `gpt-4.1-nano` | Synthesis + faithfulness LLM |
| `VISION_SYNTHESIS_MODEL` | `gpt-4o-mini` | LLM used for synthesis when crop images are present |
| `VISION_MAX_IMAGE_PX` | `800` | Long-edge pixel cap before base64 encoding (LANCZOS JPEG q85) |
| `VLM_MODEL` | `gpt-4o-mini` | Vision model for figure/table descriptions |
| `VLM_RETRY_MAX` | `3` | Max VLM retries with exponential backoff on 429 rate-limit errors |
| `USE_VLM_SUMMARIES` | `true` | GPT-4o-mini vision call per table/figure crop |
| `USE_DOCUMENT_INTELLIGENCE` | `true` | LLM section + document summarization during preprocessing |
| `FAST_MODE` | `false` | Skip all VLM + LLM preprocessing calls |
| `PREFER_CHROMA` | `false` | ChromaDB instead of JSON store |

## Known layout limitation

Two-column academic PDFs (e.g. conference papers): small-font section headers like "6 Results" may be classified as `text_block` rather than `title` by PP-DocLayout_plus-L. `HierarchyStage` runs two recovery passes:

1. **`_promote_title_candidates()`** (before `_assign_titles`) — uses two-column x-gap heuristics + height/pattern criteria to promote candidate headings.
2. **`_recover_titles()`** (after `_assign_titles`) — pattern-match pass over regions still in the "untitled" zone: if the text matches a numbered-section pattern (`\d+(\.\d+)*\s+\w`), is ≤60 chars, and spans >40% of page width, it is promoted to a title and propagated forward. Logged at DEBUG as `[Hierarchy] recovered title: "…" page N (pattern match)`.

Residual misdetected headers (e.g. multi-line headings or non-numbered appendix titles) that survive both passes are still mitigated by metric query detection + global Pool B supplement.
