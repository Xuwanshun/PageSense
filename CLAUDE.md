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

Reads `chunks.json` → builds two embedding pools via `rag/chunk.py`:

- **Pool A** (`sections.json`) — one record per unique section, embed text: `"{section_title} — {parent_subtitle}: {section_summary}"` (subtitle/summary included when present)
- **Pool B** (`blocks.json`) — one record per block, embed text: `"[{section_title}] > {content}"` (+ adjacent figure description if present; + `[has visual crop: path]` for table/figure blocks with crop images)

Both pools stored in `data/embedded/` as JSON (`JsonVectorStore`). `JsonVectorStore` caches rows in memory keyed by file `mtime` — both `query()` and `get_all_chunks()` share the cache, avoiding a double disk read per `--ask` call.

### Stage 3 — `rag/qa.py` · question → answer

Four steps, no LLM until Step 3:

1. **Metric query detection** — if ≥2 metric/result terms appear in the query (e.g. "bleu", "score", "results", "table", "benchmark"), `is_metric_query=True` and the section pre-filter threshold is lowered from `SECTION_FILTER_THRESHOLD` to `METRIC_QUERY_THRESHOLD` (0.35) for this query only.
2. **Section pre-filter** — embed question → cosine search Pool A → keep top 3 sections ≥ threshold. If none pass → global Pool B fallback. For metric queries, additionally supplement block candidates with the top 2 globally-scored Pool B blocks outside the candidate sections (catches results/metrics blocks whose section header was misdetected by the layout model).
3. **Block retrieval + boost** — search Pool B scoped to candidate sections → additive boosts (figure +0.15, table +0.10, adjacent +0.05, token overlap +0.01×count) → ±2 neighbor expansion → `list[BlockWindow]`.
4. **Synthesis** — one LLM call (`SYNTHESIS_MODEL`), context formatted as `[Section | Page | Type]`. If a block has crop images and `USE_VLM_SUMMARIES=false`, a note is appended to the context advising the LLM that a visual asset exists.
5. **Faithfulness gate** — optional second LLM call (`USE_FAITHFULNESS_CHECK=true`): verifies claims, strips unsupported ones. Source scores use the parent `BlockWindow.score` for neighbour blocks (which have no individual cosine score).

**Configuration** (`config.py`): `Settings` (pydantic-settings) reads all config from env vars / `.env`. Never call `os.getenv()` — always use `Settings`. `ensure_data_dirs(settings)` is called at startup, not inside `Settings.__init__`, so `Settings()` is safe to construct in tests without side effects.

## Key modules

| File | Role |
|------|------|
| `document_Process/pipeline.py` | Orchestrates the 5-stage preprocessing pipeline |
| `document_Process/models/legacy.py` | Frozen Pydantic models (`ProcessedChunk`, `ProcessedDocument`) — do not modify |
| `document_Process/models/internal.py` | Pipeline-internal dataclasses (`LoadResult`, `HierarchyResult`, etc.) |
| `rag/index.py` | `index_document()`, `index_all_documents()` |
| `rag/chunk.py` | `section_records_from_processed_chunks()`, `block_records_from_processed_chunks()` |
| `rag/retrieve.py` | `JsonVectorStore` (with mtime cache), `DocumentRetriever`, boost/expand helpers |
| `rag/qa.py` | `answer_question()`, `QAResponse`, `SourceRef`, `BlockWindow`, metric query detection |
| `rag/embed.py` | `EmbeddingBackend` protocol + `OpenAIEmbeddingBackend` |
| `rag/faithfulness.py` | `FaithfulnessChecker` (optional gate) |
| `config.py` | `Settings` — all configuration lives here |

## Feature flags

| Variable | Default | Effect |
|----------|---------|--------|
| `SECTION_FILTER_THRESHOLD` | `0.55` | Min cosine similarity for section pre-filter |
| `METRIC_QUERY_THRESHOLD` | `0.35` | Lowered threshold for numeric/metric queries |
| `DEFAULT_TOP_K` | `4` | Block windows returned to synthesis |
| `USE_FAITHFULNESS_CHECK` | `false` | Step 5 faithfulness gate |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding backend |
| `SYNTHESIS_MODEL` | `gpt-4.1-nano` | Synthesis + faithfulness LLM |
| `VLM_MODEL` | `gpt-4o-mini` | Vision model for figure/table descriptions |
| `USE_VLM_SUMMARIES` | `false` | GPT-4o-mini vision call per table/figure crop |
| `USE_DOCUMENT_INTELLIGENCE` | `true` | LLM section + document summarization during preprocessing |
| `FAST_MODE` | `false` | Skip all VLM + LLM preprocessing calls |
| `PREFER_CHROMA` | `false` | ChromaDB instead of JSON store |

## Known layout limitation

Two-column academic PDFs (e.g. conference papers): small-font section headers like "6 Results" may be classified as `text_block` rather than `title` by PP-DocLayout_plus-L. Those blocks get absorbed into the preceding section. The metric query detection + global Pool B supplement mitigates retrieval failures for result/metric queries caused by this misdetection.
