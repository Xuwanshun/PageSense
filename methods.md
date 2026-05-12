# RAG System — Methods & Workflow Reference

> A comprehensive overview of the PDF document processing pipeline and the Retrieval-Augmented Generation (RAG) pipeline for this project.

---

## Table of Contents

1. [System Architecture Overview](#1-system-architecture-overview)
2. [Stage 1 — PDF Processing Pipeline](#2-stage-1--pdf-processing-pipeline)
   - [2.1 Five-Stage Service Chain](#21-five-stage-service-chain)
   - [2.2 Artifact Output Structure](#22-artifact-output-structure)
3. [Stage 2 — Vector Index Building](#3-stage-2--vector-index-building)
4. [Stage 3 — RAG Query Pipeline](#4-stage-3--rag-query-pipeline)
   - [4.1 Metric Query Detection](#41-metric-query-detection)
   - [4.2 Section Pre-Filter](#42-section-pre-filter)
   - [4.3 Block Retrieval & Boost](#43-block-retrieval--boost)
   - [4.4 Synthesis](#44-synthesis)
   - [4.5 Faithfulness Gate](#45-faithfulness-gate-optional)
5. [Feature Flags Reference](#5-feature-flags-reference)
6. [Key Data Structures](#6-key-data-structures)
7. [LLM Call Budget Per Query](#7-llm-call-budget-per-query)

---

## 1. System Architecture Overview

The system has three top-level stages, each independently runnable via the CLI:

```
┌─────────────────────────────────────────────────────────────────────┐
│                   PDF Documents (data/raw/ or --pdf PATH)           │
└──────────────────────────────────┬──────────────────────────────────┘
                                   │  python main.py --preprocess / --pdf PATH
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│              STAGE 1 · document_Process/                            │
│   Load → Order → Visual → Hierarchy → Summarize                    │
│   Output: frozen JSON artifacts  (data/processed/<doc_id>/)        │
└──────────────────────────────────┬──────────────────────────────────┘
                                   │  python main.py --index
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│              STAGE 2 · rag/index.py                                 │
│   Reads chunks.json → builds Pool 0 (documents) + Pool A (sections)│
│   + Pool B (blocks)                                                 │
│   Output: data/embedded/documents.json + sections.json + blocks.json│
└──────────────────────────────────┬──────────────────────────────────┘
                                   │  python main.py --ask "question"
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│              STAGE 3 · rag/qa.py                                    │
│   Metric detect → Doc pre-filter (Pool 0) → Section pre-filter    │
│   → Block retrieval + boost → Synthesis → Faithfulness gate        │
│   Output: QAResponse (answer + sources + faithfulness label)       │
└─────────────────────────────────────────────────────────────────────┘
```

**Entry points:**

| Command | Module | Description |
|---------|--------|-------------|
| `python main.py --preprocess` | `document_Process/pipeline.py` | OCR + freeze all PDFs in `data/raw/` |
| `python main.py --pdf PATH` | `document_Process/pipeline.py` | OCR + freeze a specific PDF at any path |
| `python main.py --index` | `rag/index.py` | Build two-pool vector index |
| `python main.py --ask "…"` | `rag/qa.py` | 5-step query pipeline |

---

## 2. Stage 1 — PDF Processing Pipeline

### 2.1 Five-Stage Service Chain

Each PDF passes through five deterministic stages. Intermediate results are frozen to disk so re-processing is skipped on the next run unless `--force-preprocess` is passed.

```
PDF file
  │
  ▼
┌──────────────────────────────────────────────────────────┐
│  LoadStage  (stages/load.py)                             │
│  • document_id = SHA-256(file bytes)                    │
│  • Renders each page → PNG  (pypdfium2, scale 3.0×)    │
│  • Text extraction: tries pdfplumber first (direct PDF  │
│    text layer); falls back to PaddleOCR only when the   │
│    page has ≤50 non-whitespace chars from pdfplumber    │
│  • PP-DocLayout_plus-L → layout regions (text/table/   │
│    figure) with deduplication                           │
│  • Crops table/figure regions to image files           │
│  Output: LoadResult                                     │
└──────────────────────┬───────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────┐
│  OrderStage  (stages/order.py)                           │
│  • Detects reading direction (LTR / RTL)               │
│  • Detects multi-column layout via x-gap analysis      │
│  • Sorts OCR items: (y-bucket, x, y) per page         │
│  • Assigns global reading_order attribute on each item │
│  Output: OrderResult                                    │
└──────────────────────┬───────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────┐
│  VisualStage  (stages/visual.py)                         │
│  • USE_VLM_SUMMARIES=false (default): emits placeholder │
│    text "[Figure on page N: type]" for each region     │
│  • USE_VLM_SUMMARIES=true: async GPT-4o-mini vision    │
│    calls (one per table/figure crop, concurrency-       │
│    limited by VLM_CONCURRENCY_LIMIT)                   │
│  Output: list[VisualRegion]  (inline_text + summary)   │
└──────────────────────┬───────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────┐
│  HierarchyStage  (stages/hierarchy.py)                   │
│  • _build_blocks() — matches OCR items to layout        │
│    regions via bbox overlap; sets item.region_id        │
│  • _promote_title_candidates() — promotes text_block   │
│    regions that look like headings in two-column PDFs  │
│  • _assign_titles() — propagates parent_title /        │
│    parent_subtitle from title regions to all            │
│    subsequent non-title regions (reading order)        │
│  • _recover_titles() — second pass after _assign_titles │
│    recovers numbered headers (e.g. "1 Introduction")   │
│    still in "untitled" zone: must match \d+(\.\d+)*\s+ │
│    pattern, ≤60 chars, bbox width > 40% of page width; │
│    logs "[Hierarchy] recovered title: …" at DEBUG      │
│  • _build_hierarchy() — groups blocks into             │
│    Document → Section → Block tree; inserts            │
│    VisualRegion blocks at correct reading position     │
│  Output: HierarchyResult                               │
└──────────────────────┬───────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────┐
│  SummarizeStage  (stages/summarize.py)                   │
│  • Splits OrderedTextBlocks into ProcessedChunks       │
│    (target ~1800 chars, 200-char overlap carry-forward)│
│  • Page boundaries AND section boundaries (parent_title│
│    change) force chunk splits; no overlap carry-forward│
│    across section boundaries                           │
│  • USE_DOCUMENT_INTELLIGENCE=true + API key: async LLM │
│    section summaries (synthesis_model); propagated     │
│    back into chunk.metadata["section_summary"]         │
│  • Document-level summary from aggregated section text │
│  Output: SummarizeResult (chunks + document_summary)   │
└──────────────────────┬───────────────────────────────────┘
                       │
                       ▼
              _export()
              → writes document.json, chunks.json, manifest.json
```

**Stage data flow:**

| Stage | Input | Output type | ML model |
|-------|-------|-------------|----------|
| Load | PDF path | `LoadResult` | PaddleOCR, PP-DocLayout_plus-L |
| Order | `LoadResult` | `OrderResult` | Geometric sort |
| Visual | `LoadResult` | `list[VisualRegion]` | GPT-4o-mini (opt.) |
| Hierarchy | Load + Order + Visual | `HierarchyResult` | — |
| Summarize | Load + Hierarchy | `SummarizeResult` | `synthesis_model` (opt.) |

---

### 2.2 Artifact Output Structure

```
data/processed/
└── <document_id>/          (SHA-256 of PDF bytes)
    ├── manifest.json        schema_version, status, chunk_count
    ├── document.json        ProcessedDocument + processing_summary
    └── chunks.json          list[ProcessedChunk]
```

**`ProcessedChunk` fields used by the RAG layer:**

| Field | Type | Description |
|-------|------|-------------|
| `chunk_id` | str | `"{doc_id}:chunk:{n}"` |
| `page_content` | str | Chunk text (the `text` field is no longer populated — always `""`) |
| `page_number` | int | Primary page |
| `region_types` | list[str] | `"text_block"` / `"table"` / `"figure"` / etc. |
| `crop_references` | list[str] | Paths to cropped figure/table PNG files |
| `metadata.parent_title` | str | Section heading from HierarchyStage |
| `metadata.parent_subtitle` | str | Subtitle heading (if present) |
| `metadata.section_summary` | str | LLM summary of section (if USE_DOCUMENT_INTELLIGENCE=true) |

`manifest.json` now also carries `document_summary` and `source_filename` so that indexing (`rag/index.py`) can read them without loading the heavier `document.json`.

---

## 3. Stage 2 — Vector Index Building

```
data/processed/  (all preprocessed documents)
  │
  ▼  For each document_id:
  │
  ├── _load_document_chunks()
  │   → ProcessedDocument + list[ProcessedChunk]
  │
  ├── _index_pool0()                             [Pool 0]
  │   → one DocumentRecord per document
  │   embed text: "{source_filename}: {document_summary}"
  │               (falls back to aggregated section summary sentences
  │               if document_summary is empty; minimum is filename)
  │   reads document_summary from manifest.json first; falls back to
  │   document.json for backward compatibility with old artifacts
  │   → upsert to documents.json  metadata: doc_id, source_filename,
  │     page_count, section_count
  │
  ├── section_records_from_processed_chunks()   [Pool A]
  │   → one SectionRecord per unique section
  │   embed text: "{section_title} — {subtitle}: {section_summary}"
  │               (subtitle/summary omitted when empty)
  │   → upsert to sections.json  metadata: pool="section", doc_id
  │
  └── block_records_from_processed_chunks()     [Pool B]
      → one BlockRecord per paragraph/table/figure block
      embed text: "[{section_title}] > {content}"
                  (+ " | {figure_desc}" if has_adjacent_figure)
                  (+ " [has visual crop: {path}]" for table/figure
                     blocks with crop_references)
      → upsert to blocks.json  metadata: pool="block", doc_id,
        crop_references  (section_summary NOT stored here — already in Pool A)
```

**Block type classification** (from `region_types`):

| `region_types` contains | `block_type` |
|------------------------|--------------|
| `figure`, `chart`, `figure_description` | `"figure_description"` |
| `table` | `"table"` |
| `caption`, `figure_caption`, `table_caption` | `"caption"` |
| `list` | `"list"` |
| anything else | `"paragraph"` |

**`JsonVectorStore` caching:** rows are deserialized once per file modification. Both `query()` and `get_all_chunks()` share the in-memory cache (keyed by `mtime`), eliminating the double JSON parse that would otherwise occur on every `--ask` call.

---

## 4. Stage 3 — RAG Query Pipeline

The pipeline in `rag/qa.py` runs five steps. The first two involve no LLM calls:

```
User question
  │
  ▼
embed question  (text-embedding-3-small)
  │
  ▼  Step 1
┌─────────────────────────────────────────────────────────────────────┐
│  Metric Query Detection                                             │
│  METRIC_TERMS = {bleu, score, accuracy, performance, result,        │
│    results, benchmark, metric, metrics, percentage, wer, rouge,     │
│    f1, precision, recall, table, comparison, versus, vs}            │
│  If ≥2 terms match → is_metric_query = True                        │
│    → effective threshold = METRIC_QUERY_THRESHOLD (0.35)           │
│    → log: "[QA] metric query detected — lowering threshold to 0.35"│
│  Else effective threshold = SECTION_FILTER_THRESHOLD (0.55)        │
└─────────────────────────────────────────────────────────────────────┘
  │
  ▼  Step 1.5
┌─────────────────────────────────────────────────────────────────────┐
│  Document Pre-Filter  (Pool 0 search — USE_DOCUMENT_PREFILTER=true)│
│  • cosine search Pool 0 (documents.json), top DOCUMENT_FILTER_TOP_K│
│    × 2 candidates                                                   │
│  • keep docs with score ≥ DOCUMENT_FILTER_THRESHOLD (0.45)         │
│  • cap at DOCUMENT_FILTER_TOP_K (3) passing docs                   │
│  • if none pass → candidate_doc_ids = {} (global fallback, same    │
│    as USE_DOCUMENT_PREFILTER=false)                                 │
│  • logs: "[QA] document pre-filter: N doc(s) pass threshold 0.45   │
│    — ['filename.pdf', ...]"                                         │
│                                                                     │
│  Output: candidate_doc_ids: set[str]  (empty = no filter)          │
└─────────────────────────────────────────────────────────────────────┘
  │
  ▼  Step 2
┌─────────────────────────────────────────────────────────────────────┐
│  Section Pre-Filter  (Pool A search, scoped to candidate_doc_ids)   │
│  • cosine search Pool A, filtered to candidate_doc_ids              │
│  • keep sections with score ≥ effective threshold                   │
│  • cap at 3 candidate sections                                      │
│  • if none pass → candidate_sections = {} (global fallback)         │
│                                                                     │
│  Multi-doc diversity injection (Fix I):                             │
│  If Pool 0 passed >1 doc, check that Pool A selected sections from  │
│  each candidate doc. For any missing doc, inject its highest-scoring│
│  Pool A section regardless of threshold, ensuring all Pool-0-passing│
│  documents are represented in retrieval.                            │
│                                                                     │
│  Output: candidate_section_ids: set[str]                            │
└─────────────────────────────────────────────────────────────────────┘
  │
  ▼  Step 3
┌─────────────────────────────────────────────────────────────────────┐
│  Block Retrieval + Boost  (Pool B search)                           │
│                                                                     │
│  Search Pool B, scoped to candidate_section_ids                     │
│  (fallback: global search if candidate_sections is empty)           │
│                                                                     │
│  Metric query supplement: always add top 2 globally-scored Pool B  │
│  blocks that fall outside the candidate sections. This catches      │
│  results/table blocks whose section header the layout model         │
│  misdetected (e.g. "6 Results" absorbed into "5.4 Regularization"; │
│  its section Pool A score is ~0.16 but its blocks score ~0.61).    │
│                                                                     │
│  Additive score boosts:                                             │
│  +0.15  if block_type == "figure_description" AND visual query     │
│  +0.10  if block_type == "table" AND data query                    │
│  +0.05  if has_adjacent_figure == true                             │
│  +0.01 × overlap_count  (token overlap, always on)                │
│                                                                     │
│  Visual query terms:  figure, chart, diagram, shows, illustrated,  │
│                       image, plot, visual                           │
│  Data query terms:    how many, how much, compare, total, percent, │
│                       list, table, count, number, rate, breakdown  │
│                                                                     │
│  For each top_k anchor block:                                       │
│  • fetch ±2 neighbors within same section (by block_index)         │
│  • deduplicate by block_id within the window                       │
│  • preserve reading order                                           │
│  Output: list[BlockWindow]  (anchor + neighbors, reading order)    │
└─────────────────────────────────────────────────────────────────────┘
  │
  ▼  Step 4  (1 LLM call)
┌─────────────────────────────────────────────────────────────────────┐
│  Synthesis                                                          │
│                                                                     │
│  Context format per block:                                          │
│  [Section: {section_title} | Page {page} | Type: {block_type}]    │
│  {content}                                                          │
│  [Note: has visual crop at {path} — enable USE_VLM_SUMMARIES=true] │
│    (appended when crop_references present + USE_VLM_SUMMARIES=false)│
│                                                                     │
│  System prompt: answer from retrieved blocks only, cite evidence.  │
│  Model: SYNTHESIS_MODEL (default gpt-4.1-nano)                     │
│  Output: answer string                                              │
└─────────────────────────────────────────────────────────────────────┘
  │
  ▼  Step 5  (1 LLM call — optional)
┌─────────────────────────────────────────────────────────────────────┐
│  Faithfulness Gate  (USE_FAITHFULNESS_CHECK=true)                   │
│                                                                     │
│  FaithfulnessChecker.check():                                       │
│  • Breaks answer into individual claims                             │
│  • Checks each: SUPPORTED / INFERRED / UNSUPPORTED                 │
│  • Returns overall_verdict + recommended_action                     │
│                                                                     │
│  If recommended_action != "return_as_is":                          │
│    FaithfulnessChecker.correct():                                   │
│    • Removes/hedges unsupported claims                              │
│    • faithfulness label → "UNSUPPORTED_REMOVED"                    │
│  Else: faithfulness label → "FAITHFUL"                              │
│                                                                     │
│  Source scores: neighbour blocks (score=0.0 from get_all_chunks)   │
│  report the parent BlockWindow.score instead of 0.0               │
└─────────────────────────────────────────────────────────────────────┘
  │
  ▼
QAResponse
  question     str
  answer       str
  sources      list[SourceRef]  (block_id, section_title, page, score, block_type)
  faithfulness str | None       "FAITHFUL" | "UNSUPPORTED_REMOVED" | None
```

---

### 4.1 Metric Query Detection

Before running the Pool A section pre-filter, the query is scanned for numeric/metric intent terms. If ≥2 terms match, the section filter threshold is dynamically lowered to `METRIC_QUERY_THRESHOLD` (0.35) for this query only.

A secondary check also triggers metric mode when the query contains any decimal number (`\d+\.\d+`, e.g. "93.8") or percentage (`\d+%`), regardless of how many METRIC_TERMS appear.

**Why:** In two-column academic PDFs, section headers like "6 Results" are often misclassified as `text_block` by the layout model. Those blocks get absorbed into the preceding section (e.g. "5.4 Regularization"), whose Pool A embedding has nothing to do with results. The Pool A score for that section against a BLEU/metric query is ~0.16 — well below any reasonable threshold. Lowering the threshold admits more candidate sections, and the global Pool B supplement injects the actual results blocks directly from cosine search (where the block text does mention BLEU scores and scores ~0.61).

---

### 4.2 Section Pre-Filter

Embeds the raw user question and searches Pool A (section summaries). This is a coarse pre-filter: it identifies which sections of the document are topically relevant before doing fine-grained block search.

- **Normal threshold:** `SECTION_FILTER_THRESHOLD=0.55`
- **Metric threshold:** `METRIC_QUERY_THRESHOLD=0.35` (activated by Step 1)
- **Cap:** all sections within 0.12 cosine similarity of the top-scoring section, with a hard maximum of `SECTION_FILTER_MAX` (default 8). Minimum 1 when at least one section clears the threshold.
- **Name-match injection:** after the cosine filter, any section whose title contains a word of ≥5 characters that also appears in the query is injected into the candidate set regardless of cosine score, up to `SECTION_FILTER_MAX`. This recovers sections like "1 Introduction" when the query says "introduction" but the section's cosine score is below threshold due to sparse section content.
- **Fallback:** if no section clears the threshold, skip scoping and search all blocks globally

---

### 4.3 Block Retrieval & Boost

Searches Pool B (individual blocks) scoped to the candidate sections. Applies score boosts without re-embedding:

| Boost | Amount | Condition |
|-------|--------|-----------|
| Figure boost | +0.15 | `block_type == "figure_description"` AND query contains visual terms |
| Table boost | +0.10 | `block_type == "table"` AND query contains data terms |
| Adjacent figure | +0.05 | `has_adjacent_figure == true` |
| Token overlap | +0.01 × n | n query tokens found in block content |

For metric queries, up to 2 globally-scored blocks outside the candidate sections are appended before boosting, ensuring misattributed results blocks can still be retrieved.

After boosting, for each of the top-k anchor blocks, neighbors at `block_index ±2` within the same section are fetched and grouped into a `BlockWindow`. Neighbor expansion is section-boundary-capped: only blocks with the same `section_id` as the anchor are included. Results are deduplicated by block_id and sorted by reading order.

---

### 4.4 Synthesis

Builds a context string from the `BlockWindow` list and makes a single LLM call. Each block is formatted as:

```
[Section: Introduction | Page 3 | Type: paragraph]
The document describes three key phases of the process...

[Section: 3.2.1 Scaled Dot-Product Attention | Page 4 | Type: figure_description]
[Note: this block has an associated figure/table image at data/processed/.../crops/region_42.png
 — enable USE_VLM_SUMMARIES=true to include visual descriptions]
```

When `USE_VLM_SUMMARIES=true`, the synthesis LLM call is made as a multimodal (vision) request: the first crop image per block is resized so the long edge is at most `VISION_MAX_IMAGE_PX` (default 800 px) using `Image.LANCZOS`, saved as JPEG quality 85 to a `BytesIO` buffer, then base64-encoded. This keeps payloads small (~15–30 KB vs. ~291 KB for raw PNG crops). When at least one image content block is present, `VISION_SYNTHESIS_MODEL` (default `gpt-4o-mini`) is used instead of `SYNTHESIS_MODEL`, since `gpt-4o-mini` has full vision capability. When there are no vision blocks, `SYNTHESIS_MODEL` is used as usual.

The system prompt instructs the model to answer only from the provided blocks and to cite `(Section: <title>, Page <n>)` for every claim.

---

### 4.5 Faithfulness Gate (optional)

Controlled by `USE_FAITHFULNESS_CHECK=true`. Adds at most one LLM call (check) plus optionally a second (correction).

`FaithfulnessChecker.check()` breaks the answer into claims and classifies each:
- **SUPPORTED** — directly stated or numerically entailed by a source block
- **INFERRED** — reasonable inference, not directly stated
- **UNSUPPORTED** — no basis in source blocks (hallucination)

If the verdict is not `return_as_is`, `FaithfulnessChecker.correct()` rewrites the answer: removes unsupported claims, hedges inferred ones, and returns the corrected text with label `"UNSUPPORTED_REMOVED"`.

---

## 5. Feature Flags Reference

All flags are read from environment variables or `.env` via pydantic-settings.

**RAG flags:**

| Variable | Default | Effect |
|----------|---------|--------|
| `USE_DOCUMENT_PREFILTER` | `true` | Pool 0 document-level pre-filter before Pool A |
| `DOCUMENT_FILTER_THRESHOLD` | `0.45` | Min cosine similarity for Pool 0 |
| `DOCUMENT_FILTER_TOP_K` | `3` | Max documents to pass Pool 0 |
| `SECTION_FILTER_THRESHOLD` | `0.55` | Min cosine similarity for Pool A pre-filter (normal queries) |
| `METRIC_QUERY_THRESHOLD` | `0.35` | Lowered threshold when ≥2 metric terms detected in query |
| `DEFAULT_TOP_K` | `4` | Block windows passed to synthesis |
| `USE_FAITHFULNESS_CHECK` | `true` | Step 5 claim verification + correction |
| `FAST_QUERY_MODE` | `false` | Skip faithfulness gate + reduce neighbor expansion (±1) |
| `SECTION_FILTER_MAX` | `8` | Max candidate sections after score-gap filter |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding backend |
| `SYNTHESIS_MODEL` | `gpt-4.1-nano` | Synthesis + faithfulness LLM |
| `VISION_SYNTHESIS_MODEL` | `gpt-4o-mini` | LLM used for synthesis when crop images are present |
| `VISION_MAX_IMAGE_PX` | `800` | Long-edge pixel cap before base64 encoding (LANCZOS JPEG q85) |
| `PREFER_CHROMA` | `false` | ChromaDB instead of JSON store |

**Preprocessing flags:**

| Variable | Default | Effect |
|----------|---------|--------|
| `USE_VLM_SUMMARIES` | `false` | GPT-4o-mini vision call per table/figure crop |
| `VLM_MODEL` | `gpt-4o-mini` | Vision model for figure/table descriptions |
| `USE_DOCUMENT_INTELLIGENCE` | `true` | LLM section + document summarization |
| `FAST_MODE` | `false` | Skip VLM + LLM summarization stages |
| `PREPROCESS_CHUNK_SIZE` | `1800` | Target chunk size (characters) |
| `PREPROCESS_CHUNK_OVERLAP` | `200` | Overlap between consecutive chunks |
| `PDF_RENDER_SCALE` | `3.0` | Page image scale (higher = better OCR quality) |
| `VLM_CONCURRENCY_LIMIT` | `4` | Max parallel VLM calls during preprocessing |
| `LLM_CONCURRENCY_LIMIT` | `8` | Max parallel LLM calls during preprocessing |

---

## 6. Key Data Structures

### Preprocessing output

```
ProcessedChunk          (frozen artifact — do not modify the schema)
  chunk_id              "doc_id:chunk:N"
  text                  raw text
  page_content          cleaned text (preferred by rag/)
  page_number           int
  region_types          list[str]  ("text_block" | "table" | "figure" | ...)
  crop_references       list[str]  (paths to cropped PNG files for visual regions)
  metadata
    .parent_title       str  (section heading from HierarchyStage)
    .parent_subtitle    str  (subtitle heading, if present)
    .section_summary    str  (LLM summary if USE_DOCUMENT_INTELLIGENCE=true)
```

### Indexing records

```
SectionRecord  (Pool A)
  section_id   str   (hash of doc_id + section_title)
  text         str   "{section_title} — {subtitle}: {section_summary}"
                     (subtitle/summary omitted when empty)
  metadata     dict  pool="section", doc_id, section_id, section_subtitle,
                     section_summary, page_start, page_end

BlockRecord  (Pool B)
  block_id     str   "{chunk_id}:block:{local_index}"
  text         str   "[{section_title}] > {content}"
                     (+ " | {fig_desc}" if has_adjacent_figure)
                     (+ " [has visual crop: {path}]" for table/figure)
  metadata     dict  pool="block", doc_id, section_id, block_type,
                     has_adjacent_figure, block_index, page, content,
                     crop_references
```

### Retrieval & response types

```
RetrievedChunk          (output of JsonVectorStore.query())
  chunk_id              str
  text                  str
  metadata              dict
  score                 float  (cosine similarity, or 0.0 for get_all_chunks rows)

BlockWindow             (groups anchor block with its ±2 neighbors)
  anchor_block_id       str
  section_title         str
  page                  int
  blocks                list[RetrievedChunk]  (reading order)
  score                 float  (anchor cosine + boosts)

SourceRef               (one entry per block in the response)
  block_id              str
  section_title         str
  page                  int
  score                 float  (window.score used for neighbour blocks)
  block_type            str

QAResponse              (final pipeline output)
  question              str
  answer                str
  sources               list[SourceRef]
  faithfulness          str | None  ("FAITHFUL" | "UNSUPPORTED_REMOVED" | None)
```

---

## 7. LLM Call Budget Per Query

| Step | Calls | Condition |
|------|-------|-----------|
| Synthesis | 1 | always |
| Faithfulness check | 1 | `USE_FAITHFULNESS_CHECK=true` |
| Faithfulness correction | 1 | `USE_FAITHFULNESS_CHECK=true` AND unsupported claims found |
| **Maximum total** | **3** | faithfulness on + correction triggered |
| **Minimum total** | **1** | all optional features off |

**Typical configuration (faithfulness off):** 1 LLM call + 1 embedding call per query.

**Preprocessing budget** (per document, one-time):

| Step | Calls | Condition |
|------|-------|-----------|
| VLM per visual region | 1 each | `USE_VLM_SUMMARIES=true` |
| Section summarization | 1 per section | `USE_DOCUMENT_INTELLIGENCE=true` + API key |
| Document summarization | 1 | `USE_DOCUMENT_INTELLIGENCE=true` + API key |
