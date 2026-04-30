# RAG System — Methods & Workflow Reference

> A comprehensive overview of the PDF document processing pipeline and the Retrieval-Augmented Generation (RAG) pipeline for this project.

---

## Table of Contents

1. [System Architecture Overview](#1-system-architecture-overview)
2. [Stage 1 — PDF Processing Pipeline](#2-stage-1--pdf-processing-pipeline)
   - [2.1 Service Chain](#21-service-chain)
   - [2.2 Document Intelligence (optional)](#22-document-intelligence-optional)
   - [2.3 Artifact Output Structure](#23-artifact-output-structure)
3. [Stage 2 — Vector Index Building](#3-stage-2--vector-index-building)
4. [Stage 3 — RAG Query Pipeline](#4-stage-3--rag-query-pipeline)
   - [4.1 Query Enhancement](#41-query-enhancement)
   - [4.2 Retrieval Strategies](#42-retrieval-strategies)
   - [4.3 Reranking (Two-Pass)](#43-reranking-two-pass)
   - [4.4 Multi-Agent Routing & Synthesis](#44-multi-agent-routing--synthesis)
   - [4.5 Context Compression](#45-context-compression)
   - [4.6 Faithfulness Check & Correction](#46-faithfulness-check--correction)
5. [API Layer](#5-api-layer)
6. [Feature Flags Reference](#6-feature-flags-reference)
7. [Key Data Structures](#7-key-data-structures)
8. [LLM Call Budget Per Query](#8-llm-call-budget-per-query)

---

## 1. System Architecture Overview

The system has three top-level stages, each independently runnable:

```
┌─────────────────────────────────────────────────────────────────────┐
│                        PDF Documents (data/raw/)                    │
└──────────────────────────────────┬──────────────────────────────────┘
                                   │  python main.py --preprocess
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│              STAGE 1 · document_Process/                            │
│   OCR → Layout → Association → Chunking → Visual Summaries         │
│   Optional: Document Intelligence, VLM Enrichment                   │
│   Output: frozen JSON artifacts  (data/processed/<doc_id>/)         │
└──────────────────────────────────┬──────────────────────────────────┘
                                   │  python main.py --index
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│              STAGE 2 · rag/retrieve.py                              │
│   Embed chunks → Upsert to vector store                             │
│   Output: data/embedded/store.json  (or ChromaDB)                  │
└──────────────────────────────────┬──────────────────────────────────┘
                                   │  python main.py --ask "question"
                                   │  POST /query
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│              STAGE 3 · rag/qa.py                                    │
│   Query Enhancement → Retrieval → Reranking → Synthesis            │
│   Optional: Context Compression, Faithfulness Check                 │
│   Output: MultiAgentQAResponse (answer + sources + routing info)   │
└─────────────────────────────────────────────────────────────────────┘
```

**Entry Points:**

| Command / Endpoint | Module | Description |
|--------------------|--------|-------------|
| `python main.py --preprocess` | `document_Process/pipeline.py` | OCR + freeze all PDFs in `data/raw/` |
| `python main.py --index` | `rag/retrieve.py` | Embed chunks + build vector store |
| `python main.py --ask "…"` | `rag/qa.py` | Query pipeline (CLI) |
| `POST /documents/preprocess` | `api/routers/documents.py` | Upload + preprocess a single PDF |
| `POST /documents/index` | `api/routers/documents.py` | Rebuild entire vector store |
| `POST /query` | `api/routers/query.py` | Query pipeline (HTTP) |

---

## 2. Stage 1 — PDF Processing Pipeline

### 2.1 Service Chain

Each PDF passes through a deterministic six-service pipeline. Intermediate results are frozen to disk so re-processing is skipped if artifacts already exist.

```
PDF file
  │
  ▼
┌──────────────────────────────────────────────────────────┐
│  DocumentLoaderService                                   │
│  • Computes document_id = SHA256(file bytes)            │
│  • Renders each page → PNG (scale: 3.0×, via pypdfium2) │
│  Output: LoadedDocument  (page images + metadata)       │
└──────────────────────┬───────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────┐
│  OCRService  (PaddleOCR)                                 │
│  • Extracts text items + bounding boxes per page        │
│  • Each item: item_id, text, bbox, confidence           │
│  Output: list[OCRPageResult]                            │
└──────────────────────┬───────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────┐
│  ReadingOrderService                                     │
│  • Sorts OCR items top→bottom, left→right               │
│  • Assigns a global document-level reading order        │
│  Output: reading_order dict  (order_item_ids per page)  │
└──────────────────────┬───────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────┐
│  LayoutDetectionService  (PaddleX PP-DocLayout_plus-L)  │
│  • Detects layout regions: text_block / table / figure  │
│  • Deduplicates overlapping bounding boxes              │
│  Output: list[LayoutRegion]  (bbox + type + confidence) │
└──────────────────────┬───────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────┐
│  AssociationService                                      │
│  • Matches OCR items to layout regions via bbox overlap │
│  • Groups matched items into OrderedTextBlock per region│
│  • Fallback: line-based grouping if no region matches   │
│  Output: list[OrderedTextBlock] + full_ordered_text     │
└──────────────────────┬───────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────┐
│  CroppingService                                         │
│  • Extracts table/figure crops from page PNG images     │
│  • Saves to crops/tables/ and crops/figures/            │
│  Output: list[CroppedRegionAsset]  (crop image paths)  │
└──────────────────────┬───────────────────────────────────┘
                       │
             ┌─────────┴─────────┐
             │  Intelligence?    │  use_document_intelligence
             │     (optional)    │  use_vlm_summaries
             └─────────┬─────────┘
                       │  YES
                       ▼
              ┌─────────────────┐
              │ See §2.2        │
              └────────┬────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────┐
│  build_chunks()                                          │
│  • Splits OrderedTextBlocks into ProcessedChunks        │
│  • Default: ~1800 chars per chunk, 200-char overlap     │
│  • Adaptive: strategy chosen per-doc (if intelligence)  │
│  • Each chunk carries: region_ids, page_number,         │
│    parent_title, parent_subtitle (if intelligence)      │
│  Output: list[ProcessedChunk]                           │
└──────────────────────┬───────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────┐
│  build_visual_summaries()                                │
│  • One VisualRegionSummary per table/figure region      │
│  • summary_text: overlapping OCR text (fallback) or    │
│    VLM description (if use_vlm_summaries)              │
│  Output: list[VisualRegionSummary]                      │
└──────────────────────┬───────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────┐
│  export_artifacts()                                      │
│  • Writes all JSON files to data/processed/<doc_id>/    │
│  Output: frozen artifact directory  (§2.3)              │
└──────────────────────────────────────────────────────────┘
```

**Service-level data flow:**

| Step | Input | Output | Key Model |
|------|-------|--------|-----------|
| Load | PDF path | `LoadedDocument` | — |
| OCR | Page PNGs | `list[OCRPageResult]` | PaddleOCR |
| Reading Order | `OCRPageResult` list | Reading order dict | Geometric sort |
| Layout Detection | Page PNGs + OCR | `list[LayoutRegion]` | PP-DocLayout_plus-L |
| Association | OCR + order + regions | `list[OrderedTextBlock]` | Bbox overlap |
| Cropping | Page PNGs + regions | `list[CroppedRegionAsset]` | PIL crop |
| Chunking | `OrderedTextBlock` list | `list[ProcessedChunk]` | Character splitter |
| Visual Summaries | Regions + chunks + crops | `list[VisualRegionSummary]` | GPT-4o (opt.) |

---

### 2.2 Document Intelligence (optional)

Activated by `USE_DOCUMENT_INTELLIGENCE=true`. Runs after Cropping, before Chunking.

```
Regions + OrderedTextBlocks
  │
  ▼
┌──────────────────────────────────────────────────────────┐
│  Step A · _assign_titles()                               │
│  • Identifies title/subtitle layout regions             │
│  • Stamps parent_title + parent_subtitle onto all       │
│    subsequent non-title regions (reading order)         │
└──────────────────────┬───────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────┐
│  Step B · _group_into_sections()                         │
│  • Groups regions by parent_title                       │
│  • Produces Section objects with flat_text (≤2000 chars)│
│  • Counts tables/figures per section                    │
└──────────────────────┬───────────────────────────────────┘
                       │
            ┌──────────┴──────────┐
            │  use_vlm_summaries? │
            └──────────┬──────────┘
                  YES  │
                       ▼
              ┌──────────────────────────────┐
              │  Step C · _read_visual()     │
              │  • GPT-4o vision API call    │
              │    per table/figure crop     │
              │  • Returns: type, summary,   │
              │    key_finding, data, conf.  │
              └──────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────┐
│  Step D · _summarize_section()  (gpt-4o-mini)           │
│  • Generates section-level summaries                    │
│  • Builds document descriptor:                          │
│    summary, topics, doc_type, domain,                   │
│    visual_heavy, data_heavy, likely_questions           │
│  • Embeds descriptor summary → summary_embedding        │
│    (stored in document.json, used for doc pre-filter)  │
└──────────────────────┬───────────────────────────────────┘
                       │
            ┌──────────┴──────────┐
            │ use_adaptive_       │
            │    chunking?        │
            └──────────┬──────────┘
                  YES  │
                       ▼
              ┌──────────────────────────────────────────┐
              │  Step E · _decide_strategy()             │
              │  (gpt-4o-mini)                           │
              │  Strategies:                             │
              │  • semantic_section (default)            │
              │  • layout_aware (table-dominant docs)    │
              │  • recursive_large (unstructured docs)   │
              │  • semantic_fixed                        │
              │  Output: chunk_size, overlap, strategy  │
              └──────────────────────────────────────────┘
```

---

### 2.3 Artifact Output Structure

After preprocessing, every document produces a frozen artifact directory:

```
data/processed/
└── <document_id>/          (SHA256 of PDF bytes)
    ├── manifest.json        schema, pipeline_stage, status
    ├── document.json        ProcessedDocument  +  descriptor  +  summary_embedding
    ├── chunks.json          list[ProcessedChunk]
    ├── ocr.json             list[OCRPageResult]
    ├── layout.json          regions + associations
    ├── reading_order.json   document-order item IDs
    ├── cropped_assets.json  list[CroppedRegionAsset]
    ├── visual_summaries.json list[VisualRegionSummary]  (table/figure descriptions)
    ├── metadata.json        timing + service versions
    ├── crops/
    │   ├── tables/          table_<region_id>.png
    │   └── figures/         figure_<region_id>.png
    └── source/
        ├── <original.pdf>
        └── pages/
            ├── page_1.png
            ├── page_2.png
            └── ...
```

**ProcessedChunk metadata fields:**

| Field | Type | Description |
|-------|------|-------------|
| `chunk_id` | str | Unique ID (doc-scoped sequence) |
| `text` | str | Cleaned chunk text |
| `page_number` | int | Primary page |
| `source_region_ids` | list[str] | Layout regions this chunk spans |
| `region_types` | list[str] | `text_block`, `table`, `figure` |
| `parent_title` | str | Section title (if intelligence enabled) |
| `parent_subtitle` | str | Subsection title (if intelligence enabled) |
| `document_id` | str | Parent document |
| `source_filename` | str | Original PDF filename |
| `crop_asset_ids` | list[str] | Linked crop image asset IDs |

---

## 3. Stage 2 — Vector Index Building

```
data/processed/  (all preprocessed documents)
  │
  ▼  For each document_id:
  │
  ├── load chunks.json → list[ProcessedChunk]
  │
  ├── chunk_records_from_processed_chunks()
  │   → list[ChunkRecord]  (chunk_id + text + metadata)
  │
  ├── embedding_backend.embed_texts([chunk.text, ...])
  │   → list[list[float]]  (via text-embedding-3-small)
  │
  └── vector_store.upsert(records, embeddings)
      → updates data/embedded/store.json
          or ChromaDB collection (if PREFER_CHROMA=true)

Output: vector store maps chunk_id → (embedding, metadata)
```

**Store formats:**

| Store | Backend | BM25 Support | Notes |
|-------|---------|-------------|-------|
| `JsonVectorStore` | `data/embedded/store.json` | Yes (in-memory, built on query) | Default |
| `ChromaVectorStore` | ChromaDB collection | No (stubs return `[]`) | Requires `PREFER_CHROMA=true` + chromadb installed |

---

## 4. Stage 3 — RAG Query Pipeline

The complete pipeline in `rag/qa.py` runs up to eleven logical steps, several of which are optional feature flags:

```
User Question
  │
  ▼
┌─────────────────────────────────────────────────────────────────────┐
│  [OPTIONAL]  Document Pre-Filter                                    │
│  use_document_intelligence required                                 │
│  • Embeds question                                                  │
│  • Cosine similarity vs. every document's summary_embedding        │
│  • Keeps only documents above doc_filter_threshold (default 0.60)  │
│  • If no documents pass → returns "No relevant documents found"    │
└──────────────────────────────┬──────────────────────────────────────┘
                               │  doc_filter: list[doc_id] | None
                               ▼
                    ┌──────────────────┐
                    │ §4.1 Query       │
                    │ Enhancement      │
                    └────────┬─────────┘
                             │ effective_query
                             ▼
                    ┌──────────────────┐
                    │ §4.2 Retrieval   │
                    └────────┬─────────┘
                             │ raw_chunks (fetch_k = top_k × 2)
                             ▼
                    ┌──────────────────┐
                    │ §4.3 Reranking   │
                    │  (two passes)    │
                    └────────┬─────────┘
                             │ retrieved (top_k chunks)
                             ▼
                    ┌──────────────────┐
                    │ §4.4 Routing &   │
                    │    Synthesis     │
                    └────────┬─────────┘
                             │ answer (str)
                             ▼
                    ┌──────────────────┐
                    │ §4.5 Compression │
                    │  (optional)      │
                    └────────┬─────────┘
                             │ compressed_context | None
                             ▼
                    ┌──────────────────┐
                    │ §4.6 Faithfulness│
                    │  (optional)      │
                    └────────┬─────────┘
                             │
                             ▼
               MultiAgentQAResponse
               (answer, sources, router, specialists)
```

---

### 4.1 Query Enhancement

Controlled by `USE_QUERY_ENHANCEMENT=true` (default enabled).

```
Question
  │
  ▼
classify_query()  →  "simple" | "complex"
  │                  (single LLM call)
  │
  ├── "simple"
  │     └── hyde_enhance()
  │           • LLM generates a 2-4 sentence hypothetical answer
  │           • That text is used as the retrieval query
  │           • Improves semantic matching for factual questions
  │
  └── "complex"
        └── decompose_query()
              • LLM breaks question into 2-4 independent sub-queries
              • Each sub-query is retrieved separately
              • Results deduplicated by chunk_id
```

**When disabled:** original question text is used directly for retrieval.

---

### 4.2 Retrieval Strategies

Three retrieval paths, selected by feature flags:

```
                         ┌──────────────────────────────────┐
                         │      USE_HYBRID_RETRIEVAL?       │
                         └──────────┬───────────────────────┘
                          YES       │       NO
              ┌───────────────────┐ │ ┌────────────────────────────┐
              │  Hybrid Retrieve  │ │ │   USE_QUERY_ENHANCEMENT?   │
              │                   │ │ └────────────┬───────────────┘
              │  Dense Leg:       │ │    YES        │       NO
              │  embed + cosine   │ │    │          │
              │  similarity       │ │    │          ▼
              │       +           │ │    │   Dense-Only Retrieve
              │  Sparse Leg:      │ │    │   (original question)
              │  BM25 in-memory   │ │    │
              │  (k1=1.5, b=0.75) │ │    ▼
              │       ↓           │ │  complex → sub-queries
              │  RRF Fusion       │ │  → retrieve each → deduplicate
              │  score = 1/(60+d) │ │
              │        +1/(60+s)  │ │  simple → HyDE → dense retrieve
              │       ↓           │ │
              │  Region Boost     │ └────────────────────────────────
              │  (1.3× tables/   │
              │   figures if      │
              │   data-seeking    │
              │   query)          │
              │       ↓           │
              │  Parent Expansion │
              │  (if ≥2 sibling   │
              │   chunks in top-K │
              │   → add full      │
              │   parent section) │
              └───────────────────┘
```

**BM25 Parameters:**

| Parameter | Value | Role |
|-----------|-------|------|
| k1 | 1.5 | Term-frequency saturation |
| b | 0.75 | Document-length normalization |
| rrf_k | 60 | RRF rank dampening constant |
| Region boost | 1.3× | Applied to table/figure chunks when query has numeric/list intent |
| Parent threshold | 2 | Min sibling count to trigger section expansion |

**Data-seeking intent triggers (for region boost):**
- Query contains digits
- Query contains terms: `how many`, `total`, `average`, `percent`, `compare`, `list`, `top`, `rank`, `highest`, `lowest`, `sum`, `breakdown`, `distribution`, etc.

---

### 4.3 Reranking (Two-Pass)

```
raw_chunks  (fetch_k candidates, typically top_k × 2)
  │
  ▼
┌─────────────────────────────────────────────┐
│  Pass 1 · Token Overlap Boost  (always on)  │
│  For each chunk:                            │
│    overlap = |question_tokens ∩ chunk_tokens|│
│    new_score = old_score + (overlap × 0.01) │
│  Sort descending → keep top_k               │
└──────────────────────────┬──────────────────┘
                           │ top_k chunks
                           ▼
              ┌────────────────────────────┐
              │  USE_LLM_RERANKER?         │
              └────────────┬───────────────┘
                    YES    │       NO
                           │
                           ▼
┌─────────────────────────────────────────────┐
│  Pass 2 · LLM Batch Reranker               │
│                                             │
│  • Formats all candidates as:              │
│    [ID: chunk_id]                          │
│    Type: PROSE|TABLE_DESC|FIGURE_DESC|     │
│           SECTION_HEADER                   │
│    Section: parent_title > parent_subtitle │
│    Content: <best content, ≤600 chars>     │
│             (visual summary preferred for  │
│              table/figure chunks)          │
│                                             │
│  • Single LLM call → JSON response:        │
│    {ranking: [...], scores: {...},         │
│     dropped: [...], top_region_type: "..."}│
│                                             │
│  • Filters: score < 0.3 → dropped         │
│  • Fallback: original order if LLM fails  │
│  • Fallback: original list if all dropped  │
└─────────────────────────────────────────────┘
```

**Chunk type classification for reranker:**

| Label | Condition |
|-------|-----------|
| `TABLE_DESC` | `"table"` in `region_types` |
| `FIGURE_DESC` | `"figure"` in `region_types` |
| `SECTION_HEADER` | text length < 80 characters |
| `PROSE` | default |

---

### 4.4 Multi-Agent Routing & Synthesis

```
retrieved  (final top_k chunks after reranking)
  │
  ▼
Load visual_summaries.json for each source document
  │
  ▼
┌──────────────────────────────────────────────────────────────┐
│  _route_question()                                           │
│                                                              │
│  Collect region_ids from chunk metadata that appear in      │
│  visual_summaries (only meaningful summaries)               │
│                                                              │
│  use_table_agent = table_regions exist                       │
│                    AND question contains:                    │
│                    "table" | "row" | "column"               │
│                                                              │
│  use_figure_agent = figure_regions exist                     │
│                     AND question contains:                   │
│                     "figure" | "chart" | "image" |          │
│                     "diagram" | "shown"                      │
└──────────────────────────────────────────────────────────────┘
  │                                    │
  │ table agent triggered              │ figure agent triggered
  ▼                                    ▼
┌──────────────────────┐  ┌──────────────────────────────────┐
│  Table Specialist    │  │  Figure Specialist               │
│  Agent               │  │  Agent                           │
│                      │  │                                  │
│  System: "You are a  │  │  System: "You are a grounded     │
│  grounded table      │  │  figure specialist. Answer       │
│  specialist. Answer  │  │  only from frozen preprocessing  │
│  only from frozen    │  │  summaries."                     │
│  preprocessing       │  │                                  │
│  summaries."         │  │  Evidence: visual summary text   │
│                      │  │  per figure region               │
│  Evidence: visual    │  │                                  │
│  summary text per    │  │  Output: SpecialistResult        │
│  table region        │  │  (agent_name, output, region_ids)│
│                      │  └──────────────────────────────────┘
│  Output: SpecialistResult│
└──────────────────────┘
  │                                    │
  └──────────────┬─────────────────────┘
                 │ specialists: list[SpecialistResult]
                 ▼
┌──────────────────────────────────────────────────────────────┐
│  _synthesize_answer()                                        │
│                                                              │
│  System: "You are a synthesis agent for document-grounded   │
│  QA. Answer only from retrieved chunk evidence and          │
│  specialist outputs. Cite chunk ids and page numbers."      │
│                                                              │
│  Evidence sources:                                           │
│  • Retrieved chunks (or compressed context if §4.5)        │
│  • Table specialist output (if routed)                      │
│  • Figure specialist output (if routed)                     │
│                                                              │
│  Model: openai_model (default gpt-4.1-mini)                │
└──────────────────────────────────────────────────────────────┘
```

---

### 4.5 Context Compression

Controlled by `USE_CONTEXT_COMPRESSION=true`. Runs after reranking, before synthesis.

```
retrieved  (reranked top_k chunks)
  │
  ▼
Format each chunk as:
  [CHUNK <id> | <type> | <title> > <subtitle> | score: <x.xxxx>]
  <content>   (visual summary text preferred for table/figure)
  │
  ▼
LLM call (openai_model) with CONTEXT_COMPRESSION_PROMPT
  │
  Instructions to LLM:
  ├── Remove OCR noise (garbled chars, broken hyphenation)
  ├── Remove repeated headers/footers
  ├── Remove navigation text ("see section 3.2")
  ├── Remove boilerplate (copyright, revision history)
  ├── For PROSE: extract only query-relevant sentences (exact wording)
  ├── For TABLE_DESC / FIGURE_DESC: keep full description
  └── Drop passages: score < compression_threshold (default 0.5)
      AND contains no unique information
  │
  Target: ≥40% token reduction while preserving all answer-bearing content
  │
  ▼
compressed_context  (string with chunk tags preserved for citation)
  │
  Fallback: uncompressed passage text if LLM call fails
```

---

### 4.6 Faithfulness Check & Correction

Controlled by `USE_FAITHFULNESS_CHECK=true`. Runs after synthesis.

```
generated_answer  +  retrieved  (source chunks)
  │
  ▼
┌──────────────────────────────────────────────────────────────┐
│  FaithfulnessChecker.check()                                 │
│                                                              │
│  LLM breaks answer into individual claims and               │
│  checks each against source chunks:                         │
│                                                              │
│  SUPPORTED  → directly stated or numerically entailed       │
│  INFERRED   → reasonable inference, not explicitly stated   │
│  UNSUPPORTED→ no basis in sources  (hallucination)          │
│                                                              │
│  Returns FaithfulnessResult:                                │
│  • claims: list[ClaimVerdict]                               │
│  • overall_verdict: FAITHFUL | PARTIALLY_FAITHFUL |         │
│                     UNFAITHFUL                              │
│  • confidence_score: 0.0–1.0                               │
│  • recommended_action: return_as_is |                       │
│                         flag_for_review |                   │
│                         regenerate_without_unsupported_claims│
└──────────────────────────────┬───────────────────────────────┘
                               │
              ┌────────────────┴─────────────────┐
              │  recommended_action == return_as_is?│
              └────────────────┬──────────────────┘
                        NO     │      YES
                               │       └─→  return answer unchanged
                               ▼
┌──────────────────────────────────────────────────────────────┐
│  FaithfulnessChecker.correct()                               │
│                                                              │
│  LLM rewrites answer with rules:                            │
│  1. Remove UNSUPPORTED claims  (do not replace with guess)  │
│  2. Add hedging to INFERRED:                                │
│     "the document suggests…", "based on the table…"        │
│  3. Keep SUPPORTED claims exactly as written                │
│  4. If removal leaves answer unable to address query:       │
│     "The retrieved documents do not contain sufficient      │
│      information to answer this."                           │
│                                                              │
│  Fallback: original answer if LLM call fails               │
└──────────────────────────────────────────────────────────────┘
```

---

## 5. API Layer

The FastAPI server (`api/`) wraps the same core pipeline used by the CLI.

```
HTTP Client
  │
  ▼
api/app.py  create_app()
  │  On startup:
  │  • ensure data directories exist
  │  • if S3_BUCKET_NAME: sync artifacts + vector store from S3
  │
  ├── GET  /                   → static frontend (api/static/index.html)
  ├── GET  /health             → {"status": "ok"}
  │
  ├── POST /documents/preprocess
  │     body: multipart PDF upload
  │     → preprocess_document()  → frozen artifacts
  │     → if S3: sync processed dir to S3
  │     ← {document_id, chunk_count, page_count, warnings}
  │
  ├── POST /documents/index
  │     → deletes existing store.json
  │     → index_all_processed_documents()
  │     → if S3: sync vector store to S3
  │     ← {indexed_documents, total_chunks, documents: {doc_id: count}}
  │
  └── POST /query
        body: {question: str, top_k: int (default 4)}
        → answer_question_from_frozen_artifacts()
        ← {answer, sources: [...], router: {...}}
```

**`/query` response structure:**

```json
{
  "answer": "...",
  "sources": [
    {
      "chunk_id": "...",
      "page_number": 3,
      "document_id": "...",
      "source_filename": "report.pdf",
      "region_ids": ["region_42"],
      "crop_asset_ids": [],
      "score": 0.8712
    }
  ],
  "router": {
    "use_table_agent": true,
    "use_figure_agent": false,
    "table_regions": ["region_42"],
    "figure_regions": []
  }
}
```

---

## 6. Feature Flags Reference

All flags are read from environment variables or `.env` file via pydantic-settings.

| Environment Variable | Default | Stage | Effect |
|---------------------|---------|-------|--------|
| `USE_DOCUMENT_INTELLIGENCE` | `false` | Preprocess | Title propagation, section grouping, document descriptor + summary embedding |
| `USE_ADAPTIVE_CHUNKING` | `false` | Preprocess | LLM selects chunk strategy per document (requires document intelligence) |
| `USE_VLM_SUMMARIES` | `false` | Preprocess | GPT-4o vision call per table/figure crop; replaces OCR fallback |
| `USE_QUERY_ENHANCEMENT` | `true` | Query | HyDE (simple) or sub-query decomposition (complex) before retrieval |
| `USE_HYBRID_RETRIEVAL` | `false` | Query | Dense + BM25 via RRF, region boost, parent-context expansion |
| `USE_LLM_RERANKER` | `false` | Query | LLM batch-scores all candidates; drops score < 0.3 |
| `USE_CONTEXT_COMPRESSION` | `false` | Query | LLM strips OCR noise + irrelevant sentences before synthesis |
| `USE_FAITHFULNESS_CHECK` | `false` | Query | Claim-level verification + rewrite after synthesis |
| `PREFER_CHROMA` | `false` | Index | Use ChromaDB instead of JSON vector store |

**LLM model assignments:**

| Task | Model Variable | Default |
|------|---------------|---------|
| Synthesis, specialists, routing | `OPENAI_MODEL` | `gpt-4.1-mini` |
| Section + document summarization | `DESCRIPTOR_MODEL` | `gpt-4o-mini` |
| Table/figure vision descriptions | `VLM_MODEL` | `gpt-4o` |
| Embedding | `EMBEDDING_MODEL` | `text-embedding-3-small` |

**Tunable thresholds:**

| Variable | Default | Meaning |
|----------|---------|---------|
| `DOC_FILTER_THRESHOLD` | `0.60` | Min cosine similarity for document pre-filter |
| `COMPRESSION_THRESHOLD` | `0.50` | Passages below this score may be dropped during compression |
| `DEFAULT_TOP_K` | `4` | Chunks returned to synthesis |
| `PREPROCESS_CHUNK_SIZE` | `1800` | Target chunk size in characters |
| `PREPROCESS_CHUNK_OVERLAP` | `200` | Overlap between consecutive chunks |
| `PDF_RENDER_SCALE` | `3.0` | Page image render scale (higher = better OCR quality) |

---

## 7. Key Data Structures

### Core chunk lifecycle

```
ProcessedChunk          (output of document_Process/)
  chunk_id              "doc_{id}_chunk_{n}"
  text                  raw chunk text
  page_number           int
  source_region_ids     list[str]  (layout region IDs)
  region_types          list[str]  ("text_block"|"table"|"figure")
  parent_title          str  (section title, if intelligence enabled)
  parent_subtitle       str  (subsection title, if intelligence enabled)
  metadata              dict  (document_id, source_filename, …)

      ↓   chunk_records_from_processed_chunks()

ChunkRecord             (ready for vector store insertion)
  chunk_id              str
  text                  str
  metadata              dict  (same as above)

      ↓   vector_store.upsert(records, embeddings)
          stored as: chunk_id → (embedding, text, metadata)

      ↓   vector_store.query() / bm25_query()

RetrievedChunk          (output of retrieval)
  chunk_id              str
  text                  str
  metadata              dict
  score                 float  (cosine similarity or BM25 or RRF)
```

### Response types

```
SpecialistResult
  agent_name            "table" | "figure"
  output                str  (specialist LLM answer)
  region_ids            list[str]

MultiAgentQAResponse
  question              str
  answer                str
  sources               list[dict]  (chunk_id, page, doc_id, score, …)
  router                dict  (use_table_agent, use_figure_agent, regions)
  specialists           list[SpecialistResult]

FaithfulnessResult
  claims                list[ClaimVerdict]
  overall_verdict       "FAITHFUL" | "PARTIALLY_FAITHFUL" | "UNFAITHFUL"
  confidence_score      float  (0.0–1.0)
  recommended_action    "return_as_is" | "flag_for_review" |
                        "regenerate_without_unsupported_claims"
```

---

## 8. LLM Call Budget Per Query

The table below shows the maximum number of LLM API calls for a single query, depending on enabled features:

| Step | Calls | Condition |
|------|-------|-----------|
| Query classification | 1 | `use_query_enhancement` |
| HyDE or sub-query decomposition | 1 | `use_query_enhancement` |
| LLM reranker | 1 | `use_llm_reranker` |
| Table specialist agent | 1 | routed + table regions exist |
| Figure specialist agent | 1 | routed + figure regions exist |
| Context compression | 1 | `use_context_compression` |
| Answer synthesis | 1 | always |
| Faithfulness check | 1 | `use_faithfulness_check` |
| Faithfulness correction | 1 | `use_faithfulness_check` AND issues found |
| **Maximum total** | **9** | all features enabled + both specialists triggered |
| **Minimum total** | **1** | all optional features disabled |

**Typical production configuration** (hybrid + reranker + faithfulness):

```
use_query_enhancement=true     → +2 calls
use_hybrid_retrieval=true      →  0 calls (no extra LLM)
use_llm_reranker=true          → +1 call
synthesis                      → +1 call
use_faithfulness_check=true    → +1–2 calls
────────────────────────────────────────
Total: 5–6 calls per query
```
