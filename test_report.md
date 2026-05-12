# Pipeline Test Report
Generated: 2026-05-13 (v5)
Documents: attentionisallyouneed.pdf (DOC_A, 15 pages), handbook.pdf (DOC_B, 9 pages)

## 1. Environment
| Check | Result |
|-------|--------|
| ruff lint | PASS |
| ruff format | PASS |
| SECTION_FILTER_MAX | 8 |
| USE_DOCUMENT_PREFILTER | True |
| DOCUMENT_FILTER_THRESHOLD | 0.45 |
| VLM_RETRY_MAX | 3 |
| USE_DOCUMENT_SUMMARY | True |
| USE_VLM_SUMMARIES | True |
| USE_FAITHFULNESS_CHECK | True |
| SYNTHESIS_MODEL | gpt-4.1-nano |
| VISION_SYNTHESIS_MODEL | gpt-4o-mini |
| VISION_MAX_IMAGE_PX | 800 |

## 2. Preprocessing
| Check | DOC_A (attentionisallyouneed.pdf) | DOC_B (handbook.pdf) |
|-------|-----------------------------------|----------------------|
| Total chunks | 89 | 41 |
| parent_title coverage (real title) | 50/89 | 30/41 |
| parent_title == None (orphan/heading) | 39/89 | 11/41 |
| parent_title == "untitled" | 0/89 | 0/41 |
| section_summary coverage | 50/89 | 30/41 |
| crop_references coverage | 8/89 | 14/41 |
| text field still populated | 0/89 (correct) | 0/41 (correct) |
| chunks spanning multiple sections | 0/89 (correct) | 0/41 (correct) |
| document_summary in manifest | yes | yes |

Unique parent_titles in DOC_A: Abstract, Attention Is All You Need, 1 Introduction, 2 Background,
3 ModelArchitecture, 3.1 EncoderandDecoderStacks, 3.2 Attention,
3.2.1 ScaledDot-ProductAttention, 3.2.2 Multi-HeadAttention,
3.2.3 ApplicationsofAttentioninourModel, 3.3 Position-wiseFeed-ForwardNetworks,
3.4 EmbeddingsandSoftmax, 3.5 PositionalEncoding, 4 WhySelf-Attention,
5 Training, 5.1 TrainingDataandBatching, 5.2 HardwareandSchedule,
5.3 Optimizer, 5.4 Regularization, 6.1 MachineTranslation,
6.2 ModelVariations, 6.3 EnglishConstituencyParsing, 7 Conclusion, References

Notes:
- `text` field always `""` (correct)
- 0 cross-section chunks (correct)
- `parent_title=="untitled"` is 0 — all content blocks have a real title or are heading/orphan blocks
- The 39 None-parent_title chunks are: ~21 section heading blocks (title regions don't receive parent_title) + ~18 orphan page-1 items (authors, affiliations, permission notice — no layout region match)

## 3. Vector Index
| Check | Value |
|-------|-------|
| Pool 0 record count | 2 |
| All processed docs in Pool 0 | PASS |
| Pool A "1 Introduction" present | YES |
| Pool A "1 Introduction" cosine vs "introduction" query | 0.219 (below threshold — now reached via name-match injection) |
| blocks.json total records | 913 |
| embedding dimensions | 1536 |

## 4. Pool 0 Retrieval
| Query | DOC_A score | DOC_B score | Passes threshold | Correct |
|-------|-------------|-------------|------------------|---------|
| Transformer/attention (DOC_A specific) | 0.6419 | 0.2271 | attentionisallyouneed.pdf | yes |
| AI handbook (DOC_B specific) | 0.3489 | 0.5265 | handbook.pdf | yes |
| generic ("what is the main topic") | 0.3044 | 0.2951 | none (fallback) | yes |

## 5. Query Results
| Query | Completed | Answer quality | Faithfulness | Sources | Correct doc | Notes |
|-------|-----------|----------------|--------------|---------|-------------|-------|
| doc_a_specific | yes | good — explains multi-head attention, different representations | FAITHFUL | 15 | yes (attentionisallyouneed.pdf) | |
| doc_b_specific | yes | good — ML vs DL, feature extraction distinction | FAITHFUL | 16 | yes (handbook.pdf) | No timeout (Fix v4-1) |
| multi_doc | yes | good — listed topics from both docs | FAITHFUL | 20 | both | Global fallback works |
| metric_decimal | yes | good — 27.3/28.4 EN-DE, 38.1/41.8 EN-FR BLEU | FAITHFUL | 19 | attentionisallyouneed.pdf | Decimal pattern triggered metric mode |
| visual | yes | good — Scaled Dot-Product + Multi-Head diagrams described | FAITHFUL | 19 | attentionisallyouneed.pdf | Vision synthesis without timeout (Fix v4-1) |
| table | yes | good — Table 3 model variations listed correctly | FAITHFUL | 15 | attentionisallyouneed.pdf | |
| section_specific | yes | **good** — RNN limitations, Transformer motivation from §1 | FAITHFUL | 14 | attentionisallyouneed.pdf | **Fixed** — name-match injection routes to "1 Introduction" |
| fast_mode | yes | good — 8 A100 GPUs, training steps | None (correct) | 8 | attentionisallyouneed.pdf | faithfulness gate correctly skipped |
| no_faithfulness | yes | good — Transformer state-of-the-art conclusion | None (correct) | 13 | attentionisallyouneed.pdf | Conclusion section cited correctly |

## 6. Issues

No open issues from this run. All 9 queries completed with FAITHFUL or correct-None faithfulness labels, correct document routing, and substantive answers.

## 7. Comparison across versions
| Metric | v3 (initial) | v4 (vision+title) | v5 (name-match) | change |
|--------|-------------|-------------------|-----------------|--------|
| parent_title "untitled" count | 39 | 0 | 0 | v4 Fix 2 eliminated all untitled |
| section_specific answer | "specific content not provided" | "blocks do not contain introduction info" | introduction content returned | **Fixed in v5** |
| visual query timeout | yes (60s fallback) | no | no | v4 Fix 1 |
| doc_a_specific quality | poor | good | good | improved |
| metric_decimal BLEU accuracy | 28.4/41.8 only | 28.4/41.8 only | all 4 scores (27.3/28.4/38.1/41.8) | more complete |
| ruff lint | PASS | PASS | PASS | maintained |
| All 9 queries FAITHFUL or correct-None | no (fast_mode always None) | yes | yes | maintained |

---
End of report
