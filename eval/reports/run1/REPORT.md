# RAG Evaluation Report — run1

**Generated:** 2026-06-15 10:03
**Corpus:** 3 operation manuals — UR5_handbook.pdf (294 chunks), teslaOwnManual.pdf (572), gpt.pdf (155) = 1021 chunks indexed.
**Gold set:** 27 questions (18 factual / 4 procedure / 2 suggestion / 3 no-answer), physical-page citations.
**Pipeline:** OpenAI `gpt-4.1-mini` synthesis + `text-embedding-3-small`; query-enhancement ON; hybrid/rerank/compression/faithfulness OFF (defaults).
**VLM summaries:** fine-tuned **Qwen3-VL-4B** (`azhuang3/qwen3_vlm_task`) self-hosted on Modal A10G. 644 table/figure regions enriched (gpt 15/15, UR5 272/311, Tesla 357/357).
**Judge:** `gpt-4o` (stronger than the generator, to reduce self-bias).

---

## 1. Retrieval quality
| Metric | Value |
|---|---|
| MRR | 0.77 |
| Recall@1 | 54.2% |
| Recall@3 | 83.3% |
| Recall@5 | 95.8% |
| nDCG@5 (graded) | 0.79 |
| Scored questions | 24 (no-answer Qs excluded) |

Retrieval is strong: the right page is in the top-5 **95.8%** of the time and ranked first 54.2% of the time (MRR 0.77).

## 2. Citation quality (top-2 cited sources, ±1 page)
| Metric | Page-tolerant | Page-exact |
|---|---|---|
| Precision | 62.5% | 45.8% |
| Recall | 66.7% | 64.6% |

## 3. Answer quality
**Deterministic (offline):**
- Must-include pass rate: **88.9%**
- Must-not-include violations: 0.0%
- No-answer handling accuracy: **100.0%** (all 3 unanswerable Qs correctly refused — no hallucinated prices/tickers)
- False-refusal rate (on answerable Qs): 4.2%

**LLM-judge (gpt-4o, 1–5):**
- Answer correctness: **4.89**
- Faithfulness / groundedness: **4.89**
- Completeness: 4.85
- Hallucination rate: **7.4%** (2 of 27)

## 4. Suggestion quality (engineering recommendations)
- Evidence-supported (offline, cites a manual page): 50.0%
- Judge — actionable: 5.00/5 · grounded in cited evidence: 100.0%

## 5. Operational
- Latency: mean **10.7s**, p50 9.4s, p95 17.3s, max 32.5s
- Est. cost (approx, generation only): $0.26 over 27 queries (~$0.009/query)

---

## Headline
- **Retrieval Recall@5 95.8%, MRR 0.77** — retrieval is the strong point.
- **Faithfulness 4.89/5, hallucination 7.4%, no-answer accuracy 100.0%** — the system reliably refuses when the corpus lacks the answer.
- **Weakest link: citation precision (62.5% tolerant)** — answers are correct but cite some off-target pages; a reranker/citation-filter is the obvious next experiment (use `ablation_eval`).

## Caveats
- 27-question gold set — directional, not statistically tight. Expand to 50–100 for stable deltas.
- Two eval-metric bugs were fixed during this run: nDCG normalization (was >1.0 from duplicate page hits) and the refusal-phrase detector (missed "do not provide/not stated").
- Cost is a rough char/4 token estimate; the OpenAI client doesn't surface usage.
- Defaults only (no hybrid/rerank/compression). Run `ablation_eval` to measure each feature's contribution.
