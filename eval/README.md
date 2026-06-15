# Evaluation

A small, offline-first evaluation framework for this manual / operation-handbook
RAG system. Use it to tell whether a change (e.g. swapping OCR, quantizing the
VLM, toggling a retrieval feature) made things better or worse.

## Design rule: offline by default

The **default** code path is fully offline. It scores a recorded
`predictions.json` against a gold set using pure-Python metrics — **no Paddle,
no OpenAI, no AWS/S3/ECS, no spend.** Anything that calls a model is gated:

| Flag | Effect |
|------|--------|
| _(none)_ | Score an existing predictions file. Offline, free. |
| `--live` | Regenerate predictions through the QA pipeline / run the LLM judge. **Calls OpenAI.** |
| `--confirm-cloud-cost` | Required alongside `--live` for runs that fan out (every runner that generates). Without it the runner prints a rough estimate and exits. |

## What it measures

1. **Retrieval** (`metrics/retrieval.py`) — Recall@k, MRR, nDCG@k (graded).
2. **Citation** (`metrics/citation.py`) — precision/recall of the cited sources,
   page-exact and page-tolerant; section-aware when predictions carry sections.
3. **Answer** — deterministic offline checks (`metrics/answer.py`: must-include,
   must-not-include, no-answer/refusal handling) plus LLM-as-judge
   (`metrics/judge.py`, live: correctness, faithfulness, completeness,
   hallucination).
4. **Suggestion quality** — `requires_citation` evidence-support check (offline)
   and the judge's `suggestion_actionable` / `suggestion_supported_by_evidence`.
5. **Ops** (`metrics/ops.py`) — latency p50/p95/mean and a *rough* cost estimate
   (the OpenAI client doesn't expose token usage, so cost is approximate).

## Runners

```bash
# Offline — score a recorded predictions file (free):
python -m eval.runners.retrieval_eval --gold eval/datasets/smoke.yaml \
    --predictions eval/datasets/smoke_predictions.json
python -m eval.runners.answer_eval   --gold eval/datasets/smoke.yaml \
    --predictions eval/datasets/smoke_predictions.json
python -m eval.runners.latency_eval  --predictions eval/datasets/smoke_predictions.json

# Live — regenerate + judge (costs money, needs a built index + OPENAI_API_KEY):
python -m eval.runners.retrieval_eval --gold eval/datasets/manual_questions.yaml \
    --live --confirm-cloud-cost
python -m eval.runners.answer_eval    --gold eval/datasets/manual_questions.yaml \
    --live --confirm-cloud-cost

# Ablation — measure each feature flag's contribution (dry-run lists variants):
python -m eval.runners.ablation_eval  --gold eval/datasets/manual_questions.yaml
python -m eval.runners.ablation_eval  --gold eval/datasets/manual_questions.yaml \
    --live --confirm-cloud-cost
```

Reports land in `eval/reports/<timestamp>-<runner>/` (gitignored).

## Adding real gold questions

1. Copy the schema: `cp eval/datasets/manual_questions.example.yaml eval/datasets/manual_questions.yaml`.
2. For each question set `type`, the `question`, and `expected`:
   - **`sources`** — the file name + page numbers where the answer lives, with a
     graded `relevance` (3 = primary evidence, 1 = supporting). These drive
     Recall@k / nDCG / citation P/R, so get the pages right.
   - **`must_include` / `must_not_include`** — short, unambiguous substrings (a
     torque value, a part number). These are the cheap offline correctness gate.
   - **`reference_answer`** — used only by the live judge.
   - For `type: suggestion`, set `requires_citation: true` so an unsupported
     recommendation is scored as unsupported.
   - For `type: no_answer`, set `no_answer: true` — questions whose answer is not
     in the corpus. These test refusal/hallucination behavior; include several.
3. Aim for a spread: factual lookups, multi-page procedures, suggestions, and
   no-answer cases. ~30–50 questions gives reasonably stable numbers (the smoke
   set's 3 do not — they only prove the metrics run).
4. Build the index first (`python main.py --index`) so `--live` can retrieve.

**Privacy:** your real `manual_questions.yaml`, any `predictions.json`, and all
of `eval/reports/` are gitignored. Only the `.example` schema and the `smoke`
fixtures are committed. Do not commit private manuals.

## Live recipe (offline scoring of a paid run)

To keep judge/answer iteration free after one paid generation, save the live
predictions and re-score offline:

```bash
python -m eval.runners.retrieval_eval --gold eval/datasets/manual_questions.yaml \
    --live --confirm-cloud-cost --out-dir eval/reports/run1
# then iterate on metrics offline against eval/reports/run1/predictions.json
python -m eval.runners.retrieval_eval --gold eval/datasets/manual_questions.yaml \
    --predictions eval/reports/run1/predictions.json
```
