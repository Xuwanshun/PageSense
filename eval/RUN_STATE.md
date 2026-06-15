# Eval Run — Resume State (checkpoint 2026-06-14 23:05)

Live evaluation of the RAG pipeline over 3 manuals, using the **fine-tuned Qwen3-VL**
(deployed on Modal) for VLM summaries. This file is the resume runbook — a future
session can continue from here.

## Environment (already set up)
- Native venv: `.venv` (python 3.11, full runtime incl. Paddle). Use `.venv/bin/python`.
- `.env`: `OPENAI_API_KEY` set ✓; `USE_VLM_SUMMARIES=true`;
  `VLM_BASE_URL=https://nick-xuwanshun--qwen3-vl-rag-model-fastapi-app.modal.run/v1`;
  `VLM_SELF_HOSTED_MODEL=qwen3-vl-rag`.
- Modal app `qwen3-vl-rag` deployed (A10G), endpoint warm. Auth in `~/.modal.toml`.
- Gold set: `eval/datasets/manual_questions.yaml` (27 Q: 18 factual / 4 procedure / 2 suggestion / 3 no-answer; physical-page citations).
- Raw PDFs staged per-doc: `data/raw_ur5/`, `data/raw_tesla/`, `data/raw_gpt/`.
- Preprocessing was run with `USE_VLM_SUMMARIES=false` for fast OCR (GPT-4o VLM was hanging),
  then Qwen applied separately via `scripts/qwen_enrich.py` (decoupled from OCR).

## DONE
- ✅ gpt.pdf preprocessed (155 chunks) + Qwen-enriched (15/15 visual summaries).
- ✅ UR5_handbook.pdf preprocessed (294 chunks) + Qwen-enriched (272/311, 39 crops failed — OCR fallback kept).
- ✅ teslaOwnManual.pdf preprocessed (572 chunks, dir `0681ec6dc8ff…`). Qwen enrichment IN PROGRESS (357 regions, logs/08_enrich_tesla.log).
- ✅ Qwen endpoint verified (HTTP 200, "VLM backend used: qwen3-vl-rag").

## REMAINING STEPS (in order)
1. **Wait for Tesla enrichment**: `grep DONE logs/08_enrich_tesla.log` (357 regions, ~10-15 min).
4. **Index all 3 docs** (enrichment doesn't change chunks, so this just needs all chunks.json present):
   `.venv/bin/python main.py --index > logs/09_index.log 2>&1`
   (store.json currently has only gpt's 155 rows — re-index covers all 3.)
5. **Run live eval** against the gold set (uses Qwen summaries via visual_summaries.json):
   ```
   .venv/bin/python -m eval.runners.retrieval_eval --gold eval/datasets/manual_questions.yaml \
       --live --confirm-cloud-cost --out-dir eval/reports/run1 > logs/10_eval_retrieval.log 2>&1
   .venv/bin/python -m eval.runners.answer_eval --gold eval/datasets/manual_questions.yaml \
       --live --confirm-cloud-cost --out-dir eval/reports/run1 > logs/11_eval_answer.log 2>&1
   .venv/bin/python -m eval.runners.latency_eval --gold eval/datasets/manual_questions.yaml \
       --live --confirm-cloud-cost --out-dir eval/reports/run1 > logs/12_eval_latency.log 2>&1
   ```
   (answer_eval reuses run1/predictions.json if present; otherwise regenerates.)
6. **Compile final report** from `eval/reports/run1/{retrieval_metrics,citation_metrics,answer_metrics,llm_judge,ops_metrics}.json`
   into `eval/reports/run1/REPORT.md`. Deliver that + the `logs/` directory.
7. **COST: tear down the GPU when done** → `.venv/bin/modal app stop qwen3-vl-rag`
   (A10G bills while deployed — do not leave it running).

## Logs map
- `logs/02_native_install.log` — venv build
- `logs/05_preprocess_{ur5,tesla}_novlm.log` — OCR (VLM off)
- `logs/06_modal_deploy.log` — Qwen deploy
- `logs/08_enrich_{gpt,ur5,tesla}.log` — Qwen VLM enrichment
- `logs/09_index.log`, `logs/10-12_eval_*.log` — index + eval

[done] eval complete 2026-06-15 10:04
