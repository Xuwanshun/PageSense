# Qwen3-VL Self-Hosted Model Integration

**Date:** 2026-05-16
**Branch:** feature/qwen3-vl-integration
**Status:** Approved (revised to HuggingFace Inference Endpoints)

## Problem

The VLM enrichment step in `document_process/vlm.py` calls **gpt-4o** via the OpenAI API to generate natural-language descriptions of cropped tables and figures. This sends document content to OpenAI and incurs per-image cost.

A fine-tuned Qwen3-VL LoRA adapter exists at `s3://qwen3-vl-sft-training-604561274097/final-model/`. Deploying it to HuggingFace Inference Endpoints gives an OpenAI-compatible API with zero infrastructure to manage, auto-scales to zero when idle.

## Approach

Deploy the fine-tuned adapter to **HuggingFace Inference Endpoints**. Point `vlm.py` at the HF endpoint URL. Fall back to gpt-4o automatically if the endpoint is unavailable or returns an error.

No new AWS infrastructure. No EC2 lifecycle management. No CDK stack changes.

---

## Architecture

```
ECS Fargate Container (document processing)
    │
    └─ vlm.enrich_summaries_with_vlm()
            ├─ try:  POST to HF Inference Endpoint (Qwen3-VL adapter)
            └─ except ConnectionError/timeout: fallback → gpt-4o (OpenAI)


HuggingFace Inference Endpoints (external, managed):
    - Serves Qwen3-VL + LoRA adapter
    - OpenAI-compatible API at https://<endpoint>.huggingface.cloud/v1
    - Scales to zero when idle (~$0 between jobs)
    - ~$0.60/hr only while processing
```

---

## Components

### Changed: `config.py`

Two new settings:

```python
vlm_base_url: str | None = None            # HF endpoint URL, e.g. https://<id>.huggingface.cloud/v1
vlm_self_hosted_model: str = "tgi"         # model name HF endpoint exposes (default "tgi")
# existing vlm_model (gpt-4o) becomes the fallback when vlm_base_url is unset or unreachable
```

When `vlm_base_url` is `None`, behaviour is identical to today.

### Changed: `document_process/vlm.py`

Two changes:

1. `_describe_crop()` — when `settings.vlm_base_url` is set, create a separate `OpenAI` client pointing at the HF endpoint URL and use `settings.vlm_self_hosted_model` as the model name. The existing gpt-4o client is unchanged.

2. `enrich_summaries_with_vlm()` — wrap each `_describe_crop()` call:
   - Try self-hosted first (if `settings.vlm_base_url` is set)
   - On `httpx.ConnectError`, `TimeoutError`, or any HTTP error: log warning, retry with gpt-4o
   - gpt-4o failures propagate as before (non-fatal, keeps OCR-text fallback)

### Changed: `cdk/stacks/app_stack.py`

Add `VLM_BASE_URL` and `VLM_SELF_HOSTED_MODEL` to ECS task environment (plain env vars, not secrets — the HF endpoint token is the sensitive part).

Add a new Secrets Manager secret reference for `VLM_HF_TOKEN` (HuggingFace API token needed to call the endpoint) and inject it into the ECS task secrets map.

---

## Data Flow

### Happy path — HF endpoint available
```
enrich_summaries_with_vlm()
  → for each crop image:
      → POST https://<endpoint>.huggingface.cloud/v1/chat/completions
      → returns description
```

### Fallback — HF endpoint unreachable or returns error
```
  → POST HF endpoint → ConnectionError / non-200
  → log warning: "Self-hosted VLM unavailable, falling back to gpt-4o"
  → POST OpenAI gpt-4o → description returned
```

### Not configured (vlm_base_url unset)
```
  → self-hosted path never attempted
  → goes straight to gpt-4o (existing behaviour, no change)
```

---

## Cost

| Scenario | Cost |
|---|---|
| HF endpoint active during job | ~$0.10-0.15/job (10-15 min × $0.60/hr) |
| HF endpoint scaled to zero (idle) | $0 |
| Fallback to gpt-4o | ~$0.01/image (existing) |

---

## Testing

### Unit tests — `tests/unit/test_vlm.py` (extend existing)

Mock HTTP:
- `vlm_base_url` set, HF endpoint succeeds → gpt-4o never called
- `vlm_base_url` set, HF endpoint raises `ConnectionError` → gpt-4o called as fallback
- `vlm_base_url` set, HF endpoint times out → gpt-4o called as fallback
- `vlm_base_url=None` → self-hosted never attempted, goes straight to gpt-4o

Existing tests unaffected — `vlm_base_url` defaults to `None`.

---

## Deployment Steps (one-time manual setup)

1. Upload adapter to HuggingFace Hub:
   ```bash
   # from local machine
   aws s3 sync s3://qwen3-vl-sft-training-604561274097/final-model/ ./final-model/
   huggingface-cli upload <your-hf-username>/qwen3-vl-finetuned ./final-model/
   ```
2. Create an Inference Endpoint on huggingface.co pointing at the uploaded model
3. Copy the endpoint URL and your HF token into AWS Secrets Manager:
   ```bash
   aws secretsmanager create-secret --name rag-agent/vlm-hf-token --secret-string "<token>"
   ```
4. Set `VLM_BASE_URL` and `VLM_SELF_HOSTED_MODEL` in the CDK app context or directly in `app_stack.py`
5. `cdk deploy RagAgentApp` — injects new env vars into ECS task definition
