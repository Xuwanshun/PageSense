# Qwen3-VL Self-Hosted Model Integration

**Date:** 2026-05-16
**Branch:** feature/qwen3-vl-integration
**Status:** Approved

## Problem

The VLM enrichment step in `document_process/vlm.py` calls **gpt-4o** via the OpenAI API to generate natural-language descriptions of cropped tables and figures. This costs ~$0.01 per image and sends document content to a third-party API.

A fine-tuned Qwen3-VL LoRA adapter already exists at `s3://qwen3-vl-sft-training-604561274097/final-model/`. Using it eliminates per-image OpenAI costs for VLM and keeps document data within AWS.

## Scope

Replace the gpt-4o VLM call with the self-hosted Qwen3-VL adapter for document processing in the production ECS environment. gpt-4o remains as an automatic fallback if the model server is unavailable.

Out of scope: embedding model, QA model (gpt-4.1-mini), local development serving.

---

## Architecture

```
ECS Fargate Container (document processing)
    │
    ├─ 1. model_server.ensure_running(settings)
    │       └─ starts EC2 spot instance if stopped (~2 min wait)
    │
    ├─ 2. OCR → layout detection → chunking  (unchanged)
    │
    ├─ 3. vlm.enrich_summaries_with_vlm()
    │       ├─ try: POST to vLLM server (Qwen3-VL adapter)
    │       └─ except ConnectionError/timeout: fallback → gpt-4o
    │
    └─ 4. model_server.stop(settings)
            └─ stops EC2 instance (EBS preserved, ~$0 when stopped)


AWS Infrastructure (VPC, Public Subnet):
    EC2 g4dn.xlarge spot instance
        - Elastic IP (stable address, ~$0 when attached)
        - vLLM on port 8080
        - Security Group: port 8080 from ECS SG only
        - IAM instance profile: S3 read on qwen3-vl-sft-training-*
        - EBS volume persists between stop/start cycles
```

---

## Components

### New: `cdk/stacks/model_server_stack.py`

CDK stack provisioning:
- EC2 `g4dn.xlarge` spot instance (stopped by default)
- Elastic IP attached to the instance
- Security Group: inbound port 8080 from ECS task Security Group only
- IAM instance profile: `s3:GetObject` + `s3:ListBucket` on `qwen3-vl-sft-training-604561274097`
- User data: runs `scripts/model-server-userdata.sh` on first boot

Outputs: `ModelServerInstanceId`, `ModelServerElasticIp`

### New: `worker/model_server.py`

Lifecycle manager with two public functions:

```python
def ensure_running(settings: Settings) -> bool:
    """Start instance if stopped; poll until vLLM /health returns 200.
    Returns True if healthy, False if timeout (3 min). Never raises."""

def stop(settings: Settings) -> None:
    """Stop the instance. No-op if already stopped or terminated. Never raises."""
```

Uses `boto3` EC2 client. Polls every 10 seconds up to 3 minutes.

### New: `scripts/model-server-userdata.sh`

EC2 first-boot script:
1. Install CUDA drivers + Python + vLLM
2. `aws s3 sync s3://qwen3-vl-sft-training-604561274097/final-model/ /opt/model/adapter/`
3. Start vLLM:
   ```bash
   python -m vllm.entrypoints.openai.api_server \
     --model Qwen/Qwen2.5-VL-7B-Instruct \
     --enable-lora \
     --lora-modules qwen3-vl-finetuned=/opt/model/adapter/ \
     --port 8080 \
     --max-model-len 4096
   ```
4. Write a systemd service so vLLM restarts automatically on subsequent instance starts.

### Changed: `config.py`

Three new settings (all optional, default to None/existing behaviour):

```python
vlm_instance_id: str | None = None         # EC2 instance ID — set by CDK deploy output
vlm_base_url: str | None = None            # http://<elastic-ip>:8080/v1
vlm_self_hosted_model: str = "qwen3-vl-finetuned"  # name vLLM serves the adapter under
# existing vlm_model (gpt-4o) becomes the fallback
```

When `vlm_instance_id` and `vlm_base_url` are both unset, behaviour is identical to today.

### Changed: `document_process/vlm.py`

Two changes:

1. `_describe_crop()` accepts an optional `base_url` and `model` override. When called with the self-hosted URL, it creates a separate `OpenAI` client pointing at vLLM. The existing gpt-4o client is unchanged.

2. `enrich_summaries_with_vlm()` wraps each `_describe_crop()` call:
   - Try self-hosted first (if `settings.vlm_base_url` is set)
   - On `httpx.ConnectError`, `TimeoutError`, or any network exception: log warning, retry with gpt-4o
   - gpt-4o failures propagate as before (non-fatal, keeps OCR fallback)

### Changed: `document_process/pipeline.py`

Wrap the VLM enrichment step:

```python
from worker import model_server

server_healthy = False
if settings.use_vlm_summaries and settings.vlm_instance_id:
    server_healthy = model_server.ensure_running(settings)

try:
    summaries = enrich_summaries_with_vlm(summaries, settings=settings)
finally:
    if settings.vlm_instance_id:
        model_server.stop(settings)
```

### Changed: `cdk/app.py`

Instantiate `ModelServerStack` and pass outputs to `AppStack`.

### Changed: `cdk/stacks/app_stack.py`

- Add `VLM_INSTANCE_ID` and `VLM_BASE_URL` to the ECS task environment
- Grant the task role `ec2:StartInstances`, `ec2:StopInstances`, `ec2:DescribeInstances` scoped to the model server instance

---

## Data Flow

### Happy path
```
ensure_running()
  → describe_instances() → stopped
  → start_instances()
  → poll /health every 10s → 200 OK after ~2 min
enrich_summaries_with_vlm()
  → POST vLLM → description returned
model_server.stop()
  → stop_instances()
```

### Model server unreachable (spot reclaimed or startup failed)
```
POST vLLM → ConnectionError
  → log: "VLM server unreachable, falling back to gpt-4o"
  → POST OpenAI gpt-4o → description returned
model_server.stop() → no-op if already terminated
```

### ensure_running() timeout (3 min, no healthy response)
```
  → log warning: "Model server failed to start in 3 min"
  → returns False
  → vlm.py skips self-hosted, all images use gpt-4o directly
  → model_server.stop() called in finally block
```

### VLM disabled (USE_VLM_SUMMARIES=false)
```
  → ensure_running() never called
  → no EC2 activity, $0 cost
```

---

## Cost

| Scenario | Cost per document job |
|---|---|
| Model server starts, processes job | ~$0.03-0.05 (10-15 min × $0.16/hr spot) |
| Model server fails, gpt-4o fallback | ~$0.01/image (existing cost) |
| VLM disabled | $0 |
| Elastic IP while instance stopped | $0.005/hr |

---

## Testing

### Unit tests — `tests/unit/test_model_server.py` (new)

Mock `boto3` EC2 client:
- `ensure_running()` when instance already running → no `start_instances` call
- `ensure_running()` when instance stopped → `start_instances` called, polls until healthy
- `ensure_running()` timeout → returns `False`, no exception raised
- `stop()` when running → `stop_instances` called
- `stop()` when already stopped → no-op, no exception

### Unit tests — `tests/unit/test_vlm.py` (extend existing)

Mock HTTP:
- Self-hosted succeeds → gpt-4o never called
- Self-hosted `ConnectionError` → gpt-4o called as fallback
- Self-hosted timeout → gpt-4o called as fallback
- `vlm_base_url=None` → self-hosted never attempted, goes straight to gpt-4o

### Existing tests

Unaffected. `vlm_base_url` defaults to `None`, so all existing VLM tests exercise the gpt-4o path unchanged.

---

## Deployment Steps (after implementation)

1. `cdk deploy RagAgentModelServer` — provisions EC2 instance (stopped), Elastic IP, SG
2. Copy CDK outputs `ModelServerInstanceId` + `ModelServerElasticIp` into Secrets Manager or directly into `cdk deploy RagAgentApp` context
3. `cdk deploy RagAgentApp` — updates ECS task definition with new env vars + IAM permissions
4. Start an EC2 instance manually the first time to let user data script run (~15 min)
5. Verify vLLM healthy: `curl http://<elastic-ip>:8080/health`
6. Stop instance — it will be started on demand by the ECS Worker from this point
