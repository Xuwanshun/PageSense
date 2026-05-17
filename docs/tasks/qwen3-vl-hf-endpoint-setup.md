# Task: Deploy Qwen3-VL Fine-Tuned Model to HuggingFace Inference Endpoint

## Background

We have a fine-tuned vision-language model (Qwen3-VL) stored in AWS S3. The goal of this task is to:

1. Merge the LoRA adapter with the base model
2. Upload the merged model to HuggingFace Hub
3. Create a HuggingFace Inference Endpoint
4. Add the endpoint URL and token to AWS Secrets Manager

Once done, our document processing pipeline will use this model instead of gpt-4o to describe tables and figures in PDFs.

---

## Prerequisites

Make sure you have the following before starting:

- [ ] Access to this repository (cloned locally)
- [ ] AWS CLI installed and configured with credentials for account `604561274097`
- [ ] Python 3.10+ installed
- [ ] ~30 GB free disk space
- [ ] ~16 GB RAM
- [ ] A HuggingFace account — create one free at https://huggingface.co/join
- [ ] A HuggingFace write token — get one at https://huggingface.co/settings/tokens (click **New token**, select **Write** access)

---

## Step 1 — Run the merge and upload script

From the root of the repository, run:

```bash
./scripts/merge-and-upload-adapter.sh <hf-username> qwen3-vl-rag-finetuned
```

Replace `<hf-username>` with your HuggingFace username.

**What this does:**
- Downloads the fine-tuned adapter weights from S3
- Merges them into the base Qwen2.5-VL-7B model
- Uploads the merged model to HuggingFace Hub at `https://huggingface.co/<hf-username>/qwen3-vl-rag-finetuned`

**Expected duration:** ~15-20 minutes total (model download + merge + upload)

When prompted, paste your HuggingFace write token.

---

## Step 2 — Create the HuggingFace Inference Endpoint

1. Go to https://ui.endpoints.huggingface.co/new
2. Fill in the form:
   - **Model:** `<hf-username>/qwen3-vl-rag-finetuned`
   - **Endpoint name:** `qwen3-vl-rag`
   - **Cloud:** AWS
   - **Region:** `us-east-1`
   - **Instance type:** `Nvidia A10G · 1x GPU` (search for `nvidia-a10g-x1`)
   - **Task:** Text Generation
   - **Scale to zero:** toggle ON (important — keeps cost at $0 when idle)
3. Click **Create Endpoint**
4. Wait ~5 minutes for the endpoint to be ready (status shows **Running**)
5. Copy the **Endpoint URL** — it looks like:
   ```
   https://abc123xyz.us-east-1.aws.endpoints.huggingface.cloud
   ```

---

## Step 3 — Test the endpoint

Run this curl command to verify it works (replace the URL and token):

```bash
curl https://<your-endpoint-url>/v1/chat/completions \
  -H "Authorization: Bearer hf_..." \
  -H "Content-Type: application/json" \
  -d '{
    "model": "tgi",
    "messages": [{"role": "user", "content": "Hello, what can you do?"}],
    "max_tokens": 100
  }'
```

You should get a JSON response with a `choices` array. If you get a 200 response, the endpoint is working.

---

## Step 4 — Add credentials to AWS Secrets Manager

Run these two commands (replace placeholders with real values):

```bash
# Store the HuggingFace API token
aws secretsmanager create-secret \
  --region ca-central-1 \
  --name rag-agent/vlm-hf-token \
  --secret-string "hf_YOUR_TOKEN_HERE"

# Store the endpoint URL
aws secretsmanager create-secret \
  --region ca-central-1 \
  --name rag-agent/vlm-base-url \
  --secret-string "https://YOUR_ENDPOINT_URL/v1"
```

---

## Step 5 — Report back

Once done, share the following with the person who assigned this task:

- [ ] HuggingFace model URL (e.g. `https://huggingface.co/<username>/qwen3-vl-rag-finetuned`)
- [ ] HuggingFace endpoint URL (e.g. `https://abc123.us-east-1.aws.endpoints.huggingface.cloud`)
- [ ] Confirmation that both Secrets Manager secrets were created successfully
- [ ] Screenshot or curl output showing the endpoint returned a valid response

---

## Troubleshooting

**Script fails at "Loading base model" with OOM error:**
Close other applications to free RAM. The model requires ~16 GB.

**Script fails at S3 download with "Access Denied":**
Check your AWS credentials: `aws sts get-caller-identity` — should show account `604561274097`.

**HuggingFace endpoint stays in "Initializing" for more than 15 min:**
Check the endpoint logs in the HF UI. If the model format is incompatible, contact the person who assigned this task.

**Endpoint returns 503:**
The endpoint may have scaled to zero. Send the request again — it will warm up in ~30 seconds.
