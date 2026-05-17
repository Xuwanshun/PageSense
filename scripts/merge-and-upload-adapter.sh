#!/usr/bin/env bash
# Merges the Qwen3-VL LoRA adapter into the base model and uploads to HuggingFace Hub.
#
# Usage:
#   ./scripts/merge-and-upload-adapter.sh <hf-username> <repo-name>
#
# Example:
#   ./scripts/merge-and-upload-adapter.sh longzhuang qwen3-vl-rag-finetuned
#
# Requirements:
#   - AWS credentials configured (to pull adapter from S3)
#   - Python 3.10+ (not the project venv — this script creates its own)
#   - ~30 GB free disk space (base model ~15 GB + merged model ~15 GB)
#   - ~16 GB RAM (loads model on CPU for merge)

set -euo pipefail

HF_USERNAME="${1:?Usage: $0 <hf-username> <repo-name>}"
REPO_NAME="${2:?Usage: $0 <hf-username> <repo-name>}"

S3_BUCKET="qwen3-vl-sft-training-604561274097"
ADAPTER_DIR="./adapter-weights"
MERGED_DIR="./merged-model"
VENV_DIR="./merge-venv"

echo "==> Step 1: Create isolated Python environment"
python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

echo "==> Step 2: Install required packages"
pip install --quiet --upgrade pip
pip install --quiet \
  torch \
  transformers \
  peft \
  accelerate \
  huggingface_hub \
  safetensors \
  qwen-vl-utils

echo "==> Step 3: Download adapter from S3"
mkdir -p "$ADAPTER_DIR"
aws s3 sync "s3://${S3_BUCKET}/final-model/" "$ADAPTER_DIR/"
echo "    Downloaded adapter to $ADAPTER_DIR"

echo "==> Step 4: Merge adapter into base model"
python3 - <<'PYTHON'
import os, sys, json, torch
from pathlib import Path
from peft import PeftModel
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

adapter_dir = Path("./adapter-weights")
merged_dir  = Path("./merged-model")
merged_dir.mkdir(exist_ok=True)

# Read base model name from adapter config
with open(adapter_dir / "adapter_config.json") as f:
    adapter_cfg = json.load(f)
base_model_name = adapter_cfg.get("base_model_name_or_path", "Qwen/Qwen2.5-VL-7B-Instruct")
print(f"    Base model: {base_model_name}")

print("    Loading base model on CPU (this takes ~5 min and ~15 GB RAM)...")
base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    base_model_name,
    torch_dtype=torch.float16,
    device_map="cpu",
)

print("    Loading LoRA adapter...")
model = PeftModel.from_pretrained(base_model, str(adapter_dir))

print("    Merging adapter weights into base model...")
model = model.merge_and_unload()

print(f"    Saving merged model to {merged_dir} ...")
model.save_pretrained(str(merged_dir), safe_serialization=True)

print("    Copying processor/tokenizer files...")
processor = AutoProcessor.from_pretrained(str(adapter_dir))
processor.save_pretrained(str(merged_dir))

print("    Merge complete.")
PYTHON

echo "==> Step 5: Log in to HuggingFace and upload"
echo "    You will be prompted for your HuggingFace token."
echo "    Get one at: https://huggingface.co/settings/tokens (write access needed)"
huggingface-cli login

echo "    Uploading to huggingface.co/${HF_USERNAME}/${REPO_NAME} ..."
huggingface-cli upload "${HF_USERNAME}/${REPO_NAME}" "$MERGED_DIR" --repo-type model

echo ""
echo "==> Done!"
echo "    Model uploaded to: https://huggingface.co/${HF_USERNAME}/${REPO_NAME}"
echo ""
echo "==> Next steps:"
echo "    1. Go to https://ui.endpoints.huggingface.co/new"
echo "    2. Select model: ${HF_USERNAME}/${REPO_NAME}"
echo "    3. Cloud: AWS  |  Region: us-east-1 (or ca-central-1)"
echo "    4. Instance: nvidia-a10g-x1 (cheapest GPU for 7B model)"
echo "    5. Scale to zero: ENABLED"
echo "    6. Copy the endpoint URL — set it as VLM_BASE_URL in your .env"
echo ""
echo "==> Cleanup (optional — frees ~30 GB):"
echo "    rm -rf $ADAPTER_DIR $MERGED_DIR $VENV_DIR"
