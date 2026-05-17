# Qwen3-VL-4B SFT — Document Understanding Fine-Tuning

Supervised fine-tuning pipeline for **Qwen3-VL-4B-Instruct** on document OCR, layout understanding, document VQA, and long-document summarization. Trains on AWS EC2 (spot GPU) with QLoRA (4-bit via bitsandbytes).

---

## Datasets (11 sources)

| Dataset | Task | Size |
|---------|------|------|
| DocVQA / DUDE | Document Q&A | ~45 k + ~100 k |
| DocLayNet | Layout segmentation | 69 k pages |
| ChartCap / ChartSumm / VisText | Chart captioning & summarization | ~605 k |
| ArxivCap / SciCap | Scientific figure captioning | ~6.8 M |
| GovReport | Long-doc summarization (text-only) | 17 k |
| ShareGPT4V | General visual description | 102 k |

Sampling weights are defined in `data/config.py`.

---

## Project Structure

```
train.py          # HuggingFace Trainer SFT entry point (LoRA / QLoRA)
parameter.py      # ScriptArguments dataclass
data/
  config.py       # S3Config, dataset repos, sampling weights, prompts
  normalizers.py  # Per-dataset → unified {image, prompt, response} schema
  pipeline.py     # Dataset mixing and materialisation helpers
  s3_cache.py     # S3Cache + ChartSumm/VisText ingest helpers
aws_ec2/          # Terraform config for EC2 GPU training instance
SETUP.md          # End-to-end AWS setup and training run guide
```

---

## Quick Start

### Local dev (no S3 / GPU required)

```bash
pip install -r data/requirements.txt

python - <<'EOF'
from data import build_dataset, write_to_local_jsonl
ds = build_dataset(s3_cfg=None, skip_s3_datasets=True)
write_to_local_jsonl(ds, "data/train.jsonl", max_samples=2000)
EOF

python train.py \
  --data_path data/train.jsonl \
  --output_dir output/qwen3-vl-sft \
  --num_train_epochs 1 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --learning_rate 1e-5 \
  --bf16 True \
  --gradient_checkpointing True
```

### EC2 training

See [SETUP.md](SETUP.md) for the full AWS provisioning and training run guide.

---

## Key Hyperparameters

| Argument | Default | Notes |
|----------|---------|-------|
| `--model_name_or_path` | `Qwen/Qwen3-VL-4B-Instruct` | HF model ID or local path |
| `--load_in_4bit` | `True` | QLoRA; disable for full fine-tuning |
| `--lora_r` | `16` | LoRA rank |
| `--lora_alpha` | `32` | Standard 2× rank |
| `--model_max_length` | `4096` | Truncation length |
| `--max_pixels` | `614 k` | ~784×28×28 pixels per image |

---

## Environment Variables

| Variable | Purpose |
|----------|---------|
| `HF_TOKEN` | HuggingFace token (required for gated repos) |
| `SFT_S3_BUCKET` | Override S3 bucket at runtime |
| `AWS_PROFILE` / `AWS_DEFAULT_REGION` | boto3 credentials |

---

## Dependencies

```bash
pip install -r data/requirements.txt
# GPU only:
pip install bitsandbytes
```

Key packages: `transformers>=4.45`, `peft>=0.10`, `datasets>=2.18`, `boto3`, `gdown`.
