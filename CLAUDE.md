# Qwen3-VL-4B SFT — Project Reference

## Overview

Supervised fine-tuning (SFT) pipeline for **Qwen3-VL-4B-Instruct** targeting document OCR, layout understanding, document VQA, and long-document summarization. Training runs on AWS EC2 (GPU) with optional QLoRA (4-bit via bitsandbytes).

Current branch: `VLM_SFT`

---

## Repository Structure

```
.
├── train.py              # HuggingFace Trainer SFT script (LoRA / QLoRA)
├── parameter.py          # ScriptArguments dataclass (model, data, LoRA hyperparams)
├── data/
│   ├── __init__.py       # Public API: build_dataset, write_to_local_jsonl, write_to_s3_shards, stream_from_s3
│   ├── config.py         # S3Config, HF_REPOS, HF_SPLITS, SAMPLING_WEIGHTS, PROMPTS
│   ├── normalizers.py    # Per-dataset normalizers → unified {image, prompt, response, source, metadata}
│   ├── pipeline.py       # Loaders, dataset mixing (hf.interleave_datasets), materialisation helpers
│   ├── s3_cache.py       # S3Cache, ChartSumm/VisText ingest helpers
│   ├── requirements.txt  # Python dependencies
│   ├── image/            # Local image cache (gitignored content)
│   └── text/             # Local text cache (gitignored content)
├── aws_ec2/              # Terraform infra for GPU training instance
│   ├── main.tf
│   ├── variables.tf
│   ├── output.tf
│   └── userdata.sh.tpl
└── CLAUDE.md             # This file
```

---

## Datasets (11 sources)

### HuggingFace-hosted (streamed directly, no pre-processing needed)

| Key | HF Repo | Split | Size | Task |
|-----|---------|-------|------|------|
| `chartcap` | `junyoung-00/ChartCap` | train | 508 783 | Chart captioning |
| `arxivcap` | `MMInstruction/ArxivCap` | train | ~6.4 M figures | Scientific figure captioning |
| `scicap` | `CrowdAILab/scicap` | train | ~400 k | Scientific figure captioning |
| `docvqa` | `lmms-lab/DocVQA` | validation | 5 350 | Document VQA |
| `docvqa_full` | `HuggingFaceM4/DocumentVQA` | train | 39 463 | Document VQA (full training set) |
| `doclaynet` | `ds4sd/DocLayNet` | train | 69 103 | Document layout understanding |
| `dude` | `lmms-lab/DUDE` | train | ~100 k QA pairs | Multi-domain document VQA |
| `govreport` | `ccdv/govreport-summarization` | train | 17 517 | Long-doc summarization (**text-only**) |
| `sharegpt4v` | `Lin-Chen/ShareGPT4V` | ShareGPT4V | 102 k | General visual description |

### S3-hosted (must be ingested once before training)

| Key | Source | Size | Notes |
|-----|--------|------|-------|
| `chartsumm` | Google Drive → S3 | 84 363 charts | Run `ingest_chartsumm(S3Cache(cfg))` |
| `vistext` | MIT GitHub → S3 | 12 441 charts | Run `ingest_vistext(S3Cache(cfg))` |

ShareGPT4V source images (COCO, SAM, LLaVA-CC3M, SBU) must also be uploaded to S3 manually — see `S3ImageResolver` docstring in `pipeline.py`.

---

## Unified Sample Schema

Every normalizer returns `List[dict]` with this schema:

```python
{
    "image":    PIL.Image.Image | None,  # None for GovReport (text-only)
    "prompt":   str,
    "response": str,
    "source":   str,                     # dataset key, e.g. "docvqa_full"
    "metadata": dict,                    # dataset-specific fields
}
```

Text-only samples (`image=None`) are supported. `write_to_local_jsonl` omits the `"image"` key for these records; `train.py`'s `VLMDataset` handles them as pure-text conversation turns.

---

## Sampling Weights

Defined in `data/config.py::SAMPLING_WEIGHTS` (must sum to 1.0):

```
chartcap      8 %
chartsumm     8 %
arxivcap      7 %
scicap        7 %
docvqa        6 %   ← validation split (5 k), kept for diversity
docvqa_full  12 %   ← full train split (39 k)
doclaynet    12 %   ← layout understanding
dude         12 %   ← multi-domain doc VQA
govreport    10 %   ← long-doc summarization (text-only)
vistext       8 %
sharegpt4v   10 %
```

---

## Data Pipeline Usage

### Local training (small-scale / dev)

```python
from data import build_dataset, write_to_local_jsonl

# HF-only datasets (no S3 required)
ds = build_dataset(s3_cfg=None, skip_s3_datasets=True)
write_to_local_jsonl(ds, "data/train.jsonl", max_samples=5000)
```

### Full pipeline (EC2 + S3)

```python
from data import build_dataset, write_to_s3_shards
from data.config import S3Config

cfg = S3Config(bucket="my-sft-bucket", region="us-east-1")

# One-time ingest of S3-hosted datasets
from data.s3_cache import ingest_chartsumm, ingest_vistext, S3Cache
cache = S3Cache(cfg)
ingest_chartsumm(cache)
ingest_vistext(cache)

# Build and upload mixed shards
ds = build_dataset(s3_cfg=cfg)
write_to_s3_shards(ds, cfg, split="train", shard_size=500)
```

### Stream from S3 during training

```python
import datasets as hf
from data import stream_from_s3
from data.config import S3Config

cfg = S3Config(bucket="my-sft-bucket")
train_ds = hf.IterableDataset.from_generator(
    stream_from_s3, gen_kwargs={"s3_cfg": cfg, "split": "train"}
)
```

---

## Training

### Minimal run (local JSONL)

```bash
python train.py \
  --data_path data/train.jsonl \
  --output_dir output/qwen3-vl-sft \
  --num_train_epochs 3 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --learning_rate 1e-5 \
  --bf16 True \
  --gradient_checkpointing True \
  --lr_scheduler_type cosine \
  --warmup_ratio 0.03 \
  --save_steps 100 \
  --logging_steps 10
```

### Key `ScriptArguments` (parameter.py)

| Arg | Default | Notes |
|-----|---------|-------|
| `model_name_or_path` | `Qwen/Qwen3-VL-4B-Instruct` | HF model ID or local path |
| `load_in_4bit` | `True` | QLoRA; disables for projector fine-tuning |
| `lora_r` | `16` | LoRA rank |
| `lora_alpha` | `32` | 2× rank is standard |
| `model_max_length` | `4096` | Truncation length |
| `max_pixels` | `784×28×28` | ~614 k pixels per image |
| `tune_mm_vision` | `False` | Fine-tune vision encoder (rarely needed) |
| `tune_mm_mlp` | `False` | Fine-tune vision-LLM projector |

---

## DocLayNet Specifics

DocLayNet v1.2 provides page-level layout segmentation with 11 element categories:
`Caption`, `Footnote`, `Formula`, `List-item`, `Page-footer`, `Page-header`, `Picture`, `Section-header`, `Table`, `Text`, `Title`

The normalizer (`normalize_doclaynet`) reads `labels_block` (v1.2 schema) or `labels` (older schema) and generates a structured natural-language count response, e.g.:

```
This document page contains the following layout elements:
- 1 Title
- 2 Section-headers
- 5 Texts
- 1 Table
- 1 Page-footer
```

---

## GovReport Specifics

Text-only dataset — no image. Reports are truncated to 12 000 characters before insertion into the prompt template to stay within `model_max_length=4096` tokens. The `truncated` flag is stored in sample metadata.

---

## DUDE Specifics

Multi-domain doc VQA with extractive, abstractive, and not-answerable question types. `normalize_dude` skips not-answerable samples (no useful generation target) and handles both single-image and multi-page documents (uses first page only for multi-page).

---

## AWS / Infrastructure

Terraform config in `aws_ec2/` provisions a GPU EC2 instance for training. Set variables in `aws_ec2/variables.tf` or via `-var` flags. Instance user-data (`userdata.sh.tpl`) installs dependencies and launches training.

---

## Environment Variables

| Variable | Purpose |
|----------|---------|
| `HF_TOKEN` | HuggingFace access token (needed for gated repos) |
| `SFT_S3_BUCKET` | Override `S3Config.bucket` at runtime |
| `AWS_PROFILE` / `AWS_DEFAULT_REGION` | boto3 credentials |

---

## Dependencies

See `data/requirements.txt`. Key packages:
- `datasets>=2.18.0` — streaming HF datasets
- `transformers>=4.45.0` — Qwen3-VL model + processor
- `peft>=0.10.0` — LoRA
- `bitsandbytes` — 4-bit QLoRA (install separately; GPU-only)
- `gdown>=5.1.0` — ChartSumm Google Drive download
- `boto3>=1.34.0` — S3 access
