"""Pipeline configuration: S3, dataset IDs, sampling ratios, prompt templates."""
import os
from dataclasses import dataclass, field
from typing import Dict


# ---------------------------------------------------------------------------
# S3 / AWS
# ---------------------------------------------------------------------------

@dataclass
class S3Config:
    bucket: str = "your-sft-data-bucket"
    region: str = "us-east-1"
    prefix: str = "qwen3-vl-sft"              # root key prefix in the bucket

    # Sub-prefixes for datasets that require manual S3 caching
    # (ChartSumm and VisText originate from GitHub/GDrive, not HF Hub)
    chartsumm_prefix: str = "chartsumm"
    vistext_prefix: str = "vistext"

    # Sub-prefix for ShareGPT4V source images (COCO etc. must be uploaded separately)
    sharegpt4v_images_prefix: str = "sharegpt4v-images"

    def __post_init__(self) -> None:
        # Allow env-var overrides so callers don't need to hardcode credentials.
        self.bucket = os.environ.get("SFT_S3_BUCKET", self.bucket)
        self.region = os.environ.get("AWS_DEFAULT_REGION", self.region)


# ---------------------------------------------------------------------------
# Sampling weights  (must sum exactly to 1.0)
# ---------------------------------------------------------------------------
#   Chart captioning   (ChartCap + ChartSumm)         → 16 %
#   Scientific figures (ArXivCap + SciCap)            →  14 %
#   Document VQA       (DocVQA val + DocVQA full train) → 18 %
#   Document layout    (DocLayNet v1.2)                → 12 %
#   Multi-domain doc VQA (DUDE)                       → 12 %
#   Long-doc summarization (GovReport, text-only)     → 10 %
#   Chart-text align   (VisText)                      →  8 %
#   General visual     (ShareGPT4V)                   → 10 %

SAMPLING_WEIGHTS: Dict[str, float] = {
    "chartcap":     0.08,
    "chartsumm":    0.08,
    "arxivcap":     0.07,
    "scicap":       0.07,
    # lmms-lab/DocVQA validation (5 350 samples — kept for cross-split diversity)
    "docvqa":       0.06,
    # HuggingFaceM4/DocumentVQA full training split (39 463 samples)
    "docvqa_full":  0.12,
    # DocLayNet v1.2 — document layout understanding (69 103 pages)
    "doclaynet":    0.12,
    # DUDE — multi-domain document VQA (~27 k documents, ~100 k QA pairs)
    "dude":         0.12,
    # GovReport — long government-report summarization (text-only, 17 517 samples)
    "govreport":    0.10,
    "vistext":      0.08,
    "sharegpt4v":   0.10,
}

assert abs(sum(SAMPLING_WEIGHTS.values()) - 1.0) < 1e-9, "Weights must sum to 1.0"


# ---------------------------------------------------------------------------
# HuggingFace dataset identifiers
# ---------------------------------------------------------------------------

HF_REPOS: Dict[str, str] = {
    "chartcap":     "junyoung-00/ChartCap",
    "arxivcap":     "MMInstruction/ArxivCap",
    "scicap":       "CrowdAILab/scicap",
    # Original DocVQA — validation only (lmms-lab mirror)
    "docvqa":       "lmms-lab/DocVQA",
    # Full DocVQA training split via HuggingFaceM4 mirror
    "docvqa_full":  "HuggingFaceM4/DocumentVQA",
    # DocLayNet v1.2 — IBM document layout segmentation
    "doclaynet":    "ds4sd/DocLayNet",
    # DUDE — multi-domain document understanding (lmms-lab preprocessed)
    "dude":         "lmms-lab/DUDE",
    # GovReport — US government report summarization (text-only)
    "govreport":    "ccdv/govreport-summarization",
    "sharegpt4v":   "Lin-Chen/ShareGPT4V",
    # chartsumm and vistext are NOT on HF Hub — loaded from S3 cache
}

# Which split to use for each HF dataset
HF_SPLITS: Dict[str, str] = {
    "chartcap":     "train",
    "arxivcap":     "train",
    "scicap":       "train",
    # lmms-lab/DocVQA exposes only validation and test (test has no answers).
    "docvqa":       "validation",
    # Full training split: 39 463 QA pairs with document images.
    "docvqa_full":  "train",
    # DocLayNet v1.2: 69 103 train / 6 489 val / 4 993 test pages.
    "doclaynet":    "train",
    # DUDE: train split (test labels are withheld for the benchmark).
    "dude":         "train",
    # GovReport: 17 517 train / 973 val / 973 test reports.
    "govreport":    "train",
    # Use the curated 100 k GPT-4V subset; swap to "ShareGPT4V-PT" for 1.25 M.
    "sharegpt4v":   "ShareGPT4V",
}


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

PROMPTS: Dict[str, str] = {
    "chartcap": (
        "You are a chart analysis assistant. "
        "Describe this chart in detail, covering its type, axes, data series, "
        "key values, and the main insight it conveys."
    ),
    "chartsumm": (
        "You are a data visualization expert. "
        "Summarize the information presented in this chart. "
        "Highlight key trends, notable values, and the overall message."
    ),
    "arxivcap": (
        "You are a scientific figure captioning assistant. "
        "Describe what is shown in this figure from an academic paper, "
        "including its type, axes or labels if present, and the finding it illustrates."
    ),
    "scicap": (
        "You are a scientific figure captioning assistant. "
        "Write a detailed, accurate caption for this scientific figure, "
        "describing its content, methodology represented, and key results shown."
    ),
    # DocVQA and DocVQA-full share the same template; {question} filled at runtime.
    "docvqa": (
        "You are a document understanding assistant. "
        "Answer the following question based solely on the document image provided.\n"
        "Question: {question}"
    ),
    # DocLayNet layout description prompt.
    "doclaynet": (
        "You are a document layout analysis assistant. "
        "Identify and describe all document layout elements visible in this page. "
        "List each element type and how many instances of it appear."
    ),
    # DUDE reuses the DocVQA template (same VQA format); {question} filled at runtime.
    "dude": (
        "You are a document understanding assistant. "
        "Answer the following question based solely on the document image provided.\n"
        "Question: {question}"
    ),
    # GovReport long-document summarization (text-only; {report} filled at runtime).
    "govreport": (
        "You are a government document summarization assistant. "
        "Write a concise, factual summary of the following government report. "
        "Cover the main findings, recommendations, and conclusions.\n\n"
        "Report:\n{report}"
    ),
    "vistext": (
        "You are a chart captioning assistant. "
        "Generate a rich, semantically detailed caption for this chart. "
        "Include both the chart's visual/structural properties (L1) and "
        "the trends, comparisons, or statistical observations it conveys (L2/L3)."
    ),
    "sharegpt4v": "{question}",   # reuse the original GPT-4V human prompt verbatim
}


# ---------------------------------------------------------------------------
# DocLayNet category list (indices 0–10 match label IDs in the HF dataset)
# ---------------------------------------------------------------------------

DOCLAYNET_CATEGORIES = [
    "Caption",        # 0
    "Footnote",       # 1
    "Formula",        # 2
    "List-item",      # 3
    "Page-footer",    # 4
    "Page-header",    # 5
    "Picture",        # 6
    "Section-header", # 7
    "Table",          # 8
    "Text",           # 9
    "Title",          # 10
]


# ---------------------------------------------------------------------------
# External source metadata
# ---------------------------------------------------------------------------

# ChartSumm: 84 363 charts (bar/line/pie) with short/long summaries.
# Hosted on Google Drive. Use gdown to download.
# Expected local layout after download:
#   chartsumm/annotations/{train,val,test}/{chart_id}.json
#   chartsumm/images/{train,val,test}/{chart_id}.png
CHARTSUMM_GDRIVE_FOLDER_ID = "1HPsFUojoHctFD2AGuotRPKRz-o0jXfRJ"

# VisText: 12 441 charts with L1 (synthetic) + L2/L3 (human) captions.
# MIT GitHub. Run `bash download_data.sh --images` inside the repo.
# Expected local layout after download:
#   vistext/data/data_{train,validation,test}.json
#   vistext/data/images/{chart_id}.png
VISTEXT_DOWNLOAD_SCRIPT_URL = (
    "https://raw.githubusercontent.com/mitvis/vistext/main/download_data.sh"
)
