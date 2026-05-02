from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ScriptArguments:
    """Model, data, and LoRA arguments for Qwen3-VL-4B SFT.

    Paired with transformers.TrainingArguments via HfArgumentParser.
    Required TrainingArguments fields (e.g. --output_dir) must be passed
    on the command line or via a JSON config file.
    """

    # ── Model ──────────────────────────────────────────────────────────────────
    model_name_or_path: str = field(
        default="Qwen/Qwen3-VL-4B-Instruct",
        metadata={"help": "HuggingFace model ID or local path"},
    )
    attn_implementation: str = field(
        default="sdpa",
        metadata={"help": "Attention impl: 'sdpa' (default) or 'flash_attention_2' (requires flash-attn installed)"},
    )

    # ── Data ───────────────────────────────────────────────────────────────────
    data_path: str = field(
        default="dataset/train.json",
        metadata={"help": "Path to training JSONL/JSON file"},
    )
    eval_data_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to evaluation JSONL/JSON file; omit to skip eval"},
    )
    num_proc: int = field(
        default=4,
        metadata={"help": "Number of worker processes for data preprocessing"},
    )
    model_max_length: int = field(
        default=4096,
        metadata={"help": "Maximum total sequence length (tokens)"},
    )
    max_pixels: int = field(
        default=784 * 28 * 28,
        metadata={"help": "Upper pixel budget per image (~614K); reduces VRAM for high-res inputs"},
    )
    min_pixels: int = field(
        default=256 * 28 * 28,
        metadata={"help": "Lower pixel budget per image (~200K)"},
    )

    # ── LoRA ───────────────────────────────────────────────────────────────────
    lora_r: int = field(default=16, metadata={"help": "LoRA rank"})
    lora_alpha: int = field(default=32, metadata={"help": "LoRA alpha (2× lora_r is a common rule)"})
    lora_dropout: float = field(default=0.05, metadata={"help": "LoRA dropout"})

    # ── Quantization ───────────────────────────────────────────────────────────
    load_in_4bit: bool = field(
        default=True,
        metadata={"help": "QLoRA 4-bit loading via bitsandbytes (reduces VRAM ~4×)"},
    )

    # ── Multimodal component freeze control ────────────────────────────────────
    tune_mm_vision: bool = field(
        default=False,
        metadata={"help": "Fine-tune vision encoder; keep False for most SFT tasks"},
    )
    tune_mm_mlp: bool = field(
        default=False,
        metadata={"help": "Fine-tune vision-LLM projector; requires load_in_4bit=False for stability"},
    )
