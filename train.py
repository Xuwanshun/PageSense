"""Qwen3-VL-4B SFT training script for OCR caption generation and text summarization.

Dataset format (JSONL, one JSON object per line):

  Image sample (OCR / caption):
    {"image": "dataset/image/001.jpg", "conversations": [
        {"from": "human", "value": "<image>\nDescribe all text visible in this image."},
        {"from": "gpt",   "value": "The image contains the following text: ..."}
    ]}

  Text-only sample (summarization):
    {"conversations": [
        {"from": "human", "value": "Summarize the following article:\n<article text>"},
        {"from": "gpt",   "value": "Summary: ..."}
    ]}

Run example:
  python train.py \
    --output_dir output/qwen3-vl-sft \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --learning_rate 1e-5 \
    --max_grad_norm 1.0 \
    --bf16 True \
    --gradient_checkpointing True \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.03 \
    --save_steps 100 \
    --save_total_limit 3 \
    --logging_steps 10 \
    --eval_strategy no
"""

import contextlib
import io
import json
import logging
import os
import subprocess
from typing import List, Optional

import torch
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import Dataset
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    HfArgumentParser,
    Qwen3VLForConditionalGeneration,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

from parameter import ScriptArguments


class S3CheckpointCallback(TrainerCallback):
    """Sync checkpoints to S3 after every save (non-blocking)."""

    def __init__(self, s3_bucket: str, s3_prefix: str = "checkpoints"):
        self.s3_uri = f"s3://{s3_bucket}/{s3_prefix}/"

    def on_save(self, args, state, control, **kwargs):
        subprocess.Popen(
            ["aws", "s3", "sync", args.output_dir, self.s3_uri, "--region", "us-east-1"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return control

log = logging.getLogger(__name__)

IGNORE_INDEX = -100
# Resolved at runtime from the tokenizer in setup_model_and_processor().
IMAGE_TOKEN_INDEX: int = -1
VIDEO_TOKEN_INDEX: int = -1


# ── Dataset ────────────────────────────────────────────────────────────────────

class VLMDataset(Dataset):
    def __init__(self, data_path: str, processor, script_args: ScriptArguments):
        with open(data_path) as f:
            if data_path.endswith(".jsonl"):
                self.data = [json.loads(line) for line in f if line.strip()]
            else:
                self.data = json.load(f)

        if script_args.max_train_samples and len(self.data) > script_args.max_train_samples:
            self.data = self.data[: script_args.max_train_samples]

        self.processor = processor
        self.model_max_length = script_args.model_max_length

    def __len__(self) -> int:
        return len(self.data)

    def _build_messages(self, item: dict) -> List[dict]:
        """Convert conversations list to the processor's messages format."""
        messages = []
        image_path: Optional[str] = item.get("image")

        for turn in item["conversations"]:
            role = "user" if turn["from"] == "human" else "assistant"
            text: str = turn["value"]

            if role == "user" and image_path and "<image>" in text:
                content = [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": text.replace("<image>", "").strip()},
                ]
            else:
                content = [{"type": "text", "text": text}]

            messages.append({"role": role, "content": content})

        return messages

    def __getitem__(self, idx: int) -> dict:
        item = self.data[idx]

        # Skip samples whose image file is missing or empty (returns next valid sample)
        image_path = item.get("image")
        if image_path and (not os.path.exists(image_path) or os.path.getsize(image_path) == 0):
            log.warning("Skipping corrupt/missing image: %s", image_path)
            return self.__getitem__((idx + 1) % len(self.data))

        messages = self._build_messages(item)

        # Encode full conversation (prompt + response)
        full_inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            return_dict=True,
            return_tensors="pt",
        )
        full_inputs.pop("token_type_ids", None)

        # Determine prompt length without re-processing images: render prompt-only
        # to text, then tokenize text alone (cheap — no vision encoder pass).
        prompt_messages = [m for m in messages if m["role"] != "assistant"]
        prompt_text = self.processor.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        prompt_len = len(
            self.processor.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        )

        # Build labels: mask prompt tokens and image/video pad tokens from loss
        labels = full_inputs["input_ids"].clone()
        labels[0, :prompt_len] = IGNORE_INDEX
        labels[labels == IMAGE_TOKEN_INDEX] = IGNORE_INDEX
        labels[labels == VIDEO_TOKEN_INDEX] = IGNORE_INDEX

        # Truncate to model_max_length
        seq_len = full_inputs["input_ids"].shape[1]
        if seq_len > self.model_max_length:
            for key in list(full_inputs.keys()):
                v = full_inputs[key]
                if isinstance(v, torch.Tensor) and v.shape[-1] == seq_len:
                    full_inputs[key] = v[..., : self.model_max_length]
            labels = labels[..., : self.model_max_length]

        full_inputs["labels"] = labels
        return {k: v.squeeze(0) for k, v in full_inputs.items()}


# ── Data collator ──────────────────────────────────────────────────────────────

class VLMDataCollator:
    """Right-pads sequences to the batch maximum; stacks pixel tensors."""

    def __init__(self, processor):
        self.pad_id = (
            processor.tokenizer.pad_token_id
            or processor.tokenizer.eos_token_id
        )

    def __call__(self, features: List[dict]) -> dict:
        max_len = max(f["input_ids"].shape[-1] for f in features)

        input_ids, attention_masks, labels, mm_token_type_ids = [], [], [], []
        has_mm_token_type_ids = "mm_token_type_ids" in features[0]
        for f in features:
            pad_len = max_len - f["input_ids"].shape[-1]
            input_ids.append(
                torch.cat([f["input_ids"], torch.full((pad_len,), self.pad_id)])
            )
            attention_masks.append(
                torch.cat([f["attention_mask"], torch.zeros(pad_len, dtype=torch.long)])
            )
            labels.append(
                torch.cat([f["labels"], torch.full((pad_len,), IGNORE_INDEX)])
            )
            if has_mm_token_type_ids:
                mm_token_type_ids.append(
                    torch.cat([f["mm_token_type_ids"], torch.zeros(pad_len, dtype=torch.long)])
                )

        batch = {
            "input_ids": torch.stack(input_ids),
            "attention_mask": torch.stack(attention_masks),
            "labels": torch.stack(labels),
        }
        if has_mm_token_type_ids:
            batch["mm_token_type_ids"] = torch.stack(mm_token_type_ids)

        if "pixel_values" in features[0]:
            # pixel_values shape: (total_patches, C, patch_h, patch_w) — concat across batch
            batch["pixel_values"] = torch.cat(
                [f["pixel_values"] for f in features], dim=0
            )
            batch["image_grid_thw"] = torch.cat(
                [f["image_grid_thw"].view(-1, 3) for f in features], dim=0
            )

        return batch


# ── Model setup ────────────────────────────────────────────────────────────────

def setup_model_and_processor(script_args: ScriptArguments, training_args: TrainingArguments):
    global IMAGE_TOKEN_INDEX, VIDEO_TOKEN_INDEX

    bnb_config = None
    if script_args.load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        script_args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        attn_implementation=script_args.attn_implementation,
        device_map="auto",
        quantization_config=bnb_config,
    )

    # prepare_model_for_kbit_training sets up gradient checkpointing hooks and
    # ensures LoRA adapter layers stay in full precision while the base stays 4-bit.
    if script_args.load_in_4bit:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=training_args.gradient_checkpointing,
        )
    elif training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    # Optionally unfreeze vision encoder (frozen by default — usually not needed for SFT)
    if script_args.tune_mm_vision:
        for name, p in model.named_parameters():
            if "visual" in name and "merger" not in name:
                p.requires_grad = True

    # Optionally unfreeze vision-LLM projector (only safe without 4-bit)
    if script_args.tune_mm_mlp:
        if script_args.load_in_4bit:
            log.warning(
                "tune_mm_mlp=True with load_in_4bit=True may cause instability. "
                "Set load_in_4bit=False for projector fine-tuning."
            )
        for name, p in model.named_parameters():
            if "merger" in name:
                p.requires_grad = True

    lora_config = LoraConfig(
        r=script_args.lora_r,
        lora_alpha=script_args.lora_alpha,
        lora_dropout=script_args.lora_dropout,
        # Official Qwen3-VL target: attention projections only
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        model.print_trainable_parameters()
    log.info(buf.getvalue().strip())

    processor = AutoProcessor.from_pretrained(
        script_args.model_name_or_path,
        model_max_length=script_args.model_max_length,
    )

    # Set pixel budget once here so all dataset instances share consistent limits.
    processor.image_processor.max_pixels = script_args.max_pixels
    processor.image_processor.min_pixels = script_args.min_pixels

    # Derive special token indices from the actual tokenizer vocabulary so this
    # stays correct if the checkpoint is ever swapped for a different model family.
    IMAGE_TOKEN_INDEX = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    VIDEO_TOKEN_INDEX = processor.tokenizer.convert_tokens_to_ids("<|video_pad|>")
    log.info("IMAGE_TOKEN_INDEX=%d  VIDEO_TOKEN_INDEX=%d", IMAGE_TOKEN_INDEX, VIDEO_TOKEN_INDEX)

    return model, processor


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    parser = HfArgumentParser((ScriptArguments, TrainingArguments))
    script_args, training_args = parser.parse_args_into_dataclasses()

    # Validate data paths before spending time loading the model.
    if not os.path.isfile(script_args.data_path):
        raise FileNotFoundError(f"Training data not found: {script_args.data_path!r}")
    if script_args.eval_data_path and not os.path.isfile(script_args.eval_data_path):
        raise FileNotFoundError(f"Eval data not found: {script_args.eval_data_path!r}")

    model, processor = setup_model_and_processor(script_args, training_args)

    train_dataset = VLMDataset(script_args.data_path, processor, script_args)
    eval_dataset = (
        VLMDataset(script_args.eval_data_path, processor, script_args)
        if script_args.eval_data_path
        else None
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=VLMDataCollator(processor),
        callbacks=[S3CheckpointCallback("qwen3-vl-sft-training-604561274097")],
    )
    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    trainer.save_model(training_args.output_dir)
    processor.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    main()
