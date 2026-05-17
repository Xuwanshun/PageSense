"""Qwen3-VL SFT data pipeline: loaders, mixing, and materialisation.

Public API
----------
    build_dataset(s3_cfg, weights, seed)
        → mixed datasets.IterableDataset ready for training

    write_to_local_jsonl(stream, output_path, ...)
        → JSONL + PNG files compatible with train.py

    write_to_s3_shards(stream, s3_cfg, ...)
        → gzipped JSONL shards on S3 for distributed training on EC2

    stream_from_s3(s3_cfg, split)
        → generator over pre-written S3 shards

Sampling ratios (config.SAMPLING_WEIGHTS)
-----------------------------------------
    16 % → ChartCap + ChartSumm        (chart captioning)
    14 % → ArXivCap + SciCap           (scientific figures)
    18 % → DocVQA val + DocVQA full    (document VQA)
    12 % → DocLayNet v1.2              (document layout understanding)
    12 % → DUDE                        (multi-domain document VQA)
    10 % → GovReport                   (long-doc summarization, text-only)
     8 % → VisText                     (chart-text alignment)
    10 % → ShareGPT4V                  (general visual description)

Unified sample schema
---------------------
    {"image": PIL.Image | None, "prompt": str, "response": str,
     "source": str, "metadata": dict}

    image=None is valid for text-only datasets (e.g. GovReport).
    write_to_local_jsonl and write_to_s3_shards handle None images gracefully:
    the "image" key is simply omitted from the JSONL record, which train.py
    handles as a text-only conversation turn.
"""
from __future__ import annotations

import gzip
import io
import json
import logging
import os
from collections import OrderedDict
from functools import partial
from pathlib import Path
from typing import Dict, Generator, Iterator, List, Optional

import datasets as hf
from PIL import Image

from .config import HF_REPOS, HF_SPLITS, SAMPLING_WEIGHTS, S3Config
from .normalizers import (
    normalize_arxivcap,
    normalize_chartcap,
    normalize_chartsumm,
    normalize_doclaynet,
    normalize_docvqa,
    normalize_docvqa_full,
    normalize_dude,
    normalize_govreport,
    normalize_scicap,
    normalize_sharegpt4v,
    normalize_vistext,
)
from .s3_cache import S3Cache, b64_to_pil, pil_to_b64

log = logging.getLogger(__name__)
_HF_TOKEN = os.environ.get("HF_TOKEN")


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

def _flatten(normalizer, raw_iter: Iterator[dict]) -> Generator[dict, None, None]:
    """Apply normalizer to each sample; flatten 1-to-N results; skip errors."""
    for i, sample in enumerate(raw_iter):
        try:
            yield from normalizer(sample)
        except Exception as exc:
            log.warning("Normalizer error at sample %d (%s): %s", i, type(exc).__name__, exc)


def _hf_stream(repo: str, split: str, normalizer, config_name: Optional[str] = None) -> hf.IterableDataset:
    """Stream a HF dataset and wrap every sample through a normalizer."""
    kwargs: dict = dict(split=split, streaming=True)
    if config_name:
        kwargs["name"] = config_name
    if _HF_TOKEN:
        kwargs["token"] = _HF_TOKEN
    log.info("Streaming %s split=%s", repo, split)
    return hf.IterableDataset.from_generator(
        _flatten,
        gen_kwargs={"normalizer": normalizer, "raw_iter": hf.load_dataset(repo, **kwargs)},
    )


# ---------------------------------------------------------------------------
# Per-dataset loaders
# ---------------------------------------------------------------------------

def load_chartcap() -> hf.IterableDataset:
    """junyoung-00/ChartCap — 508 783 train samples."""
    return _hf_stream(HF_REPOS["chartcap"], HF_SPLITS["chartcap"], normalize_chartcap)


def load_chartsumm(s3_cfg: S3Config, split: str = "train") -> hf.IterableDataset:
    """ChartSumm from S3 — 67 488 train charts. Run ingest_chartsumm() first."""
    cache = S3Cache(s3_cfg)
    if not cache.is_cached("chartsumm", split):
        raise RuntimeError(
            "ChartSumm not in S3. Run:  from data.s3_cache import ingest_chartsumm; "
            "ingest_chartsumm(S3Cache(cfg))"
        )
    def _gen():
        for raw in cache.stream_records("chartsumm", split):
            yield from normalize_chartsumm(raw)
    return hf.IterableDataset.from_generator(_gen)


def load_arxivcap() -> hf.IterableDataset:
    """MMInstruction/ArxivCap — 573 k papers, ~6.4 M figures (1 paper → N samples)."""
    return _hf_stream(HF_REPOS["arxivcap"], HF_SPLITS["arxivcap"], normalize_arxivcap)


def load_scicap() -> hf.IterableDataset:
    """CrowdAILab/scicap — ~400 k figures. Schema inconsistency handled gracefully."""
    repo  = HF_REPOS["scicap"]
    split = HF_SPLITS["scicap"]
    log.info("Streaming %s split=%s (schema inconsistencies handled gracefully)", repo, split)
    try:
        raw_ds = hf.load_dataset(repo, split=split, streaming=True)
    except Exception as exc:
        log.warning("SciCap fallback (no split kwarg): %s", exc)
        raw_ds = hf.load_dataset(repo, streaming=True)[split]
    return hf.IterableDataset.from_generator(
        _flatten, gen_kwargs={"normalizer": normalize_scicap, "raw_iter": raw_ds},
    )


def load_docvqa() -> hf.IterableDataset:
    """lmms-lab/DocVQA — validation split (5 350 samples with answers)."""
    return _hf_stream(HF_REPOS["docvqa"], HF_SPLITS["docvqa"], normalize_docvqa, config_name="DocVQA")


def load_docvqa_full() -> hf.IterableDataset:
    """HuggingFaceM4/DocumentVQA — full training split (39 463 QA pairs).

    Substantially larger than the lmms-lab mirror; provides the complete
    supervised signal for document VQA.
    """
    return _hf_stream(HF_REPOS["docvqa_full"], HF_SPLITS["docvqa_full"], normalize_docvqa_full)


def load_doclaynet() -> hf.IterableDataset:
    """ds4sd/DocLayNet v1.2 — 69 103 train document pages with layout annotations.

    Task: given a document page image, describe its layout structure
    (element types and counts: Title, Section-header, Text, Table, etc.).
    """
    return _hf_stream(HF_REPOS["doclaynet"], HF_SPLITS["doclaynet"], normalize_doclaynet)


def load_dude() -> hf.IterableDataset:
    """lmms-lab/DUDE — multi-domain Document Understanding Dataset & Evaluation.

    ~27 k documents, ~100 k QA pairs spanning diverse real-world document types
    (forms, receipts, scientific papers, invoices, etc.).
    Not-answerable samples are filtered by normalize_dude.
    """
    return _hf_stream(HF_REPOS["dude"], HF_SPLITS["dude"], normalize_dude)


def load_govreport() -> hf.IterableDataset:
    """ccdv/govreport-summarization — 17 517 US government report summaries.

    Text-only task: long document → concise summary.
    Samples have image=None; write_to_local_jsonl omits the "image" key,
    train.py treats them as text-only conversation turns.
    """
    return _hf_stream(HF_REPOS["govreport"], HF_SPLITS["govreport"], normalize_govreport)


def load_vistext(s3_cfg: S3Config, split: str = "train") -> hf.IterableDataset:
    """VisText from S3 — ~9 970 train charts. Run ingest_vistext() first."""
    cache = S3Cache(s3_cfg)
    if not cache.is_cached("vistext", split):
        raise RuntimeError(
            "VisText not in S3. Run:  from data.s3_cache import ingest_vistext; "
            "ingest_vistext(S3Cache(cfg))"
        )
    def _gen():
        for raw in cache.stream_records("vistext", split):
            yield from normalize_vistext(raw)
    return hf.IterableDataset.from_generator(_gen)


def load_sharegpt4v(s3_cfg: S3Config) -> hf.IterableDataset:
    """Lin-Chen/ShareGPT4V — 102 k GPT-4V caption samples. Images resolved from S3."""
    resolver = S3ImageResolver(s3_cfg)
    repo   = HF_REPOS["sharegpt4v"]
    subset = HF_SPLITS["sharegpt4v"]
    log.info("Streaming %s subset=%s", repo, subset)
    raw_ds = hf.load_dataset(
        repo, name=subset, split="train",
        streaming=True, token=_HF_TOKEN,
    )
    return hf.IterableDataset.from_generator(
        _flatten,
        gen_kwargs={"normalizer": partial(normalize_sharegpt4v, image_loader=resolver), "raw_iter": raw_ds},
    )


class S3ImageResolver:
    """Resolve ShareGPT4V relative image paths to PIL Images via S3.

    Pre-upload source collections before training:
        COCO 2017 train  (18 GB) → s3://bucket/prefix/sharegpt4v-images/coco/
        SAM              (11 TB) → s3://bucket/prefix/sharegpt4v-images/sam/
        LLaVA CC3M       ( 3 GB) → s3://bucket/prefix/sharegpt4v-images/llava/
        SBU captions     (10 GB) → s3://bucket/prefix/sharegpt4v-images/sbu/
    """

    _CACHE_MAX = 1024

    def __init__(self, s3_cfg: S3Config):
        import boto3
        self._client     = boto3.client("s3", region_name=s3_cfg.region)
        self._bucket     = s3_cfg.bucket
        self._img_prefix = f"{s3_cfg.prefix}/{s3_cfg.sharegpt4v_images_prefix}"
        self._cache: OrderedDict = OrderedDict()
        self._miss_count  = 0

    def __call__(self, image_path: str) -> Optional[Image.Image]:
        if image_path in self._cache:
            self._cache.move_to_end(image_path)
            return self._cache[image_path]
        try:
            obj = self._client.get_object(Bucket=self._bucket, Key=f"{self._img_prefix}/{image_path}")
            img = Image.open(io.BytesIO(obj["Body"].read())).convert("RGB")
            self._cache[image_path] = img
            if len(self._cache) > self._CACHE_MAX:
                self._cache.popitem(last=False)
            return img
        except Exception:
            self._miss_count += 1
            if self._miss_count % 1000 == 1:
                log.warning("ShareGPT4V: %d image misses. Upload images to s3://%s/%s/",
                            self._miss_count, self._bucket, self._img_prefix)
            return None


# ---------------------------------------------------------------------------
# Build mixed dataset
# ---------------------------------------------------------------------------

def _weighted_mix(streams: Dict, probs: list, seed: int = 42):
    """Weighted random interleaving as a plain Python generator.

    Avoids hf.interleave_datasets which tries to resolve PyArrow features and
    cannot handle PIL.Image values in the sample dicts.
    """
    import random
    rng = random.Random(seed)
    active_keys = list(streams.keys())
    active_probs = list(probs)
    iters = {k: iter(v) for k, v in streams.items()}

    while active_keys:
        key = rng.choices(active_keys, weights=active_probs, k=1)[0]
        try:
            yield next(iters[key])
        except StopIteration:
            idx = active_keys.index(key)
            active_keys.pop(idx)
            active_probs.pop(idx)


def build_dataset(
    s3_cfg: Optional[S3Config] = None,
    weights: Optional[Dict[str, float]] = None,
    seed: int = 42,
    skip_s3_datasets: bool = False,
):
    """Build a weighted mixed streaming SFT dataset from all 11 sources.

    Args:
        s3_cfg:           S3 config.  Required for ChartSumm, VisText, ShareGPT4V.
                          If None, those three are skipped and weights redistributed.
        weights:          Per-dataset sampling weights (defaults to SAMPLING_WEIGHTS).
        seed:             Interleaving seed.
        skip_s3_datasets: Skip S3-dependent sources (useful for local testing).

    Returns:
        Generator yielding {image, prompt, response, source, metadata} dicts.
        image may be None for text-only samples (GovReport).
    """
    if weights is None:
        weights = SAMPLING_WEIGHTS

    # HF-hosted datasets (no S3 required)
    streams: Dict[str, hf.IterableDataset] = {}
    for key, loader in [
        ("chartcap",    load_chartcap),
        ("arxivcap",    load_arxivcap),
        ("scicap",      load_scicap),
        ("docvqa",      load_docvqa),
        ("docvqa_full", load_docvqa_full),
        ("doclaynet",   load_doclaynet),
        ("dude",        load_dude),
        ("govreport",   load_govreport),
    ]:
        try:
            streams[key] = loader()
        except Exception as exc:
            log.warning("%s skipped: %s", key, exc)

    # S3-hosted datasets (ChartSumm, VisText, ShareGPT4V)
    if not skip_s3_datasets and s3_cfg is not None:
        for key, loader in [
            ("chartsumm",  lambda: load_chartsumm(s3_cfg)),
            ("vistext",    lambda: load_vistext(s3_cfg)),
            ("sharegpt4v", lambda: load_sharegpt4v(s3_cfg)),
        ]:
            try:
                streams[key] = loader()
            except Exception as exc:
                log.warning("%s skipped: %s", key, exc)
    elif not skip_s3_datasets:
        log.warning("s3_cfg is None — ChartSumm, VisText, ShareGPT4V skipped; weights redistributed.")

    active = {k: weights[k] for k in streams if k in weights}
    keys   = list(streams)
    total  = sum(active.get(k, 0.0) for k in keys)
    probs  = [active.get(k, 0.0) / total for k in keys]

    log.info("Mixing %d datasets:", len(keys))
    for k, p in zip(keys, probs):
        log.info("  %-16s  %.1f %%", k, p * 100)

    return _weighted_mix(streams, probs, seed)


# ---------------------------------------------------------------------------
# Materialise — local JSONL  (compatible with train.py)
# ---------------------------------------------------------------------------

def write_to_local_jsonl(
    stream: hf.IterableDataset,
    output_path: str,
    max_samples: Optional[int] = None,
    image_dir: Optional[str] = None,
) -> int:
    """Write the mixed stream to a local JSONL file for train.py.

    Image samples: image saved as PNG; JSONL line stores its absolute path.
    Text-only samples (image=None): "image" key is omitted from JSONL.

    Output format (matches VLMDataset in train.py):
        # image sample:
        {"image": "/abs/path/image.png",
         "conversations": [{"from": "human", "value": "<image>\\n<prompt>"},
                           {"from": "gpt",   "value": "<response>"}]}
        # text-only sample:
        {"conversations": [{"from": "human", "value": "<prompt>"},
                           {"from": "gpt",   "value": "<response>"}]}

    Returns number of records written.
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if image_dir is None:
        image_dir = str(out.parent / (out.stem + "_images"))
    Path(image_dir).mkdir(parents=True, exist_ok=True)

    count = 0
    with open(out, "w", encoding="utf-8") as fout:
        for sample in stream:
            if max_samples is not None and count >= max_samples:
                break

            img: Optional[Image.Image] = sample.get("image")
            prompt   = sample["prompt"]
            response = sample["response"]

            if img is not None:
                img_path = str(Path(image_dir) / f"{sample['source']}_{count:08d}.png")
                img.save(img_path, format="PNG")
                record = {
                    "image": img_path,
                    "conversations": [
                        {"from": "human", "value": f"<image>\n{prompt}"},
                        {"from": "gpt",   "value": response},
                    ],
                }
            else:
                # Text-only sample — no image key
                record = {
                    "conversations": [
                        {"from": "human", "value": prompt},
                        {"from": "gpt",   "value": response},
                    ],
                }

            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
            if count % 1000 == 0:
                log.info("write_to_local_jsonl: %d records…", count)

    log.info("write_to_local_jsonl: done — %d records → %s", count, out)
    return count


# ---------------------------------------------------------------------------
# Materialise — S3 shards  (for EC2 training)
# ---------------------------------------------------------------------------

def write_to_s3_shards(
    stream: hf.IterableDataset,
    s3_cfg: S3Config,
    split: str = "train",
    shard_size: int = 500,
    max_samples: Optional[int] = None,
) -> int:
    """Write the mixed stream to gzipped JSONL shards on S3.

    Shard path: s3://{bucket}/{prefix}/mixed/{split}/shard-{n:05d}.jsonl.gz
    Read back with stream_from_s3().

    image_b64 is omitted for text-only samples (image=None); stream_from_s3
    returns image=None for those records.

    Returns total records written.
    """
    import boto3
    client = boto3.client("s3", region_name=s3_cfg.region)
    shard_n, buf, total = 0, [], 0

    def _flush(idx: int, lines: list) -> None:
        key = f"{s3_cfg.prefix}/mixed/{split}/shard-{idx:05d}.jsonl.gz"
        client.put_object(
            Bucket=s3_cfg.bucket, Key=key,
            Body=gzip.compress("\n".join(lines).encode()),
            ContentType="application/gzip",
        )
        log.info("Uploaded s3://%s/%s (%d records)", s3_cfg.bucket, key, len(lines))

    for sample in stream:
        if max_samples is not None and total >= max_samples:
            break
        try:
            rec: dict = {
                "prompt":   sample["prompt"],
                "response": sample["response"],
                "source":   sample["source"],
                "metadata": sample.get("metadata", {}),
            }
            img: Optional[Image.Image] = sample.get("image")
            if img is not None:
                rec["image_b64"] = pil_to_b64(img)
            buf.append(json.dumps(rec, ensure_ascii=False))
            total += 1
        except Exception as exc:
            log.warning("Skipping sample (serialise error): %s", exc)
            continue

        if len(buf) >= shard_size:
            _flush(shard_n, buf)
            shard_n += 1
            buf = []

    if buf:
        _flush(shard_n, buf)
        shard_n += 1

    client.put_object(
        Bucket=s3_cfg.bucket,
        Key=f"{s3_cfg.prefix}/mixed/{split}/_index.json",
        Body=json.dumps({"total_records": total, "num_shards": shard_n}).encode(),
        ContentType="application/json",
    )
    log.info("write_to_s3_shards: %d records → %d shards at s3://%s/%s/mixed/%s/",
             total, shard_n, s3_cfg.bucket, s3_cfg.prefix, split)
    return total


# ---------------------------------------------------------------------------
# Stream back from S3 during training
# ---------------------------------------------------------------------------

def stream_from_s3(s3_cfg: S3Config, split: str = "train") -> Generator[dict, None, None]:
    """Stream pre-written mixed shards from S3, decoding images from base64.

    Reads shards written by write_to_s3_shards().
    Records without "image_b64" (text-only) are yielded with image=None.

    To use with HuggingFace Trainer:
        ds = hf.IterableDataset.from_generator(
                 stream_from_s3, gen_kwargs={"s3_cfg": cfg, "split": "train"})
    """
    for raw in S3Cache(s3_cfg).stream_records("mixed", split):
        try:
            img = b64_to_pil(raw["image_b64"]) if raw.get("image_b64") else None
            yield {
                "image":    img,
                "prompt":   raw["prompt"],
                "response": raw["response"],
                "source":   raw["source"],
                "metadata": raw.get("metadata", {}),
            }
        except Exception as exc:
            log.warning("stream_from_s3: decode error: %s", exc)
