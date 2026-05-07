"""S3 utilities and incremental caching for GitHub/GDrive-hosted datasets.

HuggingFace-hosted datasets (ChartCap, ArXivCap, SciCap, DocVQA, ShareGPT4V)
stream directly from HF Hub — no S3 needed.

ChartSumm and VisText are not on HF Hub and must be downloaded once, normalised,
and uploaded to S3 as gzipped JSONL shards.  On subsequent runs the pipeline
streams directly from S3 without re-downloading.

S3 shard layout:
    s3://{bucket}/{prefix}/chartsumm/train/shard-{n:05d}.jsonl.gz
    s3://{bucket}/{prefix}/vistext/train/shard-{n:05d}.jsonl.gz

Each shard line:
    {"image_b64": "<base64-png>", "prompt": str, "response": str,
     "source": str, "metadata": {...}}
"""
from __future__ import annotations

import base64
import gzip
import io
import json
import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Generator, Iterator, List, Optional

import boto3
from botocore.exceptions import ClientError
from PIL import Image

from .config import CHARTSUMM_GDRIVE_FOLDER_ID, PROMPTS, S3Config, VISTEXT_DOWNLOAD_SCRIPT_URL

log = logging.getLogger(__name__)

SHARD_SIZE = 500  # records per shard file


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def pil_to_b64(image: Image.Image, fmt: str = "PNG") -> str:
    buf = io.BytesIO()
    image.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode()


def b64_to_pil(b64: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")


# ---------------------------------------------------------------------------
# S3Cache
# ---------------------------------------------------------------------------

class S3Cache:
    """Thin boto3 wrapper for shard existence checks, upload, and streaming."""

    def __init__(self, cfg: S3Config):
        self.cfg = cfg
        self._client = boto3.client("s3", region_name=cfg.region)

    def _key(self, *parts: str) -> str:
        return "/".join([self.cfg.prefix, *parts])

    def shard_key(self, dataset: str, split: str, n: int) -> str:
        return self._key(dataset, split, f"shard-{n:05d}.jsonl.gz")

    def index_key(self, dataset: str, split: str) -> str:
        return self._key(dataset, split, "_index.json")

    def shard_exists(self, dataset: str, split: str, n: int) -> bool:
        try:
            self._client.head_object(Bucket=self.cfg.bucket, Key=self.shard_key(dataset, split, n))
            return True
        except ClientError:
            return False

    def list_shards(self, dataset: str, split: str) -> List[str]:
        prefix = self._key(dataset, split, "shard-")
        paginator = self._client.get_paginator("list_objects_v2")
        keys = []
        for page in paginator.paginate(Bucket=self.cfg.bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                keys.append(obj["Key"])
        return sorted(keys)

    def is_cached(self, dataset: str, split: str) -> bool:
        """True when at least one shard exists for (dataset, split)."""
        return len(self.list_shards(dataset, split)) > 0

    def upload_shards(
        self,
        dataset: str,
        split: str,
        records: Iterator[dict],
        start_shard: int = 0,
    ) -> int:
        """Stream records into gzipped JSONL shards on S3.

        Skips shards that already exist — safe to resume after interruption.
        Returns total records written.
        """
        shard_n = start_shard
        buf: List[str] = []
        total = 0

        def _flush(idx: int, lines: List[str]) -> None:
            key = self.shard_key(dataset, split, idx)
            self._client.put_object(
                Bucket=self.cfg.bucket, Key=key,
                Body=gzip.compress("\n".join(lines).encode()),
                ContentType="application/gzip",
            )
            log.info("Uploaded s3://%s/%s (%d records)", self.cfg.bucket, key, len(lines))

        for record in records:
            buf.append(json.dumps(record, ensure_ascii=False))
            total += 1
            if len(buf) >= SHARD_SIZE:
                if not self.shard_exists(dataset, split, shard_n):
                    _flush(shard_n, buf)
                else:
                    log.debug("Shard %d already exists, skipping.", shard_n)
                shard_n += 1
                buf = []

        if buf:
            if not self.shard_exists(dataset, split, shard_n):
                _flush(shard_n, buf)
            shard_n += 1

        self._client.put_object(
            Bucket=self.cfg.bucket,
            Key=self.index_key(dataset, split),
            Body=json.dumps({"total_records": total, "num_shards": shard_n}).encode(),
            ContentType="application/json",
        )
        log.info("Cached '%s/%s': %d records, %d shards.", dataset, split, total, shard_n)
        return total

    def stream_records(self, dataset: str, split: str) -> Generator[dict, None, None]:
        """Stream all records from S3 shards one at a time (no full download)."""
        keys = self.list_shards(dataset, split)
        if not keys:
            raise RuntimeError(
                f"No S3 shards found for '{dataset}/{split}'. "
                "Run the corresponding ingest function first."
            )
        for key in keys:
            obj = self._client.get_object(Bucket=self.cfg.bucket, Key=key)
            for line in gzip.decompress(obj["Body"].read()).decode().splitlines():
                line = line.strip()
                if line:
                    yield json.loads(line)


# ---------------------------------------------------------------------------
# ChartSumm ingest: Google Drive → S3
# ---------------------------------------------------------------------------

def ingest_chartsumm(cache: S3Cache, split: str = "train", force: bool = False) -> None:
    """Download ChartSumm from Google Drive and cache to S3.

    Requires:  pip install gdown
    ~15 GB download (images + JSON annotations).
    Run once on an EC2 instance with a fast internet connection.
    """
    if not force and cache.is_cached("chartsumm", split):
        log.info("ChartSumm '%s' already cached. Use force=True to re-ingest.", split)
        return

    try:
        import gdown
    except ImportError:
        raise ImportError("pip install gdown")

    with tempfile.TemporaryDirectory(prefix="chartsumm_") as tmp:
        tmpdir = Path(tmp)
        log.info("Downloading ChartSumm from Google Drive (folder %s)…", CHARTSUMM_GDRIVE_FOLDER_ID)
        gdown.download_folder(id=CHARTSUMM_GDRIVE_FOLDER_ID, output=str(tmpdir / "chartsumm"), quiet=False)
        cache.upload_shards("chartsumm", split, _parse_chartsumm_dir(tmpdir / "chartsumm", split))


def _parse_chartsumm_dir(root: Path, split: str) -> Generator[dict, None, None]:
    """Yield normalised records from the downloaded ChartSumm directory.

    Layout:
        root/annotations/{split}/{chart_id}.json   ← {x_label, y_label, data, title, summary}
        root/images/{split}/{chart_id}.png
    """
    ann_dir = root / "annotations" / split
    img_dir = root / "images" / split

    if not ann_dir.exists():
        raise FileNotFoundError(f"Annotations not found at {ann_dir}.")

    for ann_path in sorted(ann_dir.glob("*.json")):
        chart_id = ann_path.stem
        img_path = img_dir / f"{chart_id}.png"

        try:
            ann = json.loads(ann_path.read_text())
        except Exception as exc:
            log.warning("Skipping %s: JSON error: %s", chart_id, exc)
            continue

        if not img_path.exists():
            log.warning("Skipping %s: image missing.", chart_id)
            continue

        summary = ann.get("summary", "").strip()
        if not summary:
            continue

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as exc:
            log.warning("Skipping %s: PIL error: %s", chart_id, exc)
            continue

        yield {
            "image_b64": pil_to_b64(image),
            "prompt":    PROMPTS["chartsumm"],
            "response":  summary,
            "source":    "chartsumm",
            "metadata":  {"chart_id": chart_id, "title": ann.get("title", ""), "x_label": ann.get("x_label", "")},
        }


# ---------------------------------------------------------------------------
# VisText ingest: MIT GitHub → S3
# ---------------------------------------------------------------------------

def ingest_vistext(
    cache: S3Cache,
    split: str = "train",
    local_data_root: Optional[str] = None,
    force: bool = False,
) -> None:
    """Download VisText from the MIT GitHub release and cache to S3.

    Pass `local_data_root` if you have already run `download_data.sh --images`
    locally; otherwise the script is fetched and run automatically.

    Layout expected under local_data_root (or auto-downloaded temp dir):
        data/data_{train,validation,test}.json
        data/images/{chart_id}.png
    """
    if not force and cache.is_cached("vistext", split):
        log.info("VisText '%s' already cached. Use force=True to re-ingest.", split)
        return

    root = Path(local_data_root) if local_data_root else _download_vistext()
    cache.upload_shards("vistext", split, _parse_vistext_dir(root, split))


def _download_vistext() -> Path:
    import requests
    tmpdir = Path(tempfile.mkdtemp(prefix="vistext_"))
    script = tmpdir / "download_data.sh"
    log.info("Fetching VisText download script…")
    script.write_text(requests.get(VISTEXT_DOWNLOAD_SCRIPT_URL, timeout=30).text)
    script.chmod(0o755)
    log.info("Running VisText download script (may take a while)…")
    subprocess.run(["bash", str(script), "--images"], cwd=str(tmpdir), check=True)
    return tmpdir


def _parse_vistext_dir(root: Path, split: str) -> Generator[dict, None, None]:
    """Yield normalised records from the downloaded VisText directory.

    Caption priority: caption_L2L3 > caption_L2 > caption_L1
    """
    split_file = {"train": "data_train.json", "validation": "data_validation.json", "test": "data_test.json"}[split]
    json_path = root / "data" / split_file
    img_dir   = root / "data" / "images"

    if not json_path.exists():
        raise FileNotFoundError(f"VisText data not found at {json_path}.")

    for rec in json.loads(json_path.read_text()):
        chart_id = rec.get("chart_id", "")
        img_path = img_dir / f"{chart_id}.png"

        if not img_path.exists():
            log.warning("VisText: image missing for %s, skipping.", chart_id)
            continue

        caption = (rec.get("caption_L2L3") or rec.get("caption_L2") or rec.get("caption_L1") or "").strip()
        if not caption:
            continue

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as exc:
            log.warning("VisText: PIL error for %s: %s", chart_id, exc)
            continue

        yield {
            "image_b64": pil_to_b64(image),
            "prompt":    PROMPTS["vistext"],
            "response":  caption,
            "source":    "vistext",
            "metadata":  {
                "chart_id": chart_id,
                "caption_level": "L2L3" if rec.get("caption_L2L3") else ("L2" if rec.get("caption_L2") else "L1"),
            },
        }
