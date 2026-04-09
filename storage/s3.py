"""
S3 sync helpers for ECS Fargate deployments.

WHY this exists
---------------
ECS Fargate containers are ephemeral — their local filesystem is destroyed
when the container stops (deploy, crash, scale-to-zero event). This means
processed artifacts and the vector store would be lost on every restart.

The solution: S3 as the persistent layer.

  Startup:             sync S3 → local disk  (container gets the latest data)
  After preprocess:    local disk → S3        (new artifacts are saved)
  After index:         local disk → S3        (new vector store is saved)

This keeps ALL existing file-based code unchanged. The rest of the app
just reads/writes local files as always — the S3 sync is a thin wrapper.

WHY boto3 instead of `aws s3 sync`
-----------------------------------
The original implementation called `subprocess aws s3 sync`, which requires
the AWS CLI to be installed. The Debian `awscli` package pulls in Python 3.13
and ~90 system packages (~391 MB), causing out-of-memory errors during
Docker builds on machines with limited memory (Docker Desktop default: 2 GB).

boto3 is already in requirements.txt and handles S3 operations natively.
It is lighter, faster to install, and does not require any system package.

S3 bucket layout
----------------
s3://{bucket}/processed/    ← mirrors processed_documents_dir/
s3://{bucket}/embedded/     ← mirrors vectorstore_dir/

Authentication
--------------
In ECS, the Task Role (IAM) grants the container read/write access to S3.
No credentials need to be stored in the container — boto3 finds them
automatically via the instance metadata service.

Locally, boto3 uses your ~/.aws/credentials file or the standard
AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY environment variables.
"""

from __future__ import annotations

import logging
from pathlib import Path

from config import Settings

logger = logging.getLogger(__name__)


def sync_from_s3(settings: Settings) -> None:
    """
    Pull processed artifacts and vector store from S3 to local disk.
    Called once at container startup.
    """
    if not settings.s3_bucket_name:
        return

    logger.info("Syncing artifacts from s3://%s/", settings.s3_bucket_name)
    _download_prefix(
        bucket=settings.s3_bucket_name,
        prefix="processed/",
        local_dir=settings.processed_documents_dir,
        region=settings.aws_region,
    )
    _download_prefix(
        bucket=settings.s3_bucket_name,
        prefix="embedded/",
        local_dir=settings.vectorstore_dir,
        region=settings.aws_region,
    )
    logger.info("S3 sync complete")


def sync_processed_to_s3(settings: Settings) -> None:
    """
    Push local processed artifacts to S3.
    Called after a successful preprocess operation.
    """
    if not settings.s3_bucket_name:
        return

    logger.info("Uploading processed artifacts to s3://%s/processed/", settings.s3_bucket_name)
    _upload_directory(
        local_dir=settings.processed_documents_dir,
        bucket=settings.s3_bucket_name,
        prefix="processed/",
        region=settings.aws_region,
    )


def sync_embedded_to_s3(settings: Settings) -> None:
    """
    Push the local vector store to S3.
    Called after a successful index operation.
    """
    if not settings.s3_bucket_name:
        return

    logger.info("Uploading vector store to s3://%s/embedded/", settings.s3_bucket_name)
    _upload_directory(
        local_dir=settings.vectorstore_dir,
        bucket=settings.s3_bucket_name,
        prefix="embedded/",
        region=settings.aws_region,
    )


# ── Private helpers ───────────────────────────────────────────────────────────


def _get_client(region: str):
    """Return a boto3 S3 client. boto3 auto-discovers credentials from:
    1. Environment variables (AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY)
    2. ~/.aws/credentials (local development)
    3. ECS Task Role (production — no credentials needed in the container)
    """
    import boto3  # imported here so the rest of the app doesn't require boto3

    return boto3.client("s3", region_name=region)


def _upload_directory(*, local_dir: Path, bucket: str, prefix: str, region: str) -> None:
    """
    Upload all files under local_dir to s3://bucket/prefix/.

    The S3 key for each file is: prefix + relative path from local_dir.
    Example: local_dir=/app/data/processed, file=abc123/chunks.json
             → s3://bucket/processed/abc123/chunks.json
    """
    if not local_dir.exists():
        logger.debug("Upload skipped — directory does not exist: %s", local_dir)
        return

    client = _get_client(region)
    uploaded = 0

    for local_file in local_dir.rglob("*"):
        if not local_file.is_file():
            continue
        relative = local_file.relative_to(local_dir)
        s3_key = prefix + str(relative)
        logger.debug("Uploading %s → s3://%s/%s", local_file, bucket, s3_key)
        client.upload_file(str(local_file), bucket, s3_key)
        uploaded += 1

    logger.info("Uploaded %d file(s) to s3://%s/%s", uploaded, bucket, prefix)


def _download_prefix(*, bucket: str, prefix: str, local_dir: Path, region: str) -> None:
    """
    Download all objects under s3://bucket/prefix/ to local_dir/.

    The local path for each object is: local_dir / key_without_prefix.
    Example: s3://bucket/processed/abc123/chunks.json, local_dir=/app/data/processed
             → /app/data/processed/abc123/chunks.json
    """
    client = _get_client(region)
    local_dir.mkdir(parents=True, exist_ok=True)

    # list_objects_v2 returns up to 1000 objects per call.
    # The paginator handles directories with more than 1000 files automatically.
    paginator = client.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=bucket, Prefix=prefix)

    downloaded = 0
    for page in pages:
        for obj in page.get("Contents", []):
            key: str = obj["Key"]
            relative = key[len(prefix) :]
            if not relative:
                continue  # skip the prefix directory object itself

            local_file = local_dir / relative
            local_file.parent.mkdir(parents=True, exist_ok=True)

            logger.debug("Downloading s3://%s/%s → %s", bucket, key, local_file)
            client.download_file(bucket, key, str(local_file))
            downloaded += 1

    logger.info("Downloaded %d file(s) from s3://%s/%s", downloaded, bucket, prefix)
