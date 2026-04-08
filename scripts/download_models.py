"""
Pre-download PaddleOCR and PaddleX models at Docker build time.

Run during `docker build` (Stage 3) so models are baked into the image.
This means containers start instantly without needing to download models
on first use.

Called from Dockerfile:
    RUN python scripts/download_models.py
"""
import os
import sys

os.environ.setdefault("PADDLE_PDX_CACHE_HOME", "/app/paddle_models")
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

print("Downloading PaddleOCR models...")
try:
    from paddleocr import PaddleOCR
    PaddleOCR(use_gpu=False, show_log=False)
    print("PaddleOCR models ready.")
except Exception as exc:
    print(f"PaddleOCR download warning: {exc}", file=sys.stderr)

print("Downloading PaddleX layout detection models...")
try:
    from paddlex import create_pipeline
    create_pipeline(pipeline="PP-DocLayout_plus-L")
    print("PaddleX layout models ready.")
except Exception as exc:
    print(f"PaddleX download warning: {exc}", file=sys.stderr)

print("Model pre-download complete.")
