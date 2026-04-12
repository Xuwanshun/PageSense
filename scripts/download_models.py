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

# Must patch before any model creation — same workaround as services.py.
# Without this, PaddleOCR crashes on x86 with NotImplementedError (PaddlePaddle #77340),
# the exception is swallowed by the try/except, and no models get baked into the image.
import paddle.inference as _pi  # noqa: E402

_pi.Config.enable_mkldnn = lambda self: None  # type: ignore[method-assign]

print("Downloading PaddleOCR models...")
try:
    from paddleocr import PaddleOCR

    PaddleOCR(
        use_gpu=False,
        show_log=False,
        text_detection_model_name="PP-OCRv4_mobile_det",
        text_recognition_model_name="PP-OCRv4_mobile_rec",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        enable_mkldnn=False,
    )
    print("PaddleOCR models ready.")
except Exception as exc:
    print(f"PaddleOCR download warning: {exc}", file=sys.stderr)
    sys.exit(1)  # fail the build so the error is visible, not silently skipped

print("Downloading PaddleX layout detection models...")
try:
    from paddlex import create_pipeline

    create_pipeline(pipeline="PP-DocLayout_plus-L")
    print("PaddleX layout models ready.")
except Exception as exc:
    print(f"PaddleX download warning: {exc}", file=sys.stderr)
    sys.exit(1)

print("Model pre-download complete.")
