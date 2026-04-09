"""
Application configuration.

All settings are read from environment variables (or a .env file).
pydantic-settings handles this automatically — you never need to call
os.getenv() manually anywhere in the codebase.

How to use:
    from config import Settings
    settings = Settings()  # reads env vars + .env file automatically
    print(settings.openai_api_key)

To override a value locally, either:
    - Set it in your shell: export OPENAI_API_KEY=sk-...
    - Add it to your .env file: OPENAI_API_KEY=sk-...
    - Pass it directly in tests: Settings(openai_api_key="fake-key")
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Central configuration object.

    Every field corresponds to one environment variable of the same name
    (uppercased). pydantic-settings reads them automatically.
    """

    model_config = SettingsConfigDict(
        # Look for a .env file in the current working directory.
        # This is only used locally — in Docker/AWS real env vars are injected.
        env_file=".env",
        env_file_encoding="utf-8",
        # Don't crash on extra env vars that appear in the environment.
        extra="ignore",
    )

    # ── Data directories ──────────────────────────────────────────────────────
    # These defaults work locally. In Docker they are overridden via ENV
    # instructions in the Dockerfile (set to /app/data/...).
    raw_documents_dir: Path = Path("data/raw")
    processed_documents_dir: Path = Path("data/processed")
    vectorstore_dir: Path = Path("data/embedded")

    # Where PaddleOCR/PaddleX downloads and caches its ML models.
    # In Docker this points to a mounted volume so models persist across
    # container restarts and do not need re-downloading each time.
    paddle_cache_dir: Path = Path("data/.paddlex_cache")

    # ── OpenAI ────────────────────────────────────────────────────────────────
    # Required for --index and --ask (embedding + generation).
    # In AWS this is injected from Secrets Manager — never hardcode it.
    openai_api_key: str | None = None
    # Leave empty to use the default OpenAI endpoint.
    # Set this to point at a compatible alternative (e.g. Azure OpenAI, LM Studio).
    openai_base_url: str | None = None
    openai_model: str = "gpt-4.1-mini"
    embedding_model: str = "text-embedding-3-small"

    # ── Pipeline tuning ───────────────────────────────────────────────────────
    preprocess_chunk_size: int = 1800
    preprocess_chunk_overlap: int = 200
    pdf_render_scale: float = 3.0
    default_top_k: int = 4
    # Set to True if you have chromadb installed and prefer it over the
    # built-in JSON vector store.
    prefer_chroma: bool = False

    # ── Logging ───────────────────────────────────────────────────────────────
    # LOG_LEVEL: standard Python log level (DEBUG, INFO, WARNING, ERROR)
    log_level: str = "INFO"
    # LOG_FORMAT: "text" for human-readable (local dev), "json" for CloudWatch
    log_format: Literal["text", "json"] = "text"

    # ── Application mode ─────────────────────────────────────────────────────
    # "cli"  → use the argparse CLI (python main.py --preprocess ...)
    # "api"  → start the FastAPI server (python main.py --serve)
    app_mode: Literal["cli", "api"] = "cli"

    # ── VLM visual summaries ──────────────────────────────────────────────────
    # When enabled, each cropped table/figure is sent to the vision model and
    # the returned description replaces the OCR-text fallback in
    # visual_summaries.json. This is the recommended setting for production
    # because figures and charts carry zero useful OCR text.
    #
    # Cost note: one API call per detected table/figure region per document.
    # A 20-page report with 5 tables and 3 figures = 8 vision calls.
    use_vlm_summaries: bool = False
    vlm_model: str = "gpt-4o"

    # ── AWS / S3 ─────────────────────────────────────────────────────────────
    # Set S3_BUCKET_NAME when running on AWS to persist processed artifacts
    # and the vector store across container restarts (ECS tasks are ephemeral).
    # Leave empty for local development where the filesystem is persistent.
    s3_bucket_name: str | None = None
    aws_region: str = "us-east-1"


def ensure_data_dirs(settings: Settings) -> None:
    """
    Create data directories if they do not already exist.

    This is called explicitly at application startup (main.py, api/app.py),
    NOT inside Settings itself. Keeping side effects out of the config object
    makes Settings safe to instantiate in tests without touching the filesystem.
    """
    settings.raw_documents_dir.mkdir(parents=True, exist_ok=True)
    settings.processed_documents_dir.mkdir(parents=True, exist_ok=True)
    settings.vectorstore_dir.mkdir(parents=True, exist_ok=True)
    settings.paddle_cache_dir.mkdir(parents=True, exist_ok=True)
    (settings.paddle_cache_dir / "temp").mkdir(parents=True, exist_ok=True)
