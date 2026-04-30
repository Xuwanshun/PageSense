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
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Data directories ──────────────────────────────────────────────────────
    raw_documents_dir: Path = Path("data/raw")
    processed_documents_dir: Path = Path("data/processed")
    vectorstore_dir: Path = Path("data/embedded")
    paddle_cache_dir: Path = Path("data/.paddlex_cache")

    # ── OpenAI ────────────────────────────────────────────────────────────────
    openai_api_key: str | None = None
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
    log_level: str = "INFO"
    log_format: Literal["text", "json"] = "text"

    # ── VLM visual summaries ──────────────────────────────────────────────────
    # When enabled, each cropped table/figure is sent to the vision model.
    # Cost note: one API call per detected table/figure region per document.
    use_vlm_summaries: bool = False
    vlm_model: str = "gpt-4o"

    # ── Document intelligence ─────────────────────────────────────────────────
    use_document_intelligence: bool = False
    use_adaptive_chunking: bool = False
    descriptor_model: str = "gpt-4.1-mini"

    # ── Fast mode ─────────────────────────────────────────────────────────────
    # Skip all VLM and LLM calls (stages 3 and 5). Useful for offline indexing.
    fast_mode: bool = False

    # ── Async concurrency limits ──────────────────────────────────────────────
    vlm_concurrency_limit: int = 4
    llm_concurrency_limit: int = 8

    # ── Reading order ─────────────────────────────────────────────────────────
    reading_order_line_bucket_px: int = 18

    # ── Per-stage cache ───────────────────────────────────────────────────────
    stage_cache_enabled: bool = True

    # ── S3 (optional remote artifact storage) ────────────────────────────────
    s3_bucket_name: str | None = None


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
