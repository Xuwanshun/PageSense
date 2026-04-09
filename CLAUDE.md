# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment Setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then fill in OPENAI_API_KEY
```

Required env vars: `OPENAI_API_KEY`. Optional: `OPENAI_BASE_URL`.

## Common Commands

```bash
# CLI pipeline
python main.py --preprocess            # OCR + freeze artifacts from data/raw/
python main.py --index                 # build vector index from frozen artifacts
python main.py --ask "your question"   # query against the index

# API server (local)
python main.py --serve                 # starts FastAPI on port 8000
# or: APP_MODE=api python main.py --serve

# Docker
docker-compose up --build              # first build (~20 min — downloads Paddle models)
docker-compose up                      # subsequent starts

# Tests
pytest                                 # run all tests
pytest tests/unit/test_chunk.py        # run a single test file

# Lint
ruff check .
ruff format .
```

## Architecture

Two entry points share the same core pipeline:

- **CLI** (`python main.py --preprocess/--index/--ask`) — for batch use
- **API server** (`python main.py --serve` or `APP_MODE=api`) — FastAPI on port 8000, used in Docker/AWS ECS

**Pipeline flow (3 stages):**

1. **`document_Process/`** — PDF → frozen artifacts
   - `pipeline.py`: orchestrates `DocumentPreprocessingPipeline`
   - Services: `DocumentLoaderService` → `OCRService` (PaddleOCR) → `ReadingOrderService` → `LayoutDetectionService` (PP-DocLayout_plus-L) → `AssociationService` → `CroppingService`
   - Outputs `document.json` + `chunks.json` into `data/processed/<document_id>/`

2. **`rag/`** — frozen artifacts → vector index → answers
   - `chunk.py`: converts `ProcessedChunk` → `ChunkRecord`
   - `embed.py`: OpenAI `text-embedding-3-small` via `EmbeddingBackend`
   - `retrieve.py`: `DocumentRetriever` + `JsonVectorStore` (default) or `ChromaVectorStore` (opt-in via `PREFER_CHROMA=true`) + `answer_question()` calls OpenAI `gpt-4.1-mini`

3. **`api/`** — FastAPI HTTP layer
   - `app.py`: factory function `create_app(settings)` — use this pattern for tests
   - `routers/`: `health`, `documents`, `query`, `showcase`
   - On startup: syncs artifacts from S3 if `S3_BUCKET_NAME` is set (ECS stateless pattern)
   - Serves static frontend from `api/static/`

**Configuration** (`config.py`): `Settings` (pydantic-settings) reads all config from env vars / `.env`. Never call `os.getenv()` — always use `Settings`. `ensure_data_dirs(settings)` is called at startup, not inside `Settings.__init__`, so `Settings()` is safe to construct in tests without side effects.

**Storage** (`storage/`): S3 sync for ECS deployments — processed artifacts and the vector store are persisted to/loaded from S3 so containers are stateless.

**Infra** (`infra/`): Terraform for AWS — ECS Fargate, ECR, ALB, EFS, S3, IAM.

## Testing Notes

- Tests live in `tests/unit/`
- Use the `tmp_settings` fixture (from `conftest.py`) for any test that needs a `Settings` object — it points all data dirs at a temp directory
- Paddle-dependent tests must be guarded: `@pytest.mark.skipif(not os.getenv("PADDLE_AVAILABLE"), reason="paddle not installed")` — Paddle is not in `requirements-dev.txt`
- `asyncio_mode = "auto"` is set in `pyproject.toml` so async test functions work without extra decorators
