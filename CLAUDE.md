# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment Setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt        # full runtime (includes Paddle ~3.5 GB)
pip install -r requirements-dev.txt    # CI/test deps (no Paddle)
cp .env.example .env                   # then fill in OPENAI_API_KEY
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

# Docker
docker-compose up --build              # first build (~20 min — downloads Paddle models)
docker-compose up                      # subsequent starts

# Tests
pytest                                 # run all tests
pytest tests/unit/test_chunk.py        # run a single test file
PADDLE_AVAILABLE=1 pytest              # include Paddle-dependent tests locally

# Lint
ruff check .
ruff format .
ruff format --check .                  # check without modifying
```

## Architecture

Two entry points share the same core pipeline:

- **CLI** (`python main.py --preprocess/--index/--ask`) — for batch use
- **API server** (`python main.py --serve` or `APP_MODE=api`) — FastAPI on port 8000, used in Docker/AWS ECS

**Pipeline flow (3 stages):**

1. **`document_process/`** — PDF → frozen artifacts
   - `pipeline.py`: orchestrates `DocumentPreprocessingPipeline`
   - Services chain: `DocumentLoaderService` → `OCRService` (PaddleOCR) → `ReadingOrderService` → `LayoutDetectionService` (PP-DocLayout_plus-L) → `AssociationService` → `CroppingService`
   - Outputs `document.json`, `chunks.json`, and cropped region images into `data/processed/<document_id>/`
   - Optional: `vlm.py` generates `visual_summaries.json` with gpt-4o descriptions for tables/figures (opt-in via `USE_VLM_SUMMARIES=true`)

2. **`rag/`** — frozen artifacts → vector index → answers
   - `chunk.py`: converts `ProcessedChunk` → `ChunkRecord` (flat, embeddable)
   - `embed.py`: OpenAI `text-embedding-3-small` via `EmbeddingBackend`
   - `retrieve.py`: `DocumentRetriever` + `JsonVectorStore` (default) or `ChromaVectorStore` (opt-in via `PREFER_CHROMA=true`)
   - `qa.py`: calls OpenAI `gpt-4.1-mini`, returns `QAResponse` with answer + sources

3. **`api/`** — FastAPI HTTP layer
   - `app.py`: factory function `create_app(settings)` — always use this pattern for tests
   - `routers/`: `health` (ALB health checks), `documents` (upload/list/delete), `query` (POST with question), `showcase`
   - On startup: syncs artifacts from S3 if `S3_BUCKET_NAME` is set (ECS stateless pattern)
   - Serves static frontend from `api/static/`

**Configuration** (`config.py`): `Settings` (pydantic-settings) reads all config from env vars / `.env`. Never call `os.getenv()` — always use `Settings`. `ensure_data_dirs(settings)` is called at startup, not inside `Settings.__init__`, so `Settings()` is safe to construct in tests without side effects.

**Storage** (`storage/s3.py`): `sync_from_s3()` / `sync_to_s3()` for ECS stateless containers — processed artifacts and the vector store are persisted to/loaded from S3 on startup and after preprocessing.

**Logging** (`logging_config.py`): text format locally, JSON format in ECS (for CloudWatch Insights). Controlled via `LOG_FORMAT` env var.

**Infra** (`infra/`): Terraform for AWS — ECS Fargate (2 vCPU / 8 GB), ECR, ALB, EFS (Paddle model cache), S3, IAM.

## Testing Notes

- Tests live in `tests/unit/`
- Use the `tmp_settings` fixture (from `conftest.py`) for any test that needs a `Settings` object — it points all data dirs at a temp directory and uses a fake API key
- Paddle-dependent tests must be guarded: `@pytest.mark.skipif(not os.getenv("PADDLE_AVAILABLE"), reason="paddle not installed")` — Paddle is not in `requirements-dev.txt` and is excluded from CI
- `asyncio_mode = "auto"` is set in `pyproject.toml` so async test functions work without extra decorators
- FastAPI tests use `httpx` via the `TestClient` from `create_app(settings)`

## CI/CD

**GitHub Actions:**
- `ci.yml`: runs on every push — lint (ruff), unit tests (no Paddle), Docker build check (deps stage only)
- `deploy.yml`: runs on push to `main` — builds full Docker image, pushes to ECR, updates ECS task definition, rolls out service with health-check gating and automatic rollback

All required AWS secrets are sourced from Terraform outputs (see `infra/outputs.tf`).
