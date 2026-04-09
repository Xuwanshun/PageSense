"""
FastAPI application factory.

WHY a factory function instead of a module-level `app = FastAPI()`
-------------------------------------------------------------------
A factory (create_app) lets you create isolated app instances with
different settings in tests, without relying on global state. This is
the standard pattern in FastAPI and Flask.

  from api.app import create_app
  app = create_app()                          # production
  app = create_app(settings=test_settings)    # in tests

Startup lifecycle
-----------------
On startup (lifespan context):
  1. Configure logging
  2. Ensure data directories exist on disk
  3. If S3_BUCKET_NAME is set, pull processed artifacts and the vector
     store from S3 so the container has the latest data

This means every time a new ECS container starts it automatically
hydrates itself from S3 — no manual file copying required.
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from api.routers import documents, health, query, showcase
from config import Settings, ensure_data_dirs
from logging_config import configure_logging

STATIC_DIR = Path(__file__).parent / "static"

logger = logging.getLogger(__name__)


def create_app(settings: Settings | None = None) -> FastAPI:
    """
    Create and configure the FastAPI application.

    Args:
        settings: Optional Settings instance. If not provided, a new
                  Settings() is constructed (reads from environment / .env).
    """
    resolved_settings = settings or Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # ── Startup ──────────────────────────────────────────────────────
        configure_logging(
            log_level=resolved_settings.log_level,
            log_format=resolved_settings.log_format,
        )
        logger.info("Starting RAG API server")

        ensure_data_dirs(resolved_settings)

        # Pull latest artifacts from S3 if a bucket is configured.
        # This is what makes the ECS container stateless: the filesystem
        # is ephemeral but the important data lives in S3.
        if resolved_settings.s3_bucket_name:
            from storage.s3 import sync_from_s3
            try:
                logger.info("Syncing artifacts from S3 bucket: %s", resolved_settings.s3_bucket_name)
                sync_from_s3(resolved_settings)
                logger.info("S3 sync complete")
            except Exception as exc:
                # Non-fatal: the container can still serve /health and accept
                # new preprocessing jobs even if S3 sync fails.
                logger.warning("S3 sync on startup failed (continuing anyway): %s", exc)

        logger.info("Server ready")
        yield
        # ── Shutdown ─────────────────────────────────────────────────────
        logger.info("Server shutting down")

    app = FastAPI(
        title="RAG Agent for PDF Reading",
        description=(
            "Upload PDFs, preprocess them with OCR, build a vector index, "
            "and ask questions against the indexed corpus."
        ),
        version="1.0.0",
        lifespan=lifespan,
    )

    # Make settings available to all request handlers via request.app.state.settings
    app.state.settings = resolved_settings

    # Register routers (each router handles a group of related endpoints)
    app.include_router(health.router)
    app.include_router(documents.router)
    app.include_router(query.router)
    app.include_router(showcase.router)

    # Serve the main UI at /
    @app.get("/", include_in_schema=False)
    async def root():
        return FileResponse(STATIC_DIR / "index.html")

    # Mount static assets (CSS, JS, etc.) — must come after route definitions
    app.mount("/", StaticFiles(directory=str(STATIC_DIR)), name="static")

    # Global exception handler — turns unhandled RuntimeErrors into JSON 500
    # responses instead of HTML error pages.
    @app.exception_handler(RuntimeError)
    async def runtime_error_handler(request, exc: RuntimeError) -> JSONResponse:
        logger.error("Unhandled error: %s", exc)
        return JSONResponse(status_code=500, content={"error": str(exc)})

    return app
