"""
Health check endpoints.

WHY these exist
---------------
AWS ECS / Application Load Balancer needs to know whether your container
is alive and ready to accept traffic. It does this by sending an HTTP
request to a known path every 30 seconds.

  GET /health — liveness probe
    "Is the process alive?" — checked by ECS. Should be FAST (no I/O,
    no external calls). Returns 200 as long as the process is running.
    If ECS gets too many non-200 responses it replaces the container.

  GET /ready — readiness probe
    "Is the app ready to serve requests?" — more thorough check.
    Returns 200 only if the OPENAI_API_KEY is set and data directories
    exist. Used to delay sending traffic to a container that is still
    initialising (e.g. syncing artifacts from S3).

Mental model: Think of /health as "am I awake?" and /ready as "am I
dressed and ready to work?"
"""
from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

router = APIRouter(tags=["health"])

# Application version — bump this in your CI/CD pipeline when deploying.
APP_VERSION = "1.0.0"


@router.get("/health")
async def health(request: Request) -> JSONResponse:
    """
    Liveness probe — always returns 200 while the process is running.
    ECS and the ALB call this every 30 seconds.
    """
    return JSONResponse({"status": "ok", "version": APP_VERSION})


@router.get("/ready")
async def ready(request: Request) -> JSONResponse:
    """
    Readiness probe — returns 200 only when the app is fully configured.

    Checks:
    - OPENAI_API_KEY is present (needed for /index and /query)
    - Data directories exist on disk

    Returns 503 (Service Unavailable) if any check fails so the load
    balancer knows not to send requests yet.
    """
    settings = request.app.state.settings
    issues: list[str] = []

    if not settings.openai_api_key:
        issues.append("OPENAI_API_KEY is not set")

    for label, path in [
        ("raw_documents_dir", settings.raw_documents_dir),
        ("processed_documents_dir", settings.processed_documents_dir),
        ("vectorstore_dir", settings.vectorstore_dir),
    ]:
        if not path.exists():
            issues.append(f"{label} does not exist: {path}")

    if issues:
        return JSONResponse(
            status_code=503,
            content={"status": "not_ready", "issues": issues},
        )

    return JSONResponse({"status": "ready", "version": APP_VERSION})
