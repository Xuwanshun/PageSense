"""
Tests for the FastAPI /health and /ready endpoints.

These tests use FastAPI's TestClient, which runs the app in-process
without needing a real HTTP server. This makes them fast and reliable.

HOW TESTCLIENT WORKS
---------------------
TestClient wraps your FastAPI app and lets you make HTTP requests
against it using the same requests-like interface:

    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200

The app runs synchronously inside your test — no real ports are opened.
"""
from __future__ import annotations

from fastapi.testclient import TestClient

from api.app import create_app
from config import Settings


def _make_test_app(settings: Settings | None = None) -> TestClient:
    """Create a TestClient with the given settings (or sensible test defaults)."""
    s = settings or Settings(
        openai_api_key="sk-fake-test-key",
        log_level="WARNING",
    )
    app = create_app(settings=s)
    return TestClient(app, raise_server_exceptions=True)


def test_health_returns_200():
    """GET /health must return 200 regardless of configuration."""
    client = _make_test_app()
    response = client.get("/health")
    assert response.status_code == 200


def test_health_returns_ok_status():
    """GET /health body must include status=ok."""
    client = _make_test_app()
    body = client.get("/health").json()
    assert body["status"] == "ok"


def test_health_returns_version():
    """GET /health should include a version field."""
    client = _make_test_app()
    body = client.get("/health").json()
    assert "version" in body


def test_ready_returns_503_when_api_key_missing(tmp_path):
    """
    GET /ready should return 503 when OPENAI_API_KEY is not set.

    This is the readiness probe — ECS will not send traffic until it
    returns 200. We want it to fail clearly, not silently.
    """
    settings = Settings(
        openai_api_key=None,  # key is missing
        raw_documents_dir=tmp_path / "raw",
        processed_documents_dir=tmp_path / "processed",
        vectorstore_dir=tmp_path / "embedded",
        paddle_cache_dir=tmp_path / "paddle",
    )
    # Create the directories so that check passes
    settings.raw_documents_dir.mkdir(parents=True)
    settings.processed_documents_dir.mkdir(parents=True)
    settings.vectorstore_dir.mkdir(parents=True)

    client = _make_test_app(settings=settings)
    response = client.get("/ready")
    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "not_ready"
    assert any("OPENAI_API_KEY" in issue for issue in body["issues"])


def test_ready_returns_200_when_fully_configured(tmp_path):
    """GET /ready returns 200 when API key is set and data dirs exist."""
    settings = Settings(
        openai_api_key="sk-fake-test-key",
        raw_documents_dir=tmp_path / "raw",
        processed_documents_dir=tmp_path / "processed",
        vectorstore_dir=tmp_path / "embedded",
        paddle_cache_dir=tmp_path / "paddle",
    )
    # Create the directories
    settings.raw_documents_dir.mkdir(parents=True)
    settings.processed_documents_dir.mkdir(parents=True)
    settings.vectorstore_dir.mkdir(parents=True)

    client = _make_test_app(settings=settings)
    response = client.get("/ready")
    assert response.status_code == 200
    assert response.json()["status"] == "ready"
