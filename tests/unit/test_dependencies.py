import pytest
from starlette.testclient import TestClient
from api.app import create_app
from api.routers.auth import _rate_limiter
from config import Settings


@pytest.fixture
def client(tmp_path):
    # Reset the rate limiter before each test
    _rate_limiter._hits.clear()

    settings = Settings(
        raw_documents_dir=tmp_path / "raw",
        processed_documents_dir=tmp_path / "processed",
        vectorstore_dir=tmp_path / "embedded",
        paddle_cache_dir=tmp_path / "paddle",
        openai_api_key="sk-fake",
        s3_bucket_name=None,
        log_level="WARNING",
        jwt_secret_key="test-secret",
        database_url=f"sqlite:///{tmp_path}/auth.db",
    )
    with TestClient(create_app(settings)) as c:
        yield c


_register_counter = 0


def _register(client, email=None):
    global _register_counter
    if email is None:
        # Use a unique email each time to avoid rate limiter issues
        email = f"user{_register_counter}@example.com"
        _register_counter += 1
    r = client.post("/auth/register", json={"email": email, "password": "password123"})
    assert r.status_code == 200, f"Registration failed: {r.status_code} {r.json()}"
    return r.json()["access_token"]


def test_documents_requires_auth_returns_403(client):
    r = client.get("/documents")
    assert r.status_code == 403


def test_documents_rejects_invalid_token(client):
    r = client.get("/documents", headers={"Authorization": "Bearer bad.token.here"})
    assert r.status_code == 401


def test_documents_accepts_valid_token(client):
    token = _register(client)
    r = client.get("/documents", headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 200
