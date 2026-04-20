import pytest
from starlette.testclient import TestClient
from api.app import create_app
from config import Settings


@pytest.fixture
def client(tmp_path):
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


def test_register_success(client):
    r = client.post("/auth/register", json={"email": "a@example.com", "password": "password123"})
    assert r.status_code == 200
    data = r.json()
    assert "access_token" in data
    assert data["token_type"] == "bearer"


def test_register_duplicate_email_returns_409(client):
    client.post("/auth/register", json={"email": "a@example.com", "password": "password123"})
    r = client.post("/auth/register", json={"email": "a@example.com", "password": "otherpass1"})
    assert r.status_code == 409


def test_register_short_password_returns_422(client):
    r = client.post("/auth/register", json={"email": "b@example.com", "password": "short"})
    assert r.status_code == 422


def test_login_success(client):
    client.post("/auth/register", json={"email": "c@example.com", "password": "password123"})
    r = client.post("/auth/login", json={"email": "c@example.com", "password": "password123"})
    assert r.status_code == 200
    assert "access_token" in r.json()


def test_login_wrong_password_returns_401(client):
    client.post("/auth/register", json={"email": "d@example.com", "password": "password123"})
    r = client.post("/auth/login", json={"email": "d@example.com", "password": "wrongpass12"})
    assert r.status_code == 401


def test_login_unknown_email_returns_401(client):
    r = client.post("/auth/login", json={"email": "nobody@example.com", "password": "password123"})
    assert r.status_code == 401


def test_refresh_token_sets_cookie_and_returns_new_access_token(client):
    r = client.post("/auth/register", json={"email": "e@example.com", "password": "password123"})
    assert "refresh_token" in client.cookies
    r2 = client.post("/auth/refresh")
    assert r2.status_code == 200
    assert "access_token" in r2.json()


def test_logout_invalidates_refresh_token(client):
    client.post("/auth/register", json={"email": "f@example.com", "password": "password123"})
    client.post("/auth/logout")
    r = client.post("/auth/refresh")
    assert r.status_code == 401
