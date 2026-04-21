# Secure Sign-In Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add multi-user authentication (email/password + Google/GitHub OAuth) with per-user document isolation to the PDF RAG web app.

**Architecture:** JWT access tokens (15 min, stored in JS memory) plus `httponly` refresh tokens (7 days, cookie). A `get_current_user` FastAPI dependency guards all document/query routes. Per-user isolation is achieved by a `user_scoped_settings()` helper that scopes the `raw_documents_dir`, `processed_documents_dir`, and `vectorstore_dir` paths under `{user_id}/` — the pipeline code requires no changes.

**Tech Stack:** PyJWT, passlib[bcrypt], authlib (OAuth), SQLAlchemy Core (SQLite/PostgreSQL), Starlette SessionMiddleware

---

## File Map

**Create:**
- `db/__init__.py` — package marker
- `db/engine.py` — `make_engine()`, `create_tables()`
- `db/models.py` — SQLAlchemy `users` and `refresh_tokens` Table objects
- `api/auth/__init__.py` — package marker
- `api/auth/passwords.py` — bcrypt hash/verify
- `api/auth/tokens.py` — JWT create/verify, refresh token generation
- `api/auth/rate_limit.py` — in-memory per-IP rate limiter
- `api/auth/oauth.py` — authlib OAuth client setup
- `api/routers/auth.py` — `/auth/*` endpoints
- `api/dependencies.py` — `get_current_user` FastAPI dependency
- `api/static/login.html` — login/register UI page
- `api/static/login.js` — login page frontend logic
- `tests/unit/test_db.py`
- `tests/unit/test_auth_passwords.py`
- `tests/unit/test_auth_tokens.py`
- `tests/unit/test_auth_rate_limit.py`
- `tests/unit/test_auth_router.py`
- `tests/unit/test_dependencies.py`
- `tests/unit/test_user_scoping.py`

**Modify:**
- `requirements.txt` — add PyJWT, passlib, authlib, sqlalchemy
- `.env.example` — add auth env vars
- `config.py` — add auth settings + `user_scoped_settings()`
- `tests/conftest.py` — add `jwt_secret_key` + `database_url` to `tmp_settings`
- `api/app.py` — include auth router, `/login` route, SessionMiddleware, DB engine on state
- `api/routers/documents.py` — add `get_current_user`, scope all endpoints by user
- `api/routers/query.py` — add `get_current_user`, scope query by user
- `api/static/app.js` — token lifecycle, `authedFetch`, redirect to `/login`
- `api/static/sidebar.js` — accept and use `authedFetch`
- `api/static/query.js` — accept and use `authedFetch`
- `api/static/index.html` — add logout button

---

### Task 1: Dependencies, config, and test fixtures

**Files:**
- Modify: `requirements.txt`
- Modify: `.env.example`
- Modify: `config.py`
- Modify: `tests/conftest.py`

- [ ] **Step 1: Add new dependencies to requirements.txt**

After the `openai>=2.30,<3` line, add:

```
# ── Auth ─────────────────────────────────────────────────────────────────────
PyJWT>=2.8,<3
passlib[bcrypt]>=1.7,<2
authlib>=1.3,<2
sqlalchemy>=2.0,<3
```

- [ ] **Step 2: Add auth env vars to .env.example**

Append at the end of `.env.example`:

```
# ── Auth ─────────────────────────────────────────────────────────────────────
# Required: 32-byte random hex. Generate: python -c "import os; print(os.urandom(32).hex())"
JWT_SECRET_KEY=
# SQLite locally; use postgresql://user:pass@host/db on AWS RDS
DATABASE_URL=sqlite:///data/auth.db
ACCESS_TOKEN_EXPIRE_MINUTES=15
REFRESH_TOKEN_EXPIRE_DAYS=7
# Set true on HTTPS deployments to add Secure flag to the refresh cookie
HTTPS_ONLY=false
# Google OAuth (leave empty to disable Google sign-in)
GOOGLE_CLIENT_ID=
GOOGLE_CLIENT_SECRET=
# GitHub OAuth (leave empty to disable GitHub sign-in)
GITHUB_CLIENT_ID=
GITHUB_CLIENT_SECRET=
```

- [ ] **Step 3: Add auth fields to Settings in config.py**

Add this block inside the `Settings` class, after the `use_faithfulness_check` field and before `s3_bucket_name`:

```python
    # ── Auth ──────────────────────────────────────────────────────────────────
    jwt_secret_key: str | None = None
    jwt_algorithm: str = "HS256"
    database_url: str = "sqlite:///data/auth.db"
    access_token_expire_minutes: int = 15
    refresh_token_expire_days: int = 7
    https_only: bool = False
    google_client_id: str | None = None
    google_client_secret: str | None = None
    github_client_id: str | None = None
    github_client_secret: str | None = None
```

- [ ] **Step 4: Add user_scoped_settings() to config.py**

Add after the `ensure_data_dirs` function:

```python
def user_scoped_settings(settings: Settings, user_id: str) -> Settings:
    """Return a copy of settings with all data dirs scoped under user_id/."""
    return settings.model_copy(update={
        "raw_documents_dir": settings.raw_documents_dir / user_id,
        "processed_documents_dir": settings.processed_documents_dir / user_id,
        "vectorstore_dir": settings.vectorstore_dir / user_id,
    })
```

- [ ] **Step 5: Update tmp_settings fixture in tests/conftest.py**

Replace the `return Settings(...)` call inside `tmp_settings`:

```python
    return Settings(
        raw_documents_dir=tmp_path / "raw",
        processed_documents_dir=tmp_path / "processed",
        vectorstore_dir=tmp_path / "embedded",
        paddle_cache_dir=tmp_path / "paddle",
        openai_api_key="sk-test-fake-key-for-unit-tests",
        openai_base_url=None,
        s3_bucket_name=None,
        log_level="WARNING",
        jwt_secret_key="test-secret-key-for-unit-tests",
        database_url=f"sqlite:///{tmp_path}/auth.db",
    )
```

- [ ] **Step 6: Install new dependencies**

```bash
pip install PyJWT passlib[bcrypt] authlib sqlalchemy
```

Expected: installs without errors.

- [ ] **Step 7: Verify existing tests still pass**

```bash
pytest tests/ -x -q
```

Expected: all tests pass.

- [ ] **Step 8: Commit**

```bash
git add requirements.txt .env.example config.py tests/conftest.py
git commit -m "feat(auth): add auth dependencies, config fields, and user_scoped_settings"
```

---

### Task 2: Database models and engine

**Files:**
- Create: `db/__init__.py`
- Create: `db/engine.py`
- Create: `db/models.py`
- Create: `tests/unit/test_db.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_db.py`:

```python
import pytest
from datetime import datetime, timezone
from sqlalchemy import insert, select
from sqlalchemy.exc import IntegrityError
from db.engine import create_tables, make_engine
from db.models import refresh_tokens, users


@pytest.fixture
def engine(tmp_path):
    e = make_engine(f"sqlite:///{tmp_path}/test.db")
    create_tables(e)
    return e


def test_insert_and_retrieve_user(engine):
    with engine.connect() as conn:
        conn.execute(insert(users).values(
            id="u1", email="a@example.com", hashed_password="hash",
            created_at=datetime.now(timezone.utc), is_active=True,
        ))
        conn.commit()
        row = conn.execute(select(users).where(users.c.id == "u1")).first()
    assert row.email == "a@example.com"


def test_users_email_unique_constraint(engine):
    with engine.connect() as conn:
        conn.execute(insert(users).values(
            id="u2", email="dup@example.com", hashed_password=None,
            created_at=datetime.now(timezone.utc), is_active=True,
        ))
        conn.commit()
    with pytest.raises(IntegrityError):
        with engine.connect() as conn:
            conn.execute(insert(users).values(
                id="u3", email="dup@example.com", hashed_password=None,
                created_at=datetime.now(timezone.utc), is_active=True,
            ))
            conn.commit()


def test_insert_and_retrieve_refresh_token(engine):
    with engine.connect() as conn:
        conn.execute(insert(users).values(
            id="u4", email="b@example.com", hashed_password=None,
            created_at=datetime.now(timezone.utc), is_active=True,
        ))
        conn.execute(insert(refresh_tokens).values(
            token_hash="abc123", user_id="u4",
            expires_at=datetime.now(timezone.utc),
        ))
        conn.commit()
        row = conn.execute(
            select(refresh_tokens).where(refresh_tokens.c.token_hash == "abc123")
        ).first()
    assert row.user_id == "u4"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/unit/test_db.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'db'`

- [ ] **Step 3: Create db/__init__.py**

Empty file.

- [ ] **Step 4: Create db/models.py**

```python
from sqlalchemy import Boolean, Column, DateTime, MetaData, String, Table

metadata = MetaData()

users = Table(
    "users",
    metadata,
    Column("id", String(36), primary_key=True),
    Column("email", String, unique=True, nullable=False),
    Column("hashed_password", String, nullable=True),
    Column("oauth_provider", String, nullable=True),
    Column("oauth_sub", String, nullable=True),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("is_active", Boolean, nullable=False),
)

refresh_tokens = Table(
    "refresh_tokens",
    metadata,
    Column("token_hash", String, primary_key=True),
    Column("user_id", String(36), nullable=False),
    Column("expires_at", DateTime(timezone=True), nullable=False),
)
```

- [ ] **Step 5: Create db/engine.py**

```python
from sqlalchemy import Engine, create_engine

from db.models import metadata


def make_engine(database_url: str) -> Engine:
    kwargs: dict = {}
    if database_url.startswith("sqlite"):
        kwargs["connect_args"] = {"check_same_thread": False}
    return create_engine(database_url, **kwargs)


def create_tables(engine: Engine) -> None:
    metadata.create_all(engine)
```

- [ ] **Step 6: Run tests to verify they pass**

```bash
pytest tests/unit/test_db.py -v
```

Expected: all 3 tests PASS.

- [ ] **Step 7: Commit**

```bash
git add db/ tests/unit/test_db.py
git commit -m "feat(auth): add SQLAlchemy database models and engine"
```

---

### Task 3: Password hashing utilities

**Files:**
- Create: `api/auth/__init__.py`
- Create: `api/auth/passwords.py`
- Create: `tests/unit/test_auth_passwords.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_auth_passwords.py`:

```python
from api.auth.passwords import hash_password, verify_password


def test_hash_returns_nonempty_string():
    assert len(hash_password("mysecret")) > 0


def test_verify_correct_password():
    h = hash_password("correct")
    assert verify_password("correct", h) is True


def test_verify_wrong_password():
    h = hash_password("correct")
    assert verify_password("wrong", h) is False


def test_two_hashes_of_same_password_differ():
    assert hash_password("same") != hash_password("same")
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/unit/test_auth_passwords.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'api.auth'`

- [ ] **Step 3: Create api/auth/__init__.py**

Empty file.

- [ ] **Step 4: Create api/auth/passwords.py**

```python
from passlib.context import CryptContext

_ctx = CryptContext(schemes=["bcrypt"], deprecated="auto", bcrypt__rounds=12)


def hash_password(password: str) -> str:
    return _ctx.hash(password)


def verify_password(plain: str, hashed: str) -> bool:
    return _ctx.verify(plain, hashed)
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
pytest tests/unit/test_auth_passwords.py -v
```

Expected: all 4 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add api/auth/ tests/unit/test_auth_passwords.py
git commit -m "feat(auth): add bcrypt password hashing utilities"
```

---

### Task 4: JWT token utilities

**Files:**
- Create: `api/auth/tokens.py`
- Create: `tests/unit/test_auth_tokens.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_auth_tokens.py`:

```python
import time
import pytest
import jwt as pyjwt
from api.auth.tokens import create_access_token, create_refresh_token, verify_access_token

SECRET = "test-secret"
ALGO = "HS256"


def test_access_token_roundtrip():
    token = create_access_token("user-123", SECRET, ALGO, expire_minutes=15)
    assert verify_access_token(token, SECRET, ALGO) == "user-123"


def test_access_token_wrong_secret_rejected():
    token = create_access_token("user-123", SECRET, ALGO, expire_minutes=15)
    with pytest.raises(Exception):
        verify_access_token(token, "wrong-secret", ALGO)


def test_access_token_expired_rejected():
    # expire_minutes=0 creates a token that expires immediately
    token = create_access_token("user-123", SECRET, ALGO, expire_minutes=0)
    time.sleep(1)
    with pytest.raises(Exception):
        verify_access_token(token, SECRET, ALGO)


def test_refresh_token_raw_is_64_hex_chars():
    raw, _ = create_refresh_token()
    assert len(raw) == 64


def test_refresh_token_hash_is_64_hex_chars():
    _, token_hash = create_refresh_token()
    assert len(token_hash) == 64


def test_refresh_token_raw_and_hash_differ():
    raw, token_hash = create_refresh_token()
    assert raw != token_hash


def test_two_refresh_tokens_are_unique():
    raw1, _ = create_refresh_token()
    raw2, _ = create_refresh_token()
    assert raw1 != raw2
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/unit/test_auth_tokens.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'api.auth.tokens'`

- [ ] **Step 3: Create api/auth/tokens.py**

```python
import hashlib
import os
from datetime import datetime, timedelta, timezone

import jwt


def create_access_token(
    user_id: str, secret_key: str, algorithm: str, expire_minutes: int
) -> str:
    expire = datetime.now(timezone.utc) + timedelta(minutes=expire_minutes)
    return jwt.encode(
        {"sub": user_id, "exp": expire, "type": "access"},
        secret_key,
        algorithm=algorithm,
    )


def verify_access_token(token: str, secret_key: str, algorithm: str) -> str:
    payload = jwt.decode(token, secret_key, algorithms=[algorithm])
    if payload.get("type") != "access":
        raise jwt.InvalidTokenError("wrong token type")
    return payload["sub"]


def create_refresh_token() -> tuple[str, str]:
    raw = os.urandom(32).hex()
    token_hash = hashlib.sha256(raw.encode()).hexdigest()
    return raw, token_hash
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/unit/test_auth_tokens.py -v
```

Expected: all 7 tests PASS. (`test_access_token_expired_rejected` takes ~1 s.)

- [ ] **Step 5: Commit**

```bash
git add api/auth/tokens.py tests/unit/test_auth_tokens.py
git commit -m "feat(auth): add JWT token utilities"
```

---

### Task 5: In-memory rate limiter

**Files:**
- Create: `api/auth/rate_limit.py`
- Create: `tests/unit/test_auth_rate_limit.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_auth_rate_limit.py`:

```python
from api.auth.rate_limit import RateLimiter


def test_allows_requests_under_limit():
    rl = RateLimiter(limit=3, window=60)
    assert rl.is_allowed("1.2.3.4") is True
    assert rl.is_allowed("1.2.3.4") is True
    assert rl.is_allowed("1.2.3.4") is True


def test_blocks_on_limit_exceeded():
    rl = RateLimiter(limit=3, window=60)
    for _ in range(3):
        rl.is_allowed("1.2.3.4")
    assert rl.is_allowed("1.2.3.4") is False


def test_different_ips_are_independent():
    rl = RateLimiter(limit=1, window=60)
    rl.is_allowed("10.0.0.1")
    assert rl.is_allowed("10.0.0.1") is False
    assert rl.is_allowed("10.0.0.2") is True
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/unit/test_auth_rate_limit.py -v
```

Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Create api/auth/rate_limit.py**

```python
import time
from collections import defaultdict


class RateLimiter:
    def __init__(self, limit: int = 10, window: int = 60) -> None:
        self._limit = limit
        self._window = window
        self._hits: dict[str, list[float]] = defaultdict(list)

    def is_allowed(self, ip: str) -> bool:
        now = time.monotonic()
        self._hits[ip] = [t for t in self._hits[ip] if now - t < self._window]
        self._hits[ip].append(now)
        return len(self._hits[ip]) <= self._limit
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/unit/test_auth_rate_limit.py -v
```

Expected: all 3 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add api/auth/rate_limit.py tests/unit/test_auth_rate_limit.py
git commit -m "feat(auth): add in-memory per-IP rate limiter"
```

---

### Task 6: Auth router (email/password) and app wiring

**Files:**
- Create: `api/routers/auth.py`
- Modify: `api/app.py`
- Create: `tests/unit/test_auth_router.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_auth_router.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/unit/test_auth_router.py -v
```

Expected: FAIL — `/auth/register` returns 404.

- [ ] **Step 3: Create api/routers/auth.py**

```python
from __future__ import annotations

import hashlib
import uuid
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Cookie, HTTPException, Request, Response, status
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import delete, insert, select

from api.auth.passwords import hash_password, verify_password
from api.auth.rate_limit import RateLimiter
from api.auth.tokens import create_access_token, create_refresh_token
from db.models import refresh_tokens, users

router = APIRouter(prefix="/auth", tags=["auth"])
_rate_limiter = RateLimiter(limit=10, window=60)


class RegisterRequest(BaseModel):
    email: str = Field(..., min_length=3)
    password: str = Field(..., min_length=8)

    @field_validator("email")
    @classmethod
    def normalise_email(cls, v: str) -> str:
        return v.lower().strip()


class LoginRequest(BaseModel):
    email: str
    password: str

    @field_validator("email")
    @classmethod
    def normalise_email(cls, v: str) -> str:
        return v.lower().strip()


def _set_refresh_cookie(response: Response, raw_refresh: str, settings) -> None:
    response.set_cookie(
        key="refresh_token",
        value=raw_refresh,
        httponly=True,
        samesite="lax",
        secure=settings.https_only,
        max_age=settings.refresh_token_expire_days * 86400,
    )


@router.post("/register")
async def register(body: RegisterRequest, request: Request, response: Response) -> dict:
    ip = request.client.host if request.client else "unknown"
    if not _rate_limiter.is_allowed(ip):
        raise HTTPException(status_code=429, detail="Too many requests")

    settings = request.app.state.settings
    engine = request.app.state.db_engine
    now = datetime.now(timezone.utc)

    with engine.connect() as conn:
        if conn.execute(select(users).where(users.c.email == body.email)).first():
            raise HTTPException(status_code=409, detail="Email already registered")

        user_id = str(uuid.uuid4())
        conn.execute(insert(users).values(
            id=user_id, email=body.email,
            hashed_password=hash_password(body.password),
            created_at=now, is_active=True,
        ))
        access_token = create_access_token(
            user_id, settings.jwt_secret_key, settings.jwt_algorithm,
            settings.access_token_expire_minutes,
        )
        raw_refresh, token_hash = create_refresh_token()
        conn.execute(insert(refresh_tokens).values(
            token_hash=token_hash, user_id=user_id,
            expires_at=now + timedelta(days=settings.refresh_token_expire_days),
        ))
        conn.commit()

    _set_refresh_cookie(response, raw_refresh, settings)
    return {"access_token": access_token, "token_type": "bearer",
            "expires_in": settings.access_token_expire_minutes * 60}


@router.post("/login")
async def login(body: LoginRequest, request: Request, response: Response) -> dict:
    ip = request.client.host if request.client else "unknown"
    if not _rate_limiter.is_allowed(ip):
        raise HTTPException(status_code=429, detail="Too many requests")

    settings = request.app.state.settings
    engine = request.app.state.db_engine
    now = datetime.now(timezone.utc)

    with engine.connect() as conn:
        row = conn.execute(select(users).where(users.c.email == body.email)).first()

    if not row or not row.is_active or not row.hashed_password:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")
    if not verify_password(body.password, row.hashed_password):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")

    access_token = create_access_token(
        row.id, settings.jwt_secret_key, settings.jwt_algorithm,
        settings.access_token_expire_minutes,
    )
    raw_refresh, token_hash = create_refresh_token()

    with engine.connect() as conn:
        conn.execute(insert(refresh_tokens).values(
            token_hash=token_hash, user_id=row.id,
            expires_at=now + timedelta(days=settings.refresh_token_expire_days),
        ))
        conn.commit()

    _set_refresh_cookie(response, raw_refresh, settings)
    return {"access_token": access_token, "token_type": "bearer",
            "expires_in": settings.access_token_expire_minutes * 60}


@router.post("/refresh")
async def refresh_token(
    request: Request, response: Response,
    refresh_token: str | None = Cookie(default=None),
) -> dict:
    if not refresh_token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="No refresh token")

    token_hash = hashlib.sha256(refresh_token.encode()).hexdigest()
    settings = request.app.state.settings
    engine = request.app.state.db_engine
    now = datetime.now(timezone.utc)

    with engine.connect() as conn:
        row = conn.execute(
            select(refresh_tokens).where(refresh_tokens.c.token_hash == token_hash)
        ).first()
        if row is None:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid refresh token")

        expires_at = row.expires_at
        if expires_at.tzinfo is None:
            expires_at = expires_at.replace(tzinfo=timezone.utc)
        if expires_at < now:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Refresh token expired")

        user_id = row.user_id
        conn.execute(delete(refresh_tokens).where(refresh_tokens.c.token_hash == token_hash))

        access_token = create_access_token(
            user_id, settings.jwt_secret_key, settings.jwt_algorithm,
            settings.access_token_expire_minutes,
        )
        raw_refresh, new_hash = create_refresh_token()
        conn.execute(insert(refresh_tokens).values(
            token_hash=new_hash, user_id=user_id,
            expires_at=now + timedelta(days=settings.refresh_token_expire_days),
        ))
        conn.commit()

    _set_refresh_cookie(response, raw_refresh, settings)
    return {"access_token": access_token, "token_type": "bearer",
            "expires_in": settings.access_token_expire_minutes * 60}


@router.post("/logout")
async def logout(
    request: Request, response: Response,
    refresh_token: str | None = Cookie(default=None),
) -> dict:
    if refresh_token:
        token_hash = hashlib.sha256(refresh_token.encode()).hexdigest()
        engine = request.app.state.db_engine
        with engine.connect() as conn:
            conn.execute(delete(refresh_tokens).where(refresh_tokens.c.token_hash == token_hash))
            conn.commit()
    response.delete_cookie("refresh_token")
    return {"ok": True}
```

- [ ] **Step 4: Update api/app.py**

Add to the imports at the top (replace the existing responses import line):

```python
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from starlette.middleware.sessions import SessionMiddleware
```

Replace the routers import:

```python
from api.routers import auth, documents, health, query
```

In the `lifespan` function, after `ensure_data_dirs(resolved_settings)`, add:

```python
        from db.engine import create_tables, make_engine
        engine = make_engine(resolved_settings.database_url)
        create_tables(engine)
        app.state.db_engine = engine
        logger.info("Database ready: %s", resolved_settings.database_url)
```

After the `app = FastAPI(...)` block and before `app.state.settings = resolved_settings`, add:

```python
    if resolved_settings.jwt_secret_key:
        app.add_middleware(SessionMiddleware, secret_key=resolved_settings.jwt_secret_key)
```

Add the auth router alongside the existing router includes:

```python
    app.include_router(auth.router)
```

Add the `/login` route before the `GET /` route:

```python
    @app.get("/login", include_in_schema=False)
    async def login_page():
        return FileResponse(STATIC_DIR / "login.html")
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
pytest tests/unit/test_auth_router.py -v
```

Expected: all 8 tests PASS.

- [ ] **Step 6: Run full test suite**

```bash
pytest tests/ -x -q
```

Expected: all tests pass.

- [ ] **Step 7: Commit**

```bash
git add api/routers/auth.py api/app.py tests/unit/test_auth_router.py
git commit -m "feat(auth): add email/password auth router and wire into app"
```

---

### Task 7: get_current_user dependency and protect document routes

**Files:**
- Create: `api/dependencies.py`
- Modify: `api/routers/documents.py`
- Create: `tests/unit/test_dependencies.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_dependencies.py`:

```python
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


def _register(client, email="user@example.com"):
    r = client.post("/auth/register", json={"email": email, "password": "password123"})
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/unit/test_dependencies.py -v
```

Expected: `test_documents_requires_auth_returns_403` FAILS — returns 200 (endpoint not yet protected).

- [ ] **Step 3: Create api/dependencies.py**

```python
from __future__ import annotations

import jwt
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import select

from api.auth.tokens import verify_access_token
from db.models import users

_bearer = HTTPBearer()


async def get_current_user(
    request: Request,
    credentials: HTTPAuthorizationCredentials = Depends(_bearer),
) -> dict:
    settings = request.app.state.settings
    if not settings.jwt_secret_key:
        raise HTTPException(status_code=500, detail="Auth not configured: JWT_SECRET_KEY missing")
    try:
        user_id = verify_access_token(
            credentials.credentials, settings.jwt_secret_key, settings.jwt_algorithm
        )
    except jwt.PyJWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired token"
        )
    engine = request.app.state.db_engine
    with engine.connect() as conn:
        row = conn.execute(select(users).where(users.c.id == user_id)).first()
    if row is None or not row.is_active:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")
    return dict(row._mapping)
```

- [ ] **Step 4: Add get_current_user to documents router**

In `api/routers/documents.py`, add the import:

```python
from api.dependencies import get_current_user
```

Add `user: dict = Depends(get_current_user)` to every endpoint signature. The five endpoints to update are `preprocess`, `build_index`, `list_documents`, `upload`, and `document_status`. For example:

```python
@router.get("")
async def list_documents(request: Request, user: dict = Depends(get_current_user)) -> JSONResponse:
```

```python
@router.post("/upload")
async def upload(request: Request, file: UploadFile, user: dict = Depends(get_current_user)) -> JSONResponse:
```

```python
@router.get("/status/{document_id}")
async def document_status(document_id: str, request: Request, user: dict = Depends(get_current_user)) -> JSONResponse:
```

```python
@router.post("/preprocess")
async def preprocess(request: Request, file: UploadFile, user: dict = Depends(get_current_user)) -> JSONResponse:
```

```python
@router.post("/index")
async def build_index(request: Request, user: dict = Depends(get_current_user)) -> JSONResponse:
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
pytest tests/unit/test_dependencies.py -v
```

Expected: all 3 tests PASS.

- [ ] **Step 6: Run full test suite**

```bash
pytest tests/ -x -q
```

Expected: all tests pass.

- [ ] **Step 7: Commit**

```bash
git add api/dependencies.py api/routers/documents.py tests/unit/test_dependencies.py
git commit -m "feat(auth): add get_current_user dependency and protect document routes"
```

---

### Task 8: User-scoped document and query routes

**Files:**
- Modify: `api/routers/documents.py`
- Modify: `api/routers/query.py`
- Create: `tests/unit/test_user_scoping.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_user_scoping.py`:

```python
from pathlib import Path
from config import Settings, user_scoped_settings


def test_user_scoped_settings_scopes_raw_dir():
    s = Settings(raw_documents_dir=Path("data/raw"), processed_documents_dir=Path("data/processed"),
                 vectorstore_dir=Path("data/embedded"), openai_api_key="sk-fake", jwt_secret_key="s")
    scoped = user_scoped_settings(s, "user-abc")
    assert scoped.raw_documents_dir == Path("data/raw/user-abc")


def test_user_scoped_settings_scopes_processed_dir():
    s = Settings(raw_documents_dir=Path("data/raw"), processed_documents_dir=Path("data/processed"),
                 vectorstore_dir=Path("data/embedded"), openai_api_key="sk-fake", jwt_secret_key="s")
    scoped = user_scoped_settings(s, "user-abc")
    assert scoped.processed_documents_dir == Path("data/processed/user-abc")


def test_user_scoped_settings_scopes_vectorstore_dir():
    s = Settings(raw_documents_dir=Path("data/raw"), processed_documents_dir=Path("data/processed"),
                 vectorstore_dir=Path("data/embedded"), openai_api_key="sk-fake", jwt_secret_key="s")
    scoped = user_scoped_settings(s, "user-abc")
    assert scoped.vectorstore_dir == Path("data/embedded/user-abc")


def test_user_scoped_settings_different_users_get_different_dirs():
    s = Settings(raw_documents_dir=Path("data/raw"), processed_documents_dir=Path("data/processed"),
                 vectorstore_dir=Path("data/embedded"), openai_api_key="sk-fake", jwt_secret_key="s")
    scoped_a = user_scoped_settings(s, "user-a")
    scoped_b = user_scoped_settings(s, "user-b")
    assert scoped_a.processed_documents_dir != scoped_b.processed_documents_dir
```

- [ ] **Step 2: Run test to verify it passes (confirms Task 1 implementation)**

```bash
pytest tests/unit/test_user_scoping.py -v
```

Expected: all 4 tests PASS. (These test `user_scoped_settings` added in Task 1.)

- [ ] **Step 3: Update documents.py to use user-scoped settings**

In `api/routers/documents.py`, add the import:

```python
from config import user_scoped_settings
```

In the `upload` endpoint, replace `settings` with `scoped` when interacting with the filesystem and calling the pipeline. The key change is to derive `scoped` from the authenticated user and use it throughout:

```python
@router.post("/upload")
async def upload(request: Request, file: UploadFile, user: dict = Depends(get_current_user)) -> JSONResponse:
    settings = request.app.state.settings
    scoped = user_scoped_settings(settings, user["id"])
    jobs: dict = request.app.state.jobs

    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    scoped.raw_documents_dir.mkdir(parents=True, exist_ok=True)
    dest = scoped.raw_documents_dir / file.filename
    logger.info("Saving uploaded file: %s", dest)
    try:
        with dest.open("wb") as out:
            shutil.copyfileobj(file.file, out)
    finally:
        await file.close()

    document_id = dest.stem
    jobs[document_id] = {
        "status": "preprocessing", "error": None,
        "chunk_count": None, "page_count": None, "source_filename": file.filename,
    }
    logger.info("Starting pipeline for document_id=%s", document_id)

    thread = threading.Thread(
        target=_run_pipeline,
        args=(dest, scoped, jobs, document_id),
        daemon=True,
    )
    thread.start()
    return JSONResponse({"document_id": document_id, "status": "preprocessing"})
```

In `list_documents`, replace `settings.processed_documents_dir` with `scoped.processed_documents_dir`:

```python
@router.get("")
async def list_documents(request: Request, user: dict = Depends(get_current_user)) -> JSONResponse:
    settings = request.app.state.settings
    scoped = user_scoped_settings(settings, user["id"])
    jobs: dict = request.app.state.jobs
    documents = []
    seen: set[str] = set()
    # ... rest of function: replace settings.processed_documents_dir with scoped.processed_documents_dir
    processed_dir = scoped.processed_documents_dir
    # ... rest unchanged
```

In `document_status`, replace `settings.processed_documents_dir` with `scoped.processed_documents_dir`:

```python
@router.get("/status/{document_id}")
async def document_status(document_id: str, request: Request, user: dict = Depends(get_current_user)) -> JSONResponse:
    settings = request.app.state.settings
    scoped = user_scoped_settings(settings, user["id"])
    jobs: dict = request.app.state.jobs
    # ...
    doc_dir = (scoped.processed_documents_dir / document_id).resolve()
    if not str(doc_dir).startswith(str(scoped.processed_documents_dir.resolve())):
        raise HTTPException(status_code=400, detail="Invalid document_id.")
    # ... rest unchanged
```

- [ ] **Step 4: Update query.py to scope by user**

In `api/routers/query.py`, add the imports:

```python
from api.dependencies import get_current_user
from config import user_scoped_settings
```

Update the `query` endpoint:

```python
@router.post("")
async def query(request: Request, body: QueryRequest, user: dict = Depends(get_current_user)) -> JSONResponse:
    settings = request.app.state.settings
    scoped = user_scoped_settings(settings, user["id"])

    if not scoped.openai_api_key:
        raise HTTPException(
            status_code=422,
            detail="OPENAI_API_KEY is required for queries. Set it in your environment.",
        )

    logger.info("Query received: %r (top_k=%d)", body.question, body.top_k)
    started = time.time()
    try:
        response = answer_question_from_frozen_artifacts(
            body.question,
            settings=scoped,
            top_k=body.top_k,
            doc_filter=body.doc_filter,
        )
    except Exception as exc:
        logger.exception("Query failed: %r", body.question)
        raise HTTPException(status_code=500, detail=f"Query failed: {exc}") from exc

    return JSONResponse({
        "question": response.question,
        "answer": response.answer,
        "sources": response.sources,
        "router": response.router,
        "specialists": [
            {"agent_name": s.agent_name, "output": s.output, "region_ids": s.region_ids}
            for s in response.specialists
        ],
        "latency_ms": int((time.time() - started) * 1000),
        "top_k": body.top_k,
    })
```

- [ ] **Step 5: Run tests**

```bash
pytest tests/unit/test_user_scoping.py tests/unit/test_dependencies.py tests/unit/test_auth_router.py -v
```

Expected: all tests PASS.

- [ ] **Step 6: Run full test suite**

```bash
pytest tests/ -x -q
```

Expected: all tests pass.

- [ ] **Step 7: Commit**

```bash
git add api/routers/documents.py api/routers/query.py tests/unit/test_user_scoping.py
git commit -m "feat(auth): scope document and query endpoints by authenticated user"
```

---

### Task 9: OAuth integration (Google and GitHub)

**Files:**
- Create: `api/auth/oauth.py`
- Modify: `api/routers/auth.py`
- Modify: `api/app.py`

Note: OAuth routes require live credentials and a browser to test. Verify manually after Task 11 (frontend). Unit tests are not practical here without full HTTP mocking.

- [ ] **Step 1: Create api/auth/oauth.py**

```python
from authlib.integrations.starlette_client import OAuth

oauth = OAuth()


def register_providers(
    google_client_id: str | None,
    google_client_secret: str | None,
    github_client_id: str | None,
    github_client_secret: str | None,
) -> None:
    if google_client_id and google_client_secret:
        oauth.register(
            name="google",
            client_id=google_client_id,
            client_secret=google_client_secret,
            server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
            client_kwargs={"scope": "openid email"},
        )
    if github_client_id and github_client_secret:
        oauth.register(
            name="github",
            client_id=github_client_id,
            client_secret=github_client_secret,
            access_token_url="https://github.com/login/oauth/access_token",
            authorize_url="https://github.com/login/oauth/authorize",
            api_base_url="https://api.github.com/",
            client_kwargs={"scope": "user:email"},
        )
```

- [ ] **Step 2: Add OAuth routes to api/routers/auth.py**

Add this import at the top of `api/routers/auth.py`:

```python
from api.auth.oauth import oauth
```

Add the four OAuth routes and one helper at the bottom of `api/routers/auth.py`:

```python
def _oauth_upsert_user(engine, settings, response: Response,
                        email: str, provider: str, sub: str) -> RedirectResponse:
    now = datetime.now(timezone.utc)
    with engine.connect() as conn:
        row = conn.execute(select(users).where(users.c.email == email)).first()
        if row:
            user_id = row.id
        else:
            user_id = str(uuid.uuid4())
            conn.execute(insert(users).values(
                id=user_id, email=email, hashed_password=None,
                oauth_provider=provider, oauth_sub=sub,
                created_at=now, is_active=True,
            ))
        access_token = create_access_token(
            user_id, settings.jwt_secret_key, settings.jwt_algorithm,
            settings.access_token_expire_minutes,
        )
        raw_refresh, token_hash = create_refresh_token()
        conn.execute(insert(refresh_tokens).values(
            token_hash=token_hash, user_id=user_id,
            expires_at=now + timedelta(days=settings.refresh_token_expire_days),
        ))
        conn.commit()
    redirect = RedirectResponse(url=f"/#token={access_token}")
    _set_refresh_cookie(redirect, raw_refresh, settings)
    return redirect


@router.get("/oauth/google")
async def google_login(request: Request):
    redirect_uri = str(request.url_for("google_callback"))
    return await oauth.google.authorize_redirect(request, redirect_uri)


@router.get("/oauth/google/callback", name="google_callback")
async def google_callback(request: Request, response: Response):
    settings = request.app.state.settings
    engine = request.app.state.db_engine
    token = await oauth.google.authorize_access_token(request)
    user_info = token.get("userinfo") or {}
    email = user_info.get("email", "").lower().strip()
    sub = user_info.get("sub", "")
    if not email:
        raise HTTPException(status_code=400, detail="Could not get email from Google")
    return _oauth_upsert_user(engine, settings, response, email, "google", sub)


@router.get("/oauth/github")
async def github_login(request: Request):
    redirect_uri = str(request.url_for("github_callback"))
    return await oauth.github.authorize_redirect(request, redirect_uri)


@router.get("/oauth/github/callback", name="github_callback")
async def github_callback(request: Request, response: Response):
    settings = request.app.state.settings
    engine = request.app.state.db_engine
    token = await oauth.github.authorize_access_token(request)
    resp = await oauth.github.get("user", token=token)
    user_info = resp.json()
    email = (user_info.get("email") or "").lower().strip()
    if not email:
        emails_resp = await oauth.github.get("user/emails", token=token)
        primary = next((e for e in emails_resp.json() if e.get("primary")), None)
        email = (primary["email"] if primary else "").lower().strip()
    if not email:
        raise HTTPException(status_code=400, detail="Could not get email from GitHub")
    sub = str(user_info.get("id", ""))
    return _oauth_upsert_user(engine, settings, response, email, "github", sub)
```

- [ ] **Step 3: Register OAuth providers in app lifespan**

In `api/app.py`, in the lifespan function after `create_tables(engine)`, add:

```python
        from api.auth.oauth import register_providers
        register_providers(
            resolved_settings.google_client_id,
            resolved_settings.google_client_secret,
            resolved_settings.github_client_id,
            resolved_settings.github_client_secret,
        )
```

- [ ] **Step 4: Run existing tests to confirm nothing broke**

```bash
pytest tests/ -x -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add api/auth/oauth.py api/routers/auth.py api/app.py
git commit -m "feat(auth): add Google and GitHub OAuth integration"
```

---

### Task 10: Login frontend page

**Files:**
- Create: `api/static/login.html`
- Create: `api/static/login.js`

- [ ] **Step 1: Create api/static/login.html**

```html
<!doctype html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>PDF RAG — Sign In</title>
    <link rel="preconnect" href="https://fonts.googleapis.com" />
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
    <link href="https://fonts.googleapis.com/css2?family=Sora:wght@400;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap" rel="stylesheet" />
    <link rel="stylesheet" href="./styles.css" />
    <style>
      body { display: flex; align-items: center; justify-content: center; min-height: 100vh; background: var(--bg, #0f1117); }
      .auth-card { background: var(--surface, #1a1d27); border: 1px solid var(--border, #2a2d3a); border-radius: 12px; padding: 2rem; width: 100%; max-width: 400px; }
      .auth-card__title { font-family: 'Sora', sans-serif; font-size: 1.4rem; font-weight: 700; margin-bottom: 1.5rem; color: var(--text, #e8eaf0); }
      .auth-tabs { display: flex; margin-bottom: 1.5rem; border-bottom: 1px solid var(--border, #2a2d3a); }
      .auth-tab { background: none; border: none; border-bottom: 2px solid transparent; padding: 0.5rem 1rem; cursor: pointer; font-family: 'Sora', sans-serif; font-size: 0.9rem; color: var(--text-muted, #6b7280); margin-bottom: -1px; }
      .auth-tab.active { border-bottom-color: var(--accent, #6c8ef5); color: var(--text, #e8eaf0); }
      .auth-form { display: flex; flex-direction: column; gap: 0.75rem; }
      .auth-form input { background: var(--input-bg, #12141e); border: 1px solid var(--border, #2a2d3a); border-radius: 6px; color: var(--text, #e8eaf0); font-family: 'IBM Plex Mono', monospace; padding: 0.6rem 0.8rem; font-size: 0.9rem; }
      .auth-form input:focus { outline: none; border-color: var(--accent, #6c8ef5); }
      .auth-btn { background: var(--accent, #6c8ef5); border: none; border-radius: 6px; color: #fff; cursor: pointer; font-family: 'Sora', sans-serif; font-size: 0.95rem; font-weight: 600; padding: 0.7rem; margin-top: 0.25rem; }
      .auth-btn:hover { opacity: 0.9; }
      .auth-divider { text-align: center; color: var(--text-muted, #6b7280); font-size: 0.8rem; margin: 0.75rem 0; }
      .oauth-btn { display: flex; align-items: center; justify-content: center; gap: 0.5rem; background: none; border: 1px solid var(--border, #2a2d3a); border-radius: 6px; color: var(--text, #e8eaf0); cursor: pointer; font-family: 'Sora', sans-serif; font-size: 0.9rem; padding: 0.6rem; text-decoration: none; }
      .oauth-btn:hover { border-color: var(--accent, #6c8ef5); }
      .auth-error { background: #3b1a1a; border: 1px solid #7f2020; border-radius: 6px; color: #f87171; font-size: 0.85rem; padding: 0.5rem 0.75rem; }
    </style>
  </head>
  <body>
    <div class="auth-card">
      <p class="auth-card__title">PDF RAG — Document Q&amp;A</p>
      <div class="auth-tabs">
        <button class="auth-tab active" data-tab="signin">Sign In</button>
        <button class="auth-tab" data-tab="register">Register</button>
      </div>
      <div id="auth-error" class="auth-error" hidden></div>
      <form class="auth-form" id="auth-form" novalidate>
        <input type="email" id="email" placeholder="Email" autocomplete="email" required />
        <input type="password" id="password" placeholder="Password (min 8 chars)" autocomplete="current-password" required />
        <button type="submit" class="auth-btn" id="submit-btn">Sign In</button>
      </form>
      <div class="auth-divider">or</div>
      <div style="display:flex;flex-direction:column;gap:0.5rem;">
        <a class="oauth-btn" href="/auth/oauth/google">Continue with Google</a>
        <a class="oauth-btn" href="/auth/oauth/github">Continue with GitHub</a>
      </div>
    </div>
    <script type="module" src="./login.js"></script>
  </body>
</html>
```

- [ ] **Step 2: Create api/static/login.js**

```javascript
const tabs = document.querySelectorAll('.auth-tab');
const submitBtn = document.getElementById('submit-btn');
const authForm = document.getElementById('auth-form');
const errorEl = document.getElementById('auth-error');
let mode = 'signin';

tabs.forEach((tab) => {
  tab.addEventListener('click', () => {
    mode = tab.dataset.tab;
    tabs.forEach((t) => t.classList.toggle('active', t === tab));
    submitBtn.textContent = mode === 'signin' ? 'Sign In' : 'Create Account';
    document.getElementById('password').autocomplete =
      mode === 'signin' ? 'current-password' : 'new-password';
    hideError();
  });
});

// Consume access token dropped in URL fragment by OAuth callback
const fragment = new URLSearchParams(window.location.hash.slice(1));
const fragmentToken = fragment.get('token');
if (fragmentToken) {
  history.replaceState(null, '', window.location.pathname);
  sessionStorage.setItem('__auth_token_once', fragmentToken);
  window.location.href = '/';
}

authForm.addEventListener('submit', async (e) => {
  e.preventDefault();
  hideError();
  const email = document.getElementById('email').value.trim();
  const password = document.getElementById('password').value;
  const endpoint = mode === 'signin' ? '/auth/login' : '/auth/register';

  submitBtn.disabled = true;
  submitBtn.textContent = '…';
  try {
    const r = await fetch(endpoint, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ email, password }),
    });
    const data = await r.json();
    if (!r.ok) {
      showError(data.detail || 'Authentication failed.');
      return;
    }
    sessionStorage.setItem('__auth_token_once', data.access_token);
    window.location.href = '/';
  } catch {
    showError('Network error. Is the server running?');
  } finally {
    submitBtn.disabled = false;
    submitBtn.textContent = mode === 'signin' ? 'Sign In' : 'Create Account';
  }
});

function showError(msg) { errorEl.textContent = msg; errorEl.hidden = false; }
function hideError() { errorEl.hidden = true; }
```

- [ ] **Step 3: Run existing tests**

```bash
pytest tests/ -x -q
```

Expected: all tests pass.

- [ ] **Step 4: Commit**

```bash
git add api/static/login.html api/static/login.js
git commit -m "feat(auth): add login/register frontend page"
```

---

### Task 11: Update existing frontend JS for authenticated requests

**Files:**
- Modify: `api/static/app.js`
- Modify: `api/static/sidebar.js`
- Modify: `api/static/query.js`
- Modify: `api/static/index.html`

- [ ] **Step 1: Replace api/static/app.js**

```javascript
import { initSidebar } from './sidebar.js';
import { initQuery } from './query.js';

// Access token lives here only — never in localStorage or sessionStorage
let _token = null;

export async function authedFetch(url, options = {}) {
  const headers = { ...(options.headers || {}) };
  if (_token) headers['Authorization'] = `Bearer ${_token}`;
  const r = await fetch(url, { ...options, headers });
  if (r.status === 401) {
    const ok = await tryRefresh();
    if (ok) {
      headers['Authorization'] = `Bearer ${_token}`;
      return fetch(url, { ...options, headers });
    }
    redirectToLogin();
  }
  return r;
}

async function tryRefresh() {
  try {
    const r = await fetch('/auth/refresh', { method: 'POST' });
    if (!r.ok) return false;
    _token = (await r.json()).access_token;
    return true;
  } catch {
    return false;
  }
}

function redirectToLogin() {
  _token = null;
  window.location.href = '/login';
}

async function init() {
  const once = sessionStorage.getItem('__auth_token_once');
  if (once) {
    sessionStorage.removeItem('__auth_token_once');
    _token = once;
  } else {
    const ok = await tryRefresh();
    if (!ok) { redirectToLogin(); return; }
  }

  const selectedIds = new Set();

  const query = initQuery({ getSelectedIds: () => new Set(selectedIds), authedFetch });

  const sidebar = initSidebar({
    onSelectionChange: (ids, names) => {
      selectedIds.clear();
      ids.forEach((id) => selectedIds.add(id));
      query.onSelectionChange(new Set(selectedIds), names);
    },
    authedFetch,
  });

  authedFetch('/documents')
    .then((r) => r.json())
    .then(({ documents }) => sidebar.loadExisting(documents || []))
    .catch(() => {});

  const logoutBtn = document.getElementById('logout-btn');
  if (logoutBtn) {
    logoutBtn.addEventListener('click', async () => {
      await fetch('/auth/logout', { method: 'POST' });
      redirectToLogin();
    });
  }
}

init();
```

- [ ] **Step 2: Update api/static/sidebar.js**

Change the function signature on line 9:

```javascript
export function initSidebar({ onSelectionChange, authedFetch }) {
```

Replace `fetch('/documents/upload', ...)` (line 48) with `authedFetch('/documents/upload', ...)`.

Replace `fetch(\`/documents/status/${document_id}\`)` (line 73) with `authedFetch(\`/documents/status/${document_id}\`)`.

- [ ] **Step 3: Update api/static/query.js**

Change the function signature on line 6:

```javascript
export function initQuery({ getSelectedIds, authedFetch }) {
```

Replace `fetch('/query', ...)` (line 95) with `authedFetch('/query', ...)`.

Also fix the request body on line 99 — `document_ids` should be `doc_filter`:

```javascript
body: JSON.stringify({ question, top_k: 4, doc_filter: [...ids] }),
```

- [ ] **Step 4: Add logout button to api/static/index.html**

Replace the sidebar footer:

```html
        <footer class="sidebar__footer" id="sidebar-footer">
          <span id="selection-summary">0 selected</span>
          <button id="logout-btn" style="background:none;border:none;color:var(--text-muted,#6b7280);cursor:pointer;font-size:0.8rem;padding:0.25rem 0.5rem;">Sign out</button>
        </footer>
```

- [ ] **Step 5: Run full test suite**

```bash
pytest tests/ -x -q
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add api/static/app.js api/static/sidebar.js api/static/query.js api/static/index.html
git commit -m "feat(auth): wire authedFetch and token lifecycle into frontend"
```

---

### Task 12: Manual end-to-end verification

- [ ] **Step 1: Configure .env and start the server**

```bash
# Generate a secret key
python -c "import os; print(os.urandom(32).hex())"
# Add JWT_SECRET_KEY=<output> to .env
source .venv/bin/activate
python main.py --serve
```

Expected: logs show "Database ready", "Server ready" on port 8000.

- [ ] **Step 2: Verify unauthenticated redirect**

Open `http://localhost:8000` in a browser.
Expected: redirects to `http://localhost:8000/login`.

- [ ] **Step 3: Register and verify app loads**

On `/login`, switch to "Register", enter an email and password (8+ chars), click "Create Account".
Expected: redirects to `/` and the document sidebar loads.

- [ ] **Step 4: Verify document isolation between two users**

Upload a PDF as user A. Open an incognito window, register as user B, verify the document list is empty.

- [ ] **Step 5: Verify logout**

Click "Sign out". Expected: redirects to `/login`. Navigating to `/` should redirect back to `/login`.

- [ ] **Step 6: Verify token refresh survives page reload**

Sign in, reload the page. Expected: app loads without redirecting to `/login` (silent refresh works via the `httponly` cookie).

- [ ] **Step 7: (Optional) Test OAuth with real credentials**

Add `GOOGLE_CLIENT_ID`, `GOOGLE_CLIENT_SECRET` to `.env`, restart, click "Continue with Google", complete OAuth flow, verify you land on `/`.

- [ ] **Step 8: Commit any fixes found during testing**

```bash
git add -p
git commit -m "fix(auth): address issues found in manual e2e verification"
```
