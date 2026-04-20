from __future__ import annotations

import hashlib
import uuid
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Cookie, HTTPException, Request, Response, status
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import delete, insert, select

from api.auth.oauth import oauth
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
