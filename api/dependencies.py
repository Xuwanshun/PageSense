from __future__ import annotations

import jwt
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import select

from api.auth.tokens import verify_access_token
from db.models import users

_bearer = HTTPBearer(auto_error=False)


async def get_current_user(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> dict:
    settings = request.app.state.settings

    if credentials is None:
        raise HTTPException(status_code=403, detail="Missing credentials")

    if not settings.jwt_secret_key:
        raise HTTPException(status_code=500, detail="Auth not configured: JWT_SECRET_KEY missing")
    try:
        user_id = verify_access_token(
            credentials.credentials, settings.jwt_secret_key, settings.jwt_algorithm
        )
    except jwt.PyJWTError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired token"
        ) from exc
    engine = request.app.state.db_engine
    with engine.connect() as conn:
        row = conn.execute(select(users).where(users.c.id == user_id)).first()
    if row is None or not row.is_active:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")
    return dict(row._mapping)
