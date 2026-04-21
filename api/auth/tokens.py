import hashlib
import os
from datetime import UTC, datetime, timedelta

import jwt


def create_access_token(
    user_id: str, secret_key: str, algorithm: str, expire_minutes: int
) -> str:
    expire = datetime.now(UTC) + timedelta(minutes=expire_minutes)
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
