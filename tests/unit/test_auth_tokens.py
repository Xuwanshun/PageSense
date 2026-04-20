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
