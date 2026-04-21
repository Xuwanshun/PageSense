from datetime import UTC, datetime

import pytest
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
        conn.execute(
            insert(users).values(
                id="u1",
                email="a@example.com",
                hashed_password="hash",
                created_at=datetime.now(UTC),
                is_active=True,
            )
        )
        conn.commit()
        row = conn.execute(select(users).where(users.c.id == "u1")).first()
    assert row.email == "a@example.com"


def test_users_email_unique_constraint(engine):
    with engine.connect() as conn:
        conn.execute(
            insert(users).values(
                id="u2",
                email="dup@example.com",
                hashed_password=None,
                created_at=datetime.now(UTC),
                is_active=True,
            )
        )
        conn.commit()
    with pytest.raises(IntegrityError):
        with engine.connect() as conn:
            conn.execute(
                insert(users).values(
                    id="u3",
                    email="dup@example.com",
                    hashed_password=None,
                    created_at=datetime.now(UTC),
                    is_active=True,
                )
            )
            conn.commit()


def test_insert_and_retrieve_refresh_token(engine):
    with engine.connect() as conn:
        conn.execute(
            insert(users).values(
                id="u4",
                email="b@example.com",
                hashed_password=None,
                created_at=datetime.now(UTC),
                is_active=True,
            )
        )
        conn.execute(
            insert(refresh_tokens).values(
                token_hash="abc123",
                user_id="u4",
                expires_at=datetime.now(UTC),
            )
        )
        conn.commit()
        row = conn.execute(select(refresh_tokens).where(refresh_tokens.c.token_hash == "abc123")).first()
    assert row.user_id == "u4"
