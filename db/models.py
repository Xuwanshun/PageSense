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
