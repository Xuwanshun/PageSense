from sqlalchemy import JSON, Boolean, Column, DateTime, MetaData, String, Table, Text

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

# A conversation groups all messages from one chat session.
# Tied to a user — loading their history only returns their own conversations.
conversations = Table(
    "conversations",
    metadata,
    Column("id", String(36), primary_key=True),
    Column("user_id", String(36), nullable=False),
    Column("title", String(255), nullable=True),  # first question, truncated
    Column("created_at", DateTime(timezone=True), nullable=False),
)

# One row per message — either the user's question or the assistant's answer.
messages = Table(
    "messages",
    metadata,
    Column("id", String(36), primary_key=True),
    Column("conversation_id", String(36), nullable=False),
    Column("role", String(10), nullable=False),  # "user" or "assistant"
    Column("content", Text, nullable=False),
    Column("sources", JSON, nullable=True),  # retrieved chunks, stored as JSON
    Column("created_at", DateTime(timezone=True), nullable=False),
)
