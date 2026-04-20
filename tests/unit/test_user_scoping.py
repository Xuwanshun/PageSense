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
