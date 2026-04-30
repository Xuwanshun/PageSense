"""
Tests for config.py — Settings loading and validation.

These tests confirm that:
  - Settings reads from environment variables correctly
  - Default values are correct
  - Settings construction is side-effect free (no directories created)
  - ensure_data_dirs() creates the expected directories

WHY THESE TESTS MATTER
-----------------------
Configuration bugs are silent and dangerous. A misconfigured container
pointing at the wrong S3 bucket or reading from the wrong directory will
fail in production — often at the worst moment. These tests catch that.
"""

from __future__ import annotations

from pathlib import Path

from config import Settings, ensure_data_dirs


def test_settings_defaults():
    """Default values are set correctly when no env vars override them."""
    s = Settings(openai_api_key=None)
    assert s.raw_documents_dir == Path("data/raw")
    assert s.processed_documents_dir == Path("data/processed")
    assert s.vectorstore_dir == Path("data/embedded")
    assert s.paddle_cache_dir == Path("data/.paddlex_cache")
    assert s.openai_model == "gpt-4.1-mini"
    assert s.embedding_model == "text-embedding-3-small"
    assert s.log_level == "INFO"
    assert s.log_format == "text"


def test_settings_reads_from_env_vars(monkeypatch):
    """Settings should pick up values from environment variables."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-from-env")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4o")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("DEFAULT_TOP_K", "8")

    s = Settings()
    assert s.openai_api_key == "sk-test-from-env"
    assert s.openai_model == "gpt-4o"
    assert s.log_level == "DEBUG"
    assert s.default_top_k == 8


def test_settings_construction_has_no_side_effects(tmp_path, monkeypatch):
    """Constructing Settings must NOT create directories or write files."""
    data_dir = tmp_path / "data"
    monkeypatch.setenv("RAW_DOCUMENTS_DIR", str(data_dir / "raw"))
    monkeypatch.setenv("PROCESSED_DOCUMENTS_DIR", str(data_dir / "processed"))

    Settings()

    # Directories should NOT have been created by Settings() alone
    assert not (data_dir / "raw").exists(), "Settings should not create directories on construction"
    assert not (data_dir / "processed").exists(), "Settings should not create directories on construction"


def test_ensure_data_dirs_creates_directories(tmp_path):
    """ensure_data_dirs() should create all four data directories."""
    settings = Settings(
        raw_documents_dir=tmp_path / "raw",
        processed_documents_dir=tmp_path / "processed",
        vectorstore_dir=tmp_path / "embedded",
        paddle_cache_dir=tmp_path / "paddle",
    )
    # Nothing should exist yet
    assert not (tmp_path / "raw").exists()

    ensure_data_dirs(settings)

    assert (tmp_path / "raw").exists()
    assert (tmp_path / "processed").exists()
    assert (tmp_path / "embedded").exists()
    assert (tmp_path / "paddle").exists()
    assert (tmp_path / "paddle" / "temp").exists()


def test_settings_path_fields_are_path_objects(tmp_settings):
    """Path fields should be Path objects, not strings."""
    assert isinstance(tmp_settings.raw_documents_dir, Path)
    assert isinstance(tmp_settings.processed_documents_dir, Path)
    assert isinstance(tmp_settings.vectorstore_dir, Path)
    assert isinstance(tmp_settings.paddle_cache_dir, Path)
