"""
Shared pytest fixtures.

Fixtures defined here are automatically available to all test files
without any imports. pytest discovers conftest.py automatically.

HOW FIXTURES WORK
-----------------
A pytest fixture is a function decorated with @pytest.fixture that
provides shared test resources. You use a fixture by adding its name
as a parameter to a test function:

    def test_something(tmp_settings):
        # tmp_settings is now the Settings object from the fixture
        assert tmp_settings.raw_documents_dir.exists()

pytest handles creating and cleaning up the fixture for each test.
"""

from __future__ import annotations

import pytest

from config import Settings


@pytest.fixture
def tmp_settings(tmp_path):
    """
    A Settings instance pointing all data directories at a temporary
    directory that is automatically cleaned up after the test.

    WHY we need this:
    The original Settings.__post_init__ created directories immediately,
    meaning tests would write to the real data/ directory. Now Settings
    has no side effects, and this fixture provides an isolated environment
    for each test.

    tmp_path is a built-in pytest fixture that gives you a unique
    temporary directory for each test (automatically deleted when done).
    """
    return Settings(
        raw_documents_dir=tmp_path / "raw",
        processed_documents_dir=tmp_path / "processed",
        vectorstore_dir=tmp_path / "embedded",
        paddle_cache_dir=tmp_path / "paddle",
        openai_api_key="sk-test-fake-key-for-unit-tests",
        openai_base_url=None,
        s3_bucket_name=None,
        log_level="WARNING",  # suppress log noise in tests
    )
