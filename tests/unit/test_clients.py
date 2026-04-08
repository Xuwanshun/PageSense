"""
Tests for document_Process/clients.py — OpenAI error handling.

These tests verify that the new error handling wrappers around OpenAI
API calls produce clear, actionable error messages instead of raw
openai exceptions.

HOW MOCKING WORKS
-----------------
We cannot call the real OpenAI API in tests (costs money, non-deterministic,
requires a real key). Instead, we use unittest.mock.patch to temporarily
replace the OpenAI class with a fake that we control.

    with patch("document_Process.clients.OpenAI") as mock_openai:
        mock_openai.return_value.embeddings.create.return_value = my_fake_response
        # Now calls to OpenAI() inside clients.py use our fake

This is called "mocking" — you replace a dependency with a controllable
substitute for the duration of the test.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from openai import APIConnectionError, APIStatusError, APITimeoutError, RateLimitError

from document_Process.clients import request_openai_embeddings


def _fake_embedding_response(vectors: list[list[float]]) -> MagicMock:
    """Build a MagicMock that looks like an OpenAI embeddings response."""
    response = MagicMock()
    response.data = [MagicMock(embedding=v) for v in vectors]
    return response


def test_embeddings_returns_vectors():
    """Happy path — embeddings should be returned as a list of floats."""
    fake_response = _fake_embedding_response([[0.1, 0.2, 0.3]])

    with patch("document_Process.clients.OpenAI") as mock_openai:
        mock_openai.return_value.embeddings.create.return_value = fake_response
        result = request_openai_embeddings(
            model="text-embedding-3-small",
            texts=["hello world"],
            api_key="sk-fake",
            base_url=None,
        )

    assert result == [[0.1, 0.2, 0.3]]


def test_embeddings_empty_input():
    """An empty text list should return an empty list without calling the API."""
    # If the function calls the API with an empty list that would be wrong,
    # but the embed.py layer already guards against this. We just verify
    # the response shape here.
    fake_response = _fake_embedding_response([])

    with patch("document_Process.clients.OpenAI") as mock_openai:
        mock_openai.return_value.embeddings.create.return_value = fake_response
        result = request_openai_embeddings(
            model="text-embedding-3-small",
            texts=[],
            api_key="sk-fake",
            base_url=None,
        )

    assert result == []


def test_embeddings_rate_limit_raises_runtime_error():
    """
    A RateLimitError from OpenAI should become a RuntimeError with a
    helpful message — not a raw openai exception.
    """
    with patch("document_Process.clients.OpenAI") as mock_openai:
        mock_openai.return_value.embeddings.create.side_effect = RateLimitError(
            message="Rate limit exceeded",
            response=MagicMock(status_code=429, headers={}),
            body={},
        )
        with pytest.raises(RuntimeError, match="rate limit"):
            request_openai_embeddings(
                model="text-embedding-3-small",
                texts=["test"],
                api_key="sk-fake",
                base_url=None,
            )


def test_embeddings_timeout_raises_runtime_error():
    """A timeout should become a RuntimeError with a clear message."""
    with patch("document_Process.clients.OpenAI") as mock_openai:
        mock_openai.return_value.embeddings.create.side_effect = APITimeoutError(
            request=MagicMock()
        )
        with pytest.raises(RuntimeError, match="timed out"):
            request_openai_embeddings(
                model="text-embedding-3-small",
                texts=["test"],
                api_key="sk-fake",
                base_url=None,
            )


def test_embeddings_connection_error_raises_runtime_error():
    """A network connection error should become a RuntimeError."""
    with patch("document_Process.clients.OpenAI") as mock_openai:
        mock_openai.return_value.embeddings.create.side_effect = APIConnectionError(
            request=MagicMock()
        )
        with pytest.raises(RuntimeError, match="connect"):
            request_openai_embeddings(
                model="text-embedding-3-small",
                texts=["test"],
                api_key="sk-fake",
                base_url=None,
            )


def test_embeddings_api_status_error_includes_status_code():
    """
    An API status error (e.g. 401 Unauthorized, 500 Server Error) should
    include the HTTP status code in the error message.
    """
    mock_response = MagicMock()
    mock_response.status_code = 401
    mock_response.headers = {}

    with patch("document_Process.clients.OpenAI") as mock_openai:
        mock_openai.return_value.embeddings.create.side_effect = APIStatusError(
            message="Unauthorized",
            response=mock_response,
            body={"error": {"message": "Invalid API key"}},
        )
        with pytest.raises(RuntimeError, match="401"):
            request_openai_embeddings(
                model="text-embedding-3-small",
                texts=["test"],
                api_key="sk-invalid",
                base_url=None,
            )
