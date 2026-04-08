"""
Tests for rag/chunk.py — chunk conversion logic.

These tests run WITHOUT Paddle or OpenAI — they only test the pure
data-transformation functions that convert ProcessedChunk → ChunkRecord.

WHY TEST THIS
-------------
The ChunkRecord is the unit that gets embedded and stored in the vector
index. If fields are dropped, misnamed, or metadata is wrong, you will
get poor retrieval quality (bad RAG answers) with no obvious error.
"""
from __future__ import annotations

from rag.chunk import ChunkRecord, chunk_record_from_processed_chunk, chunk_records_from_processed_chunks
from document_Process.models import ProcessedChunk


def _make_chunk(
    chunk_id: str = "chunk-001",
    text: str = "Hello world",
    page_content: str = "",
    page_number: int = 1,
) -> ProcessedChunk:
    """Helper that builds a minimal ProcessedChunk for testing."""
    return ProcessedChunk(
        chunk_id=chunk_id,
        text=text,
        page_content=page_content or text,
        page_number=page_number,
    )


def test_chunk_record_has_correct_chunk_id():
    chunk = _make_chunk(chunk_id="abc-123")
    record = chunk_record_from_processed_chunk(chunk)
    assert record.chunk_id == "abc-123"


def test_chunk_record_uses_page_content_when_available():
    """page_content takes precedence over text when both are present."""
    chunk = _make_chunk(text="raw text", page_content="formatted content")
    record = chunk_record_from_processed_chunk(chunk)
    assert record.text == "formatted content"


def test_chunk_record_falls_back_to_text_when_page_content_empty():
    chunk = ProcessedChunk(
        chunk_id="x",
        text="fallback text",
        page_content="",
        page_number=1,
    )
    record = chunk_record_from_processed_chunk(chunk)
    assert record.text == "fallback text"


def test_chunk_record_metadata_includes_document_id():
    chunk = _make_chunk()
    record = chunk_record_from_processed_chunk(chunk, document_id="doc-abc")
    assert record.metadata["document_id"] == "doc-abc"


def test_chunk_record_metadata_includes_source_filename():
    chunk = _make_chunk()
    record = chunk_record_from_processed_chunk(chunk, source_filename="report.pdf")
    assert record.metadata["source_filename"] == "report.pdf"


def test_chunk_record_metadata_includes_page_number():
    chunk = _make_chunk(page_number=5)
    record = chunk_record_from_processed_chunk(chunk)
    assert record.metadata["page_number"] == 5


def test_chunk_record_metadata_none_values_are_excluded():
    """None values should be stripped from metadata to keep it clean."""
    chunk = _make_chunk()
    record = chunk_record_from_processed_chunk(chunk, document_id=None, source_filename=None)
    assert "document_id" not in record.metadata
    assert "source_filename" not in record.metadata


def test_chunk_records_from_list_skips_empty_text():
    """Chunks with only whitespace text should be excluded from the output."""
    chunks = [
        _make_chunk(chunk_id="c1", text="real content"),
        ProcessedChunk(chunk_id="c2", text="   ", page_content="   ", page_number=1),
        _make_chunk(chunk_id="c3", text="more content"),
    ]
    records = chunk_records_from_processed_chunks(chunks, document_id="doc")
    ids = [r.chunk_id for r in records]
    assert "c1" in ids
    assert "c2" not in ids
    assert "c3" in ids


def test_chunk_records_from_list_preserves_order():
    chunks = [_make_chunk(chunk_id=f"c{i}", text=f"text {i}") for i in range(5)]
    records = chunk_records_from_processed_chunks(chunks)
    assert [r.chunk_id for r in records] == [f"c{i}" for i in range(5)]


def test_chunk_record_is_frozen():
    """ChunkRecord is a frozen dataclass — it should not be mutable."""
    chunk = _make_chunk()
    record = chunk_record_from_processed_chunk(chunk)
    with pytest.raises(Exception):  # AttributeError or FrozenInstanceError
        record.chunk_id = "mutated"  # type: ignore[misc]


import pytest  # noqa: E402  (import at bottom to keep test functions clean)
