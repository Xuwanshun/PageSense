from __future__ import annotations

import json

from rag.retrieve import JsonVectorStore


def _make_store(tmp_path, rows):
    p = tmp_path / "store.json"
    p.write_text(json.dumps({"rows": rows}), encoding="utf-8")
    return JsonVectorStore(p)


def test_query_filter_returns_only_matching_doc(tmp_path):
    store = _make_store(tmp_path, [
        {"chunk_id": "a1", "text": "alpha", "metadata": {"document_id": "doc_a"}, "embedding": [1.0, 0.0]},
        {"chunk_id": "b1", "text": "beta",  "metadata": {"document_id": "doc_b"}, "embedding": [1.0, 0.0]},
    ])
    results = store.query([1.0, 0.0], top_k=10, document_ids=["doc_a"])
    assert len(results) == 1
    assert results[0].chunk_id == "a1"


def test_query_filter_none_returns_all(tmp_path):
    store = _make_store(tmp_path, [
        {"chunk_id": "a1", "text": "alpha", "metadata": {"document_id": "doc_a"}, "embedding": [1.0, 0.0]},
        {"chunk_id": "b1", "text": "beta",  "metadata": {"document_id": "doc_b"}, "embedding": [1.0, 0.0]},
    ])
    results = store.query([1.0, 0.0], top_k=10, document_ids=None)
    assert len(results) == 2


def test_query_filter_empty_list_returns_nothing(tmp_path):
    store = _make_store(tmp_path, [
        {"chunk_id": "a1", "text": "alpha", "metadata": {"document_id": "doc_a"}, "embedding": [1.0, 0.0]},
    ])
    results = store.query([1.0, 0.0], top_k=10, document_ids=[])
    assert results == []
