from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from api.app import create_app
from rag.qa import MultiAgentQAResponse, answer_question_from_frozen_artifacts
from rag.retrieve import JsonVectorStore


def _make_store(tmp_path, rows):
    p = tmp_path / "store.json"
    p.write_text(json.dumps({"rows": rows}), encoding="utf-8")
    return JsonVectorStore(p)


def test_query_filter_returns_only_matching_doc(tmp_path):
    store = _make_store(
        tmp_path,
        [
            {"chunk_id": "a1", "text": "alpha", "metadata": {"document_id": "doc_a"}, "embedding": [1.0, 0.0]},
            {"chunk_id": "b1", "text": "beta", "metadata": {"document_id": "doc_b"}, "embedding": [1.0, 0.0]},
        ],
    )
    results = store.query([1.0, 0.0], top_k=10, document_ids=["doc_a"])
    assert len(results) == 1
    assert results[0].chunk_id == "a1"


def test_query_filter_none_returns_all(tmp_path):
    store = _make_store(
        tmp_path,
        [
            {"chunk_id": "a1", "text": "alpha", "metadata": {"document_id": "doc_a"}, "embedding": [1.0, 0.0]},
            {"chunk_id": "b1", "text": "beta", "metadata": {"document_id": "doc_b"}, "embedding": [1.0, 0.0]},
        ],
    )
    results = store.query([1.0, 0.0], top_k=10, document_ids=None)
    assert len(results) == 2


def test_query_filter_empty_list_returns_nothing(tmp_path):
    store = _make_store(
        tmp_path,
        [
            {"chunk_id": "a1", "text": "alpha", "metadata": {"document_id": "doc_a"}, "embedding": [1.0, 0.0]},
        ],
    )
    results = store.query([1.0, 0.0], top_k=10, document_ids=[])
    assert results == []


def test_qa_passes_document_ids_to_retriever(tmp_settings):
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = []

    with (
        patch("rag.qa.DocumentRetriever", return_value=mock_retriever),
        patch("rag.qa._synthesize_answer", return_value="mocked answer"),
    ):
        answer_question_from_frozen_artifacts(
            "What is X?",
            settings=tmp_settings,
            top_k=2,
            document_ids=["doc_a"],
        )

    mock_retriever.retrieve.assert_called_once()
    call_kwargs = mock_retriever.retrieve.call_args
    assert call_kwargs.kwargs.get("document_ids") == ["doc_a"]


def test_query_endpoint_forwards_document_ids(tmp_settings):
    app = create_app(tmp_settings)
    mock_response = MultiAgentQAResponse(
        question="test?",
        answer="test answer",
        sources=[],
        router={},
        specialists=[],
    )
    with patch("api.routers.query.answer_question_from_frozen_artifacts", return_value=mock_response) as mock_fn:
        with TestClient(app) as client:
            client.post(
                "/query",
                json={"question": "test?", "top_k": 2, "document_ids": ["doc_a", "doc_b"]},
            )
        mock_fn.assert_called_once()
        assert mock_fn.call_args.kwargs.get("document_ids") == ["doc_a", "doc_b"]
