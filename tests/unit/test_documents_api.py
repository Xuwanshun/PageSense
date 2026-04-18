# tests/unit/test_documents_api.py
from __future__ import annotations

import io
import json
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from api.app import create_app


@pytest.fixture
def client(tmp_settings):
    app = create_app(tmp_settings)
    tmp_settings.processed_documents_dir.mkdir(parents=True, exist_ok=True)
    with TestClient(app) as c:
        yield c


def test_list_documents_empty(client):
    r = client.get("/documents")
    assert r.status_code == 200
    assert r.json() == {"documents": []}


def test_list_documents_returns_ready_docs(client, tmp_settings):
    doc_dir = tmp_settings.processed_documents_dir / "my_doc"
    doc_dir.mkdir(parents=True)
    (doc_dir / "document.json").write_text(
        json.dumps({"source_filename": "my_doc.pdf", "page_count": 10}), encoding="utf-8"
    )
    (doc_dir / "chunks.json").write_text(json.dumps([{}, {}, {}]), encoding="utf-8")

    r = client.get("/documents")
    assert r.status_code == 200
    docs = r.json()["documents"]
    assert len(docs) == 1
    assert docs[0]["document_id"] == "my_doc"
    assert docs[0]["status"] == "ready"
    assert docs[0]["chunk_count"] == 3
    assert docs[0]["page_count"] == 10


def test_status_returns_404_for_unknown(client):
    r = client.get("/documents/status/nonexistent")
    assert r.status_code == 404


def test_status_returns_job_from_memory(client):
    app = client.app
    app.state.jobs["injected_doc"] = {
        "status": "preprocessing",
        "error": None,
        "chunk_count": None,
        "page_count": None,
        "source_filename": "injected_doc.pdf",
    }
    r = client.get("/documents/status/injected_doc")
    assert r.status_code == 200
    assert r.json()["status"] == "preprocessing"


def test_upload_rejects_non_pdf(client):
    data = {"file": ("test.txt", io.BytesIO(b"not a pdf"), "text/plain")}
    r = client.post("/documents/upload", files=data)
    assert r.status_code == 400


def test_upload_starts_pipeline_and_returns_document_id(client, tmp_settings):
    pdf_bytes = b"%PDF-1.4 fake pdf content"
    with (
        patch("api.routers.documents.preprocess_document") as mock_pre,
        patch("api.routers.documents.index_all_processed_documents") as mock_idx,
    ):
        mock_pre.return_value = type("R", (), {
            "document_id": "report", "chunk_count": 5, "page_count": 3, "warnings": []
        })()
        mock_idx.return_value = {"report": 5}

        data = {"file": ("report.pdf", io.BytesIO(pdf_bytes), "application/pdf")}
        r = client.post("/documents/upload", files=data)

    assert r.status_code == 200
    body = r.json()
    assert body["document_id"] == "report"
    assert body["status"] == "preprocessing"


def test_status_returns_ready_from_filesystem(client, tmp_settings):
    doc_dir = tmp_settings.processed_documents_dir / "fs_doc"
    doc_dir.mkdir(parents=True)
    (doc_dir / "document.json").write_text(
        json.dumps({"source_filename": "fs_doc.pdf", "page_count": 5}), encoding="utf-8"
    )
    (doc_dir / "chunks.json").write_text(json.dumps([{}, {}]), encoding="utf-8")

    r = client.get("/documents/status/fs_doc")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ready"
    assert body["document_id"] == "fs_doc"
    assert body["chunk_count"] == 2
    assert body["page_count"] == 5
    assert body["error"] is None
