# PDF RAG Web App Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the static demo frontend with a functional two-panel web app — sidebar for PDF upload/management, main panel for document-scoped live querying.

**Architecture:** Single FastAPI server gains three new endpoints (`POST /documents/upload`, `GET /documents/status/{id}`, `GET /documents`) plus a `document_ids` filter on `POST /query`. The frontend is rebuilt as three ES-module JS files (`sidebar.js`, `query.js`, `app.js`) replacing the old monolithic `app.js`. Upload triggers preprocess → index automatically; status is tracked in-memory on `app.state.jobs` and polled every 3 s by the frontend.

**Tech Stack:** FastAPI, Python 3.11, vanilla JS (ES modules, no bundler), pytest + httpx TestClient

---

## File Map

**Modified (backend):**
- `rag/retrieve.py` — add `document_ids` filter to `VectorStore` protocol, `JsonVectorStore`, `ChromaVectorStore`, `DocumentRetriever`
- `rag/qa.py` — thread `document_ids` through `answer_question_from_frozen_artifacts()`
- `api/app.py` — init `app.state.jobs` in lifespan; remove `ui` router
- `api/routers/documents.py` — add `GET /documents`, `POST /documents/upload`, `GET /documents/status/{id}`
- `api/routers/query.py` — add `document_ids` to `QueryRequest`

**Deleted:**
- `api/routers/ui.py`
- `api/static/examples.json`

**Replaced (frontend):**
- `api/static/index.html` — two-panel layout shell
- `api/static/styles.css` — dark theme, sidebar + main panel
- `api/static/app.js` — thin ES module coordinator
- `api/static/sidebar.js` — new file: upload, polling, doc list
- `api/static/query.js` — new file: query bar, chat, Ask form

**New (tests):**
- `tests/unit/test_retrieve_filter.py`
- `tests/unit/test_documents_api.py`

---

## Task 1: Add `document_ids` filter to vector store and retriever

**Files:**
- Modify: `rag/retrieve.py`
- Create: `tests/unit/test_retrieve_filter.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/unit/test_retrieve_filter.py
from __future__ import annotations

import json
import pytest
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
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
pytest tests/unit/test_retrieve_filter.py -v
```
Expected: `TypeError` — `query()` got unexpected keyword argument `document_ids`

- [ ] **Step 3: Update `rag/retrieve.py`**

Replace the `VectorStore` Protocol `query` signature, `JsonVectorStore.query`, `ChromaVectorStore.query`, and `DocumentRetriever.retrieve`:

```python
class VectorStore(Protocol):
    def upsert(self, chunks: list[ChunkRecord], embeddings: list[list[float]]) -> None: ...

    def query(
        self, embedding: list[float], top_k: int, *, document_ids: list[str] | None = None
    ) -> list[RetrievedChunk]: ...
```

```python
# In JsonVectorStore:
def query(
    self, embedding: list[float], top_k: int, *, document_ids: list[str] | None = None
) -> list[RetrievedChunk]:
    scored: list[RetrievedChunk] = []
    for row in self._load_rows():
        if document_ids is not None and row.get("metadata", {}).get("document_id") not in document_ids:
            continue
        score = _cosine_similarity(embedding, row.get("embedding", []))
        scored.append(
            RetrievedChunk(
                chunk_id=row["chunk_id"],
                text=row.get("text", ""),
                metadata=row.get("metadata", {}),
                score=score,
            )
        )
    return sorted(scored, key=lambda item: item.score, reverse=True)[:top_k]
```

```python
# In ChromaVectorStore:
def query(
    self, embedding: list[float], top_k: int, *, document_ids: list[str] | None = None
) -> list[RetrievedChunk]:
    kwargs: dict = {
        "query_embeddings": [embedding],
        "n_results": top_k,
        "include": ["documents", "metadatas", "distances"],
    }
    if document_ids is not None:
        kwargs["where"] = {"document_id": {"$in": document_ids}}
    response = self.collection.query(**kwargs)
    ids = (response.get("ids") or [[]])[0]
    documents = (response.get("documents") or [[]])[0]
    metadatas = (response.get("metadatas") or [[]])[0]
    distances = (response.get("distances") or [[]])[0]
    return [
        RetrievedChunk(
            chunk_id=chunk_id,
            text=text or "",
            metadata=metadata or {},
            score=1.0 - float(distance),
        )
        for chunk_id, text, metadata, distance in zip(ids, documents, metadatas, distances, strict=False)
    ]
```

```python
# In DocumentRetriever:
def retrieve(
    self, question: str, top_k: int | None = None, *, document_ids: list[str] | None = None
) -> list[RetrievedChunk]:
    query_embedding = self.embedding_backend.embed_texts([question])[0]
    return self.vector_store.query(
        query_embedding, top_k or self.settings.default_top_k, document_ids=document_ids
    )
```

- [ ] **Step 4: Run tests — verify they pass**

```bash
pytest tests/unit/test_retrieve_filter.py -v
```
Expected: 3 PASSED

- [ ] **Step 5: Commit**

```bash
git add rag/retrieve.py tests/unit/test_retrieve_filter.py
git commit -m "feat: add document_ids filter to vector store and retriever"
```

---

## Task 2: Thread `document_ids` through QA pipeline and query endpoint

**Files:**
- Modify: `rag/qa.py`
- Modify: `api/routers/query.py`

- [ ] **Step 1: Update `answer_question_from_frozen_artifacts` in `rag/qa.py`**

Change the function signature and pass `document_ids` to `retriever.retrieve()`:

```python
def answer_question_from_frozen_artifacts(
    question: str,
    *,
    settings: Settings | None = None,
    top_k: int | None = None,
    document_ids: list[str] | None = None,
) -> MultiAgentQAResponse:
    resolved_settings = settings or Settings()
    retriever = DocumentRetriever(resolved_settings)
    retrieved = _rerank_chunks(
        question,
        retriever.retrieve(
            question,
            top_k=(top_k or resolved_settings.default_top_k) * 2,
            document_ids=document_ids,
        ),
    )
    retrieved = retrieved[: top_k or resolved_settings.default_top_k]
    # rest of function unchanged
```

- [ ] **Step 2: Update `QueryRequest` and the query endpoint in `api/routers/query.py`**

Add `document_ids` field to the request model and pass it through:

```python
class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, description="The question to ask.")
    top_k: int = Field(default=4, ge=1, le=20, description="Number of chunks to retrieve (1-20).")
    document_ids: list[str] | None = Field(default=None, description="Limit search to these document IDs. Null searches all.")
```

In the `query` endpoint handler, add `document_ids=body.document_ids` to the call:

```python
response = answer_question_from_frozen_artifacts(
    body.question,
    settings=settings,
    top_k=body.top_k,
    document_ids=body.document_ids,
)
```

- [ ] **Step 3: Run existing tests to confirm nothing is broken**

```bash
pytest tests/unit/test_health.py tests/unit/test_config.py -v
```
Expected: all PASSED

- [ ] **Step 4: Commit**

```bash
git add rag/qa.py api/routers/query.py
git commit -m "feat: add document_ids filter to QA pipeline and query endpoint"
```

---

## Task 3: Initialize job tracker on `app.state` and remove `ui` router

**Files:**
- Modify: `api/app.py`
- Delete: `api/routers/ui.py`

- [ ] **Step 1: Update `api/app.py` — init jobs dict and remove ui router**

In the `lifespan` context manager, add job tracker initialisation before `yield`:

```python
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    configure_logging(
        log_level=resolved_settings.log_level,
        log_format=resolved_settings.log_format,
    )
    logger.info("Starting RAG API server")
    ensure_data_dirs(resolved_settings)

    # In-memory job tracker for upload pipeline status
    # Keys: document_id, Values: {"status", "error", "chunk_count", "page_count", "source_filename"}
    app.state.jobs = {}

    if resolved_settings.s3_bucket_name:
        from storage.s3 import sync_from_s3
        try:
            logger.info("Syncing artifacts from S3 bucket: %s", resolved_settings.s3_bucket_name)
            sync_from_s3(resolved_settings)
            logger.info("S3 sync complete")
        except Exception as exc:
            logger.warning("S3 sync on startup failed (continuing anyway): %s", exc)

    logger.info("Server ready")
    yield
    logger.info("Server shutting down")
```

Remove the `ui` import and `app.include_router(ui.router)` line:

```python
# Change this:
from api.routers import documents, health, query, ui

# To this:
from api.routers import documents, health, query
```

And remove:
```python
app.include_router(ui.router)
```

- [ ] **Step 2: Delete `api/routers/ui.py`**

```bash
rm api/routers/ui.py
```

- [ ] **Step 3: Verify server starts without errors**

```bash
python -c "from api.app import create_app; app = create_app(); print('OK')"
```
Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add api/app.py
git rm api/routers/ui.py
git commit -m "feat: init job tracker on app.state, remove ui router"
```

---

## Task 4: Add `GET /documents`, `POST /documents/upload`, `GET /documents/status/{document_id}`

**Files:**
- Modify: `api/routers/documents.py`
- Create: `tests/unit/test_documents_api.py`

- [ ] **Step 1: Write failing tests**

```python
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
    # Directly inject a job into app.state
    from api.app import create_app
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
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
pytest tests/unit/test_documents_api.py -v
```
Expected: failures on missing endpoints

- [ ] **Step 3: Add the three new endpoints to `api/routers/documents.py`**

Add these imports at the top of `documents.py`:

```python
import json
import threading
```

Add the `_run_pipeline` helper function (before the router endpoints):

```python
def _run_pipeline(
    dest: Path,
    settings,
    jobs: dict,
    document_id: str,
) -> None:
    """Run preprocess → index in a background thread and update jobs dict."""
    try:
        jobs[document_id]["status"] = "preprocessing"
        result = preprocess_document(dest, settings=settings, force=True, document_id=document_id)
        jobs[document_id]["status"] = "indexing"
        index_all_processed_documents(settings=settings)
        jobs[document_id].update(
            status="ready",
            chunk_count=result.chunk_count,
            page_count=result.page_count,
            error=None,
        )
        logger.info("Pipeline complete for document_id=%s", document_id)
    except Exception as exc:
        logger.exception("Pipeline failed for document_id=%s", document_id)
        jobs[document_id].update(status="error", error=str(exc))
```

Add the three new endpoint handlers after the existing `build_index` handler:

```python
@router.get("")
async def list_documents(request: Request) -> JSONResponse:
    """
    Return all documents — in-progress jobs and ready artifacts on disk.
    """
    settings = request.app.state.settings
    jobs: dict = request.app.state.jobs
    documents = []
    seen: set[str] = set()

    # In-progress and recently finished jobs (from memory)
    for document_id, job in jobs.items():
        seen.add(document_id)
        documents.append(
            {
                "document_id": document_id,
                "source_filename": job.get("source_filename", document_id),
                "status": job["status"],
                "chunk_count": job.get("chunk_count"),
                "page_count": job.get("page_count"),
                "error": job.get("error"),
            }
        )

    # Ready documents from filesystem (not in active jobs, e.g. from previous server runs)
    processed_dir = settings.processed_documents_dir
    if processed_dir.exists():
        for doc_dir in sorted(p for p in processed_dir.iterdir() if p.is_dir()):
            document_id = doc_dir.name
            if document_id in seen:
                continue
            doc_path = doc_dir / "document.json"
            chunks_path = doc_dir / "chunks.json"
            if not doc_path.exists():
                continue
            doc_data = json.loads(doc_path.read_text(encoding="utf-8"))
            chunk_count = len(json.loads(chunks_path.read_text(encoding="utf-8"))) if chunks_path.exists() else 0
            documents.append(
                {
                    "document_id": document_id,
                    "source_filename": doc_data.get("source_filename", document_id),
                    "status": "ready",
                    "chunk_count": chunk_count,
                    "page_count": doc_data.get("page_count"),
                    "error": None,
                }
            )

    return JSONResponse({"documents": documents})


@router.post("/upload")
async def upload(request: Request, file: UploadFile) -> JSONResponse:
    """
    Upload a PDF and automatically run preprocess + index in the background.

    Returns immediately with document_id and status="preprocessing".
    Poll GET /documents/status/{document_id} every 3 s to track progress.
    """
    settings = request.app.state.settings
    jobs: dict = request.app.state.jobs

    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    dest = settings.raw_documents_dir / file.filename
    settings.raw_documents_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Saving uploaded file: %s", dest)
    try:
        with dest.open("wb") as out:
            shutil.copyfileobj(file.file, out)
    finally:
        await file.close()

    document_id = dest.stem
    jobs[document_id] = {
        "status": "preprocessing",
        "error": None,
        "chunk_count": None,
        "page_count": None,
        "source_filename": file.filename,
    }
    logger.info("Starting pipeline for document_id=%s", document_id)

    thread = threading.Thread(
        target=_run_pipeline,
        args=(dest, settings, jobs, document_id),
        daemon=True,
    )
    thread.start()

    return JSONResponse({"document_id": document_id, "status": "preprocessing"})


@router.get("/status/{document_id}")
async def document_status(document_id: str, request: Request) -> JSONResponse:
    """
    Return the current pipeline status for a document.

    Status values: preprocessing | indexing | ready | error
    """
    settings = request.app.state.settings
    jobs: dict = request.app.state.jobs

    if document_id in jobs:
        job = jobs[document_id]
        return JSONResponse(
            {
                "document_id": document_id,
                "status": job["status"],
                "error": job.get("error"),
                "chunk_count": job.get("chunk_count"),
                "page_count": job.get("page_count"),
            }
        )

    # Fall back to filesystem for docs processed before this server run
    doc_dir = settings.processed_documents_dir / document_id
    if not doc_dir.exists():
        raise HTTPException(status_code=404, detail=f"Document {document_id!r} not found.")

    chunks_path = doc_dir / "chunks.json"
    doc_path = doc_dir / "document.json"
    chunk_count = len(json.loads(chunks_path.read_text(encoding="utf-8"))) if chunks_path.exists() else None
    doc_data = json.loads(doc_path.read_text(encoding="utf-8")) if doc_path.exists() else {}
    return JSONResponse(
        {
            "document_id": document_id,
            "status": "ready",
            "error": None,
            "chunk_count": chunk_count,
            "page_count": doc_data.get("page_count"),
        }
    )
```

Also add this import at the top of `documents.py` (alongside existing imports):

```python
from rag.retrieve import index_all_processed_documents
```

- [ ] **Step 4: Run tests — verify they pass**

```bash
pytest tests/unit/test_documents_api.py -v
```
Expected: all PASSED

- [ ] **Step 5: Commit**

```bash
git add api/routers/documents.py tests/unit/test_documents_api.py
git commit -m "feat: add upload, status, and list endpoints to documents router"
```

---

## Task 5: Write `index.html`

**Files:**
- Replace: `api/static/index.html`

- [ ] **Step 1: Replace `api/static/index.html`**

```html
<!doctype html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>PDF RAG — Document Q&A</title>
    <link rel="preconnect" href="https://fonts.googleapis.com" />
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
    <link href="https://fonts.googleapis.com/css2?family=Sora:wght@400;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap" rel="stylesheet" />
    <link rel="stylesheet" href="./styles.css" />
  </head>
  <body>
    <div class="app">

      <!-- ── Sidebar ─────────────────────────────────────── -->
      <aside class="sidebar" id="sidebar">
        <header class="sidebar__header">
          <p class="sidebar__title">Documents</p>
          <label class="upload-zone" id="upload-zone">
            <input type="file" id="file-input" accept=".pdf" hidden />
            <span class="upload-zone__icon">＋</span>
            <span class="upload-zone__text">Upload PDF</span>
          </label>
        </header>

        <ul class="doc-list" id="doc-list" aria-label="Document library">
          <li class="doc-list__empty" id="doc-list-empty">No documents yet. Upload a PDF to get started.</li>
        </ul>

        <footer class="sidebar__footer" id="sidebar-footer">
          <span id="selection-summary">0 selected</span>
        </footer>
      </aside>

      <!-- ── Main panel ──────────────────────────────────── -->
      <main class="main-panel">

        <div class="query-bar" id="query-bar" aria-label="Active document scope">
          <span class="query-bar__label">Querying:</span>
          <div class="query-bar__tags" id="query-tags">
            <span class="query-bar__empty">No documents selected</span>
          </div>
        </div>

        <div class="chat-history" id="chat-history" aria-live="polite">
          <p class="chat-history__empty">Select documents on the left, then ask a question.</p>
        </div>

        <div class="chat-input-area">
          <div id="ask-error" class="ask-error" hidden></div>
          <div class="chat-input">
            <textarea
              id="question-input"
              rows="2"
              placeholder="Ask a question about your documents…"
              aria-label="Question"
            ></textarea>
            <button id="ask-btn" type="button" disabled>Ask →</button>
          </div>
        </div>

      </main>
    </div>

    <script type="module" src="./app.js"></script>
  </body>
</html>
```

- [ ] **Step 2: Commit**

```bash
git add api/static/index.html
git commit -m "feat: new two-panel index.html layout shell"
```

---

## Task 6: Write `styles.css`

**Files:**
- Replace: `api/static/styles.css`

- [ ] **Step 1: Replace `api/static/styles.css`**

```css
/* ── Reset & base ──────────────────────────────────────────── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

:root {
  --bg:          #0f0f1a;
  --surface:     #1a1a2e;
  --surface2:    #2d2d4a;
  --border:      #2d2d4a;
  --accent:      #4f46e5;
  --accent-soft: #a5b4fc;
  --text:        #e2e8f0;
  --text-muted:  #6b7280;
  --text-dim:    #94a3b8;
  --green:       #22c55e;
  --amber:       #f59e0b;
  --red:         #ef4444;
  --sidebar-w:   280px;
  --radius:      8px;
  --font:        'Sora', system-ui, sans-serif;
  --mono:        'IBM Plex Mono', monospace;
}

html, body { height: 100%; }

body {
  font-family: var(--font);
  background: var(--bg);
  color: var(--text);
  font-size: 14px;
  line-height: 1.5;
}

/* ── App shell ─────────────────────────────────────────────── */
.app {
  display: flex;
  height: 100vh;
  overflow: hidden;
}

/* ── Sidebar ────────────────────────────────────────────────── */
.sidebar {
  width: var(--sidebar-w);
  min-width: var(--sidebar-w);
  background: var(--surface);
  border-right: 1px solid var(--border);
  display: flex;
  flex-direction: column;
  overflow: hidden;
}

.sidebar__header {
  padding: 16px;
  border-bottom: 1px solid var(--border);
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.sidebar__title {
  font-size: 11px;
  font-weight: 700;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: var(--text-muted);
}

/* Upload zone */
.upload-zone {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
  border: 2px dashed var(--border);
  border-radius: var(--radius);
  padding: 12px;
  cursor: pointer;
  background: var(--bg);
  transition: border-color 0.15s, background 0.15s;
}

.upload-zone:hover,
.upload-zone.drag-over {
  border-color: var(--accent);
  background: #1a1a3a;
}

.upload-zone__icon {
  font-size: 18px;
  color: var(--accent-soft);
  font-weight: 700;
}

.upload-zone__text {
  font-size: 13px;
  color: var(--accent-soft);
  font-weight: 600;
}

/* Document list */
.doc-list {
  flex: 1;
  overflow-y: auto;
  padding: 8px;
  list-style: none;
}

.doc-list__empty {
  padding: 16px;
  font-size: 12px;
  color: var(--text-muted);
  text-align: center;
  line-height: 1.6;
}

/* Document card */
.doc-card {
  background: var(--surface2);
  border-radius: var(--radius);
  padding: 10px 12px;
  margin-bottom: 6px;
  border: 1px solid var(--border);
  transition: border-color 0.15s, opacity 0.15s;
}

.doc-card--selected {
  border-color: var(--accent);
}

.doc-card--disabled {
  opacity: 0.5;
}

.doc-card__row {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 4px;
}

.doc-card__name {
  font-size: 12px;
  font-weight: 600;
  color: var(--text);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  flex: 1;
}

.doc-card__status {
  font-size: 11px;
  margin-left: 24px;
}

.doc-card__status--ready  { color: var(--green); }
.doc-card__status--processing { color: var(--amber); }
.doc-card__status--error  { color: var(--red); }

/* Pipeline steps (shown while processing) */
.doc-card__pipeline {
  margin-left: 24px;
  display: flex;
  flex-direction: column;
  gap: 3px;
  margin-top: 4px;
}

.pipeline-step {
  font-size: 11px;
  display: flex;
  align-items: center;
  gap: 6px;
}

.pipeline-step--done    { color: var(--green); }
.pipeline-step--active  { color: var(--amber); }
.pipeline-step--pending { color: var(--text-muted); }

/* Progress bar */
.doc-card__progress {
  margin: 6px 0 0 24px;
  height: 3px;
  background: var(--bg);
  border-radius: 2px;
  overflow: hidden;
}

.doc-card__progress-fill {
  height: 100%;
  background: var(--amber);
  border-radius: 2px;
  width: 0%;
  transition: width 0.5s;
}

/* Sidebar footer */
.sidebar__footer {
  padding: 10px 16px;
  border-top: 1px solid var(--border);
  font-size: 11px;
  color: var(--text-muted);
  text-align: center;
}

/* ── Main panel ─────────────────────────────────────────────── */
.main-panel {
  flex: 1;
  display: flex;
  flex-direction: column;
  overflow: hidden;
  background: var(--bg);
}

/* Query bar */
.query-bar {
  padding: 10px 16px;
  border-bottom: 1px solid var(--border);
  display: flex;
  align-items: center;
  gap: 8px;
  flex-wrap: wrap;
  min-height: 42px;
}

.query-bar__label {
  font-size: 11px;
  color: var(--text-muted);
  white-space: nowrap;
}

.query-bar__tags {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
}

.query-bar__empty {
  font-size: 11px;
  color: var(--text-muted);
  font-style: italic;
}

.query-tag {
  font-size: 11px;
  background: var(--surface2);
  color: var(--accent-soft);
  padding: 2px 10px;
  border-radius: 12px;
  font-family: var(--mono);
}

/* Chat history */
.chat-history {
  flex: 1;
  overflow-y: auto;
  padding: 20px 16px;
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.chat-history__empty {
  color: var(--text-muted);
  font-size: 13px;
  text-align: center;
  margin-top: 40px;
}

/* Chat bubbles */
.chat-turn { display: flex; flex-direction: column; gap: 8px; }

.chat-question {
  align-self: flex-end;
  max-width: 72%;
  background: var(--accent);
  color: #fff;
  padding: 10px 14px;
  border-radius: 12px 12px 2px 12px;
  font-size: 13px;
  line-height: 1.6;
}

.chat-answer {
  align-self: flex-start;
  max-width: 84%;
}

.chat-answer__text {
  background: var(--surface2);
  color: var(--text);
  padding: 12px 14px;
  border-radius: 2px 12px 12px 12px;
  font-size: 13px;
  line-height: 1.7;
}

.chat-answer__sources {
  display: flex;
  flex-wrap: wrap;
  gap: 5px;
  margin-top: 6px;
}

.source-chip {
  font-size: 10px;
  font-family: var(--mono);
  background: var(--surface);
  color: var(--text-muted);
  padding: 3px 8px;
  border-radius: 4px;
  border: 1px solid var(--border);
}

/* Chat input area */
.chat-input-area {
  padding: 12px 16px;
  border-top: 1px solid var(--border);
}

.ask-error {
  font-size: 12px;
  color: var(--red);
  margin-bottom: 8px;
}

.chat-input {
  display: flex;
  gap: 8px;
  align-items: flex-end;
}

.chat-input textarea {
  flex: 1;
  background: var(--surface2);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  color: var(--text);
  font-family: var(--font);
  font-size: 13px;
  padding: 10px 12px;
  resize: none;
  line-height: 1.5;
  outline: none;
  transition: border-color 0.15s;
}

.chat-input textarea:focus { border-color: var(--accent); }
.chat-input textarea::placeholder { color: var(--text-muted); }

.chat-input button {
  padding: 10px 20px;
  background: var(--accent);
  color: #fff;
  border: none;
  border-radius: var(--radius);
  font-family: var(--font);
  font-size: 13px;
  font-weight: 600;
  cursor: pointer;
  white-space: nowrap;
  transition: opacity 0.15s;
}

.chat-input button:disabled {
  opacity: 0.4;
  cursor: not-allowed;
}

.chat-input button:not(:disabled):hover { opacity: 0.85; }

/* Scrollbar styling */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--surface2); border-radius: 3px; }
```

- [ ] **Step 2: Commit**

```bash
git add api/static/styles.css
git commit -m "feat: new dark-theme styles for two-panel layout"
```

---

## Task 7: Write `sidebar.js`

**Files:**
- Create: `api/static/sidebar.js`

- [ ] **Step 1: Create `api/static/sidebar.js`**

```javascript
// sidebar.js — document library: upload, polling, doc list, selection

const POLL_INTERVAL_MS = 3000;

/**
 * @param {{ onSelectionChange: (ids: Set<string>) => void }} opts
 * @returns {{ loadExisting: (docs: Array) => void }}
 */
export function initSidebar({ onSelectionChange }) {
  const selectedIds = new Set();
  const pollingTimers = new Map(); // document_id -> intervalId
  const docCards = new Map();       // document_id -> li element

  const listEl = document.getElementById('doc-list');
  const emptyEl = document.getElementById('doc-list-empty');
  const summaryEl = document.getElementById('selection-summary');
  const uploadZone = document.getElementById('upload-zone');
  const fileInput = document.getElementById('file-input');

  // ── Drag-and-drop ─────────────────────────────────────────
  uploadZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadZone.classList.add('drag-over');
  });
  uploadZone.addEventListener('dragleave', () => uploadZone.classList.remove('drag-over'));
  uploadZone.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadZone.classList.remove('drag-over');
    const file = e.dataTransfer?.files[0];
    if (file) handleFile(file);
  });
  fileInput.addEventListener('change', () => {
    if (fileInput.files[0]) handleFile(fileInput.files[0]);
    fileInput.value = '';
  });

  // ── Upload ────────────────────────────────────────────────
  function handleFile(file) {
    if (!file.name.toLowerCase().endsWith('.pdf')) {
      alert('Only PDF files are supported.');
      return;
    }
    const formData = new FormData();
    formData.append('file', file);

    fetch('/documents/upload', { method: 'POST', body: formData })
      .then((r) => r.json())
      .then(({ document_id, status }) => {
        addOrUpdateCard(document_id, file.name, status, null, null);
        startPolling(document_id);
      })
      .catch(() => alert('Upload failed. Check server logs.'));
  }

  // ── Polling ───────────────────────────────────────────────
  function startPolling(document_id) {
    if (pollingTimers.has(document_id)) return;
    const id = setInterval(() => poll(document_id), POLL_INTERVAL_MS);
    pollingTimers.set(document_id, id);
  }

  function stopPolling(document_id) {
    const id = pollingTimers.get(document_id);
    if (id !== undefined) {
      clearInterval(id);
      pollingTimers.delete(document_id);
    }
  }

  function poll(document_id) {
    fetch(`/documents/status/${document_id}`)
      .then((r) => r.json())
      .then(({ status, error, chunk_count, page_count }) => {
        const card = docCards.get(document_id);
        if (!card) return;
        updateCardStatus(card, document_id, status, chunk_count, page_count, error);
        if (status === 'ready' || status === 'error') {
          stopPolling(document_id);
          if (status === 'ready') {
            enableCard(card, document_id);
          }
        }
      })
      .catch(() => {}); // silent — next tick will retry
  }

  // ── Card rendering ────────────────────────────────────────
  function addOrUpdateCard(document_id, source_filename, status, chunk_count, page_count) {
    emptyEl.hidden = true;

    if (docCards.has(document_id)) {
      updateCardStatus(docCards.get(document_id), document_id, status, chunk_count, page_count, null);
      return;
    }

    const li = document.createElement('li');
    li.className = 'doc-card doc-card--disabled';
    li.dataset.documentId = document_id;
    li.innerHTML = `
      <div class="doc-card__row">
        <input type="checkbox" disabled aria-label="${escHtml(source_filename)}" />
        <span class="doc-card__name" title="${escHtml(source_filename)}">${escHtml(source_filename)}</span>
      </div>
      <div class="doc-card__pipeline">
        <div class="pipeline-step pipeline-step--active">⟳ Preprocessing…</div>
        <div class="pipeline-step pipeline-step--pending">○ Indexing</div>
      </div>
    `;

    const checkbox = li.querySelector('input[type="checkbox"]');
    checkbox.addEventListener('change', () => {
      if (checkbox.checked) {
        selectedIds.add(document_id);
        li.classList.add('doc-card--selected');
      } else {
        selectedIds.delete(document_id);
        li.classList.remove('doc-card--selected');
      }
      updateSummary();
      onSelectionChange(new Set(selectedIds));
    });

    listEl.appendChild(li);
    docCards.set(document_id, li);

    if (status === 'ready') enableCard(li, document_id);
    updateCardStatus(li, document_id, status, chunk_count, page_count, null);
  }

  function updateCardStatus(li, document_id, status, chunk_count, page_count, error) {
    const pipeline = li.querySelector('.doc-card__pipeline');
    const statusEl = li.querySelector('.doc-card__status');

    if (status === 'preprocessing') {
      if (pipeline) pipeline.innerHTML = `
        <div class="pipeline-step pipeline-step--active">⟳ Preprocessing…</div>
        <div class="pipeline-step pipeline-step--pending">○ Indexing</div>
      `;
    } else if (status === 'indexing') {
      if (pipeline) pipeline.innerHTML = `
        <div class="pipeline-step pipeline-step--done">✓ Preprocessed</div>
        <div class="pipeline-step pipeline-step--active">⟳ Indexing…</div>
      `;
    } else if (status === 'ready') {
      const facts = [chunk_count != null ? `${chunk_count} chunks` : null, page_count != null ? `${page_count} pages` : null]
        .filter(Boolean).join(' · ');
      if (pipeline) pipeline.remove();
      const existing = li.querySelector('.doc-card__status');
      if (!existing) {
        const s = document.createElement('div');
        s.className = 'doc-card__status doc-card__status--ready';
        s.textContent = `● Ready${facts ? ' · ' + facts : ''}`;
        li.appendChild(s);
      } else {
        existing.textContent = `● Ready${facts ? ' · ' + facts : ''}`;
        existing.className = 'doc-card__status doc-card__status--ready';
      }
    } else if (status === 'error') {
      if (pipeline) pipeline.remove();
      const existing = li.querySelector('.doc-card__status');
      const msg = `✕ Error${error ? ': ' + error : ''}`;
      if (!existing) {
        const s = document.createElement('div');
        s.className = 'doc-card__status doc-card__status--error';
        s.textContent = msg;
        li.appendChild(s);
      } else {
        existing.textContent = msg;
        existing.className = 'doc-card__status doc-card__status--error';
      }
    }
  }

  function enableCard(li, document_id) {
    li.classList.remove('doc-card--disabled');
    const checkbox = li.querySelector('input[type="checkbox"]');
    checkbox.disabled = false;
    // Auto-select newly ready documents
    checkbox.checked = true;
    selectedIds.add(document_id);
    li.classList.add('doc-card--selected');
    updateSummary();
    onSelectionChange(new Set(selectedIds));
  }

  function updateSummary() {
    summaryEl.textContent = `${selectedIds.size} selected`;
  }

  function escHtml(str) {
    return String(str)
      .replace(/&/g, '&amp;').replace(/</g, '&lt;')
      .replace(/>/g, '&gt;').replace(/"/g, '&quot;');
  }

  // ── Public API ────────────────────────────────────────────
  return {
    loadExisting(docs) {
      docs.forEach(({ document_id, source_filename, status, chunk_count, page_count }) => {
        // addOrUpdateCard calls enableCard internally when status === 'ready'
        addOrUpdateCard(document_id, source_filename, status, chunk_count, page_count);
        if (status === 'preprocessing' || status === 'indexing') {
          startPolling(document_id);
        }
      });
    },
  };
}
```

- [ ] **Step 2: Commit**

```bash
git add api/static/sidebar.js
git commit -m "feat: sidebar.js — upload, polling, document list, selection"
```

---

## Task 8: Write `query.js`

**Files:**
- Create: `api/static/query.js`

- [ ] **Step 1: Create `api/static/query.js`**

```javascript
// query.js — query bar sync, chat interface, Ask form

/**
 * @param {{ getSelectedIds: () => Set<string> }} opts
 */
export function initQuery({ getSelectedIds }) {
  const queryTagsEl = document.getElementById('query-tags');
  const chatHistoryEl = document.getElementById('chat-history');
  const questionInput = document.getElementById('question-input');
  const askBtn = document.getElementById('ask-btn');
  const askErrorEl = document.getElementById('ask-error');

  // ── Query bar ──────────────────────────────────────────────
  function updateQueryBar(ids) {
    queryTagsEl.innerHTML = '';
    if (ids.size === 0) {
      queryTagsEl.innerHTML = '<span class="query-bar__empty">No documents selected</span>';
      askBtn.disabled = true;
      return;
    }
    ids.forEach((id) => {
      const tag = document.createElement('span');
      tag.className = 'query-tag';
      tag.textContent = id;
      queryTagsEl.appendChild(tag);
    });
    askBtn.disabled = false;
  }

  // Called by app.js whenever selection changes
  function onSelectionChange(ids) {
    updateQueryBar(ids);
  }

  // ── Chat ───────────────────────────────────────────────────
  function appendTurn(question, result) {
    // Remove empty state message
    const empty = chatHistoryEl.querySelector('.chat-history__empty');
    if (empty) empty.remove();

    const turn = document.createElement('div');
    turn.className = 'chat-turn';

    const qBubble = document.createElement('div');
    qBubble.className = 'chat-question';
    qBubble.textContent = question;

    const aWrap = document.createElement('div');
    aWrap.className = 'chat-answer';

    const aText = document.createElement('div');
    aText.className = 'chat-answer__text';
    aText.textContent = result.answer || '(no answer)';

    const sources = document.createElement('div');
    sources.className = 'chat-answer__sources';
    (result.sources || []).forEach((src) => {
      const chip = document.createElement('span');
      chip.className = 'source-chip';
      const file = src.source_filename || src.document_id || '?';
      const page = src.page_number != null ? ` p.${src.page_number}` : '';
      chip.textContent = `${file}${page}`;
      sources.appendChild(chip);
    });

    aWrap.appendChild(aText);
    if (sources.children.length) aWrap.appendChild(sources);
    turn.appendChild(qBubble);
    turn.appendChild(aWrap);
    chatHistoryEl.appendChild(turn);
    chatHistoryEl.scrollTop = chatHistoryEl.scrollHeight;
  }

  // ── Ask form ───────────────────────────────────────────────
  askBtn.addEventListener('click', submitQuestion);
  questionInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      submitQuestion();
    }
  });

  async function submitQuestion() {
    const question = questionInput.value.trim();
    if (!question) return;

    const ids = getSelectedIds();
    if (ids.size === 0) return;

    askErrorEl.hidden = true;
    askBtn.disabled = true;
    askBtn.textContent = '…';

    try {
      const r = await fetch('/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question, top_k: 4, document_ids: [...ids] }),
      });
      const data = await r.json();
      if (!r.ok) {
        showError(data.detail || 'Query failed.');
        return;
      }
      questionInput.value = '';
      appendTurn(question, data);
    } catch {
      showError('Network error. Is the server running?');
    } finally {
      askBtn.disabled = ids.size === 0;
      askBtn.textContent = 'Ask →';
    }
  }

  function showError(msg) {
    askErrorEl.textContent = msg;
    askErrorEl.hidden = false;
  }

  // ── Public API ─────────────────────────────────────────────
  return { onSelectionChange };
}
```

- [ ] **Step 2: Commit**

```bash
git add api/static/query.js
git commit -m "feat: query.js — query bar, chat, Ask form"
```

---

## Task 9: Write `app.js` and clean up

**Files:**
- Replace: `api/static/app.js`
- Delete: `api/static/examples.json`

- [ ] **Step 1: Replace `api/static/app.js`**

```javascript
// app.js — thin coordinator: wires sidebar and query via shared selectedIds

import { initSidebar } from './sidebar.js';
import { initQuery } from './query.js';

const selectedIds = new Set();

const query = initQuery({
  getSelectedIds: () => new Set(selectedIds),
});

const sidebar = initSidebar({
  onSelectionChange: (ids) => {
    selectedIds.clear();
    ids.forEach((id) => selectedIds.add(id));
    query.onSelectionChange(new Set(selectedIds));
  },
});

// Load documents that already exist on the server (e.g. from previous runs)
fetch('/documents')
  .then((r) => r.json())
  .then(({ documents }) => sidebar.loadExisting(documents || []))
  .catch(() => {}); // server may not have any yet — that's fine
```

- [ ] **Step 2: Delete `examples.json`**

```bash
git rm api/static/examples.json
```

- [ ] **Step 3: Commit**

```bash
git add api/static/app.js
git commit -m "feat: app.js coordinator, remove examples.json"
```

---

## Task 10: Run full test suite and smoke test

- [ ] **Step 1: Run all tests**

```bash
pytest tests/unit/ -v --ignore=tests/unit/test_services.py --ignore=tests/unit/test_vlm.py
```
Expected: all PASSED (services/vlm are skipped because they require Paddle)

- [ ] **Step 2: Start the server and verify the UI loads**

```bash
python main.py --serve
```

Open `http://localhost:8000`. Verify:
- Two-panel layout renders (sidebar on left, main panel on right)
- No JS console errors
- Drag-and-drop zone is visible
- "No documents selected" appears in query bar
- Ask button is disabled

- [ ] **Step 3: Verify `/documents` endpoint returns empty list**

```bash
curl http://localhost:8000/documents
```
Expected: `{"documents": []}`

- [ ] **Step 4: Verify `/query` with `document_ids` filter is accepted**

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "test", "top_k": 2, "document_ids": ["doc_a"]}'
```
Expected: JSON response (error about no index is fine — the endpoint accepts the request)

- [ ] **Step 5: Final commit**

```bash
git add -A
git commit -m "feat: complete PDF RAG webapp — two-panel UI with auto-pipeline"
```
