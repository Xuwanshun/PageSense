# PDF RAG Web App — Design Spec

**Date:** 2026-04-18  
**Branch:** feature/improvements

---

## Goal

Replace the existing static demo page with a functional two-panel web app where users can:
1. Upload PDFs and have them automatically preprocessed and indexed
2. Select which documents to query against
3. Ask questions and receive grounded answers with citations

---

## Layout

**Sidebar + main panel.** Fixed sidebar on the left manages the document library. Main panel on the right is the query interface.

The existing `index.html` / `app.js` / `styles.css` are replaced entirely. The demo/showcase content (architecture diagram, evaluation metrics) is removed.

---

## Backend Changes

### New: `POST /documents/upload`

Accepts a PDF via multipart form upload. Saves the file, then runs preprocess and index sequentially in a thread pool. Returns immediately with a `document_id` and initial status.

```
Request:  multipart/form-data  { file: <PDF> }
Response: { document_id: str, status: "preprocessing" }
```

Pipeline stages run in order:
1. Save file to `raw_documents_dir`
2. Run `preprocess_document()` (OCR + layout + chunking)
3. Run `index_all_processed_documents()` (embed + write vector store)

Status transitions: `preprocessing → indexing → ready` (or `error` on failure).

The existing `POST /documents/preprocess` and `POST /documents/index` are kept for CLI use but are no longer used by the frontend.

---

### New: `GET /documents/status/{document_id}`

Returns the current pipeline stage for a given document. Used by the frontend to poll every 3 seconds.

```
Response: {
  document_id: str,
  status: "preprocessing" | "indexing" | "ready" | "error",
  error: str | null,         // set only when status == "error"
  chunk_count: int | null,   // set only when status == "ready"
  page_count: int | null     // set only when status == "ready"
}
```

Status is derived by inspecting the filesystem:
- Processed artifacts missing → `preprocessing`
- Artifacts present, vector store not updated → `indexing`
- Artifacts present + vector store updated → `ready`

In-progress jobs are tracked in memory (a dict on `app.state`) so status reflects real-time pipeline state, not just filesystem snapshots.

---

### Modified: `POST /query`

Adds an optional `document_ids` filter. When provided, retrieval is scoped to chunks belonging to those documents only.

```
Request: {
  question: str,
  top_k: int = 4,
  document_ids: list[str] | null = null   // null = search all docs
}
```

The retrieval layer filters the vector store by `document_id` before similarity search. All other behaviour (reranking, specialist routing, answer synthesis) is unchanged.

---

## Frontend Changes

The static frontend is rebuilt as three focused modules:

### `sidebar.js`
- Renders the document library panel
- Handles drag-and-drop and click-to-browse upload
- Validates that only PDFs are uploaded (client-side, before sending)
- On upload: calls `POST /documents/upload`, adds doc to list with `preprocessing` status
- Polls `GET /documents/status/{id}` every 3s for each in-progress doc
- Stops polling when status is `ready` or `error`
- On `ready`: enables checkbox, auto-selects the doc, updates `selectedIds`
- On `error`: shows red error badge on the doc card
- Each checkbox change updates `selectedIds` and dispatches a `selectionchange` event

### `query.js`
- Listens for `selectionchange` events → updates the "Querying:" tag bar instantly
- Disables the Ask button when `selectedIds` is empty
- On submit: calls `POST /query` with `{question, top_k: 4, document_ids: [...selectedIds]}`
- Renders each exchange (question bubble + answer bubble + source chips) into the chat area
- On query error: shows inline error message below the input, preserves the question text

### `app.js`
- Thin coordinator: initialises sidebar and query modules, wires them via shared `selectedIds` state
- On page load: fetches existing documents from a new `GET /documents` list endpoint and populates the sidebar

### `index.html`
- Two-panel layout: fixed-width sidebar (260px) + flex-grow main panel
- Sidebar sections: header with upload zone, scrollable document list, footer with selection summary
- Main panel sections: "Querying:" tag bar, scrollable chat history, fixed input area at bottom

### `styles.css`
- Rewritten to match the new layout
- Dark theme consistent with current design language

---

## Document List Endpoint

### New: `GET /documents`

Returns all documents that have been processed (artifacts exist on disk), used to populate the sidebar on page load.

```
Response: {
  documents: [
    {
      document_id: str,
      source_filename: str,
      status: "ready" | "error",
      chunk_count: int,
      page_count: int
    }
  ]
}
```

---

## Error Handling

| Scenario | Behaviour |
|---|---|
| Non-PDF uploaded | Rejected client-side before request is sent |
| Upload fails (server error) | Error badge on doc card, polling stops |
| Preprocessing fails | Status → `error`, error message shown on card |
| Indexing fails | Status → `error`, error message shown on card |
| Query with no docs selected | Ask button disabled, cannot submit |
| Query fails (server error) | Inline error below input, question text preserved |
| Multiple concurrent uploads | Each doc tracked independently, no interference |

---

## What Is Not Changing

- FastAPI app factory pattern (`api/app.py`) — unchanged
- `api/routers/health.py` — unchanged
- `api/routers/documents.py` — kept for CLI, new `/upload`, `/status`, and `/` (list) endpoints added here
- `api/routers/query.py` — `document_ids` filter added, otherwise unchanged
- All backend pipeline code (`document_Process/`, `rag/`) — unchanged
- `config.py`, `storage/`, `main.py` — unchanged

## What Is Being Removed

- `api/routers/ui.py` — showcase data endpoint is gone; no longer needed
- `api/static/index.html` — replaced with new two-panel layout
- `api/static/app.js` — replaced with `app.js`, `sidebar.js`, `query.js`
- `api/static/styles.css` — replaced with rewritten stylesheet
- `api/static/examples.json` — no longer referenced
