# Preprocessing Page Progress Design

**Date:** 2026-05-23
**Status:** Approved

## Goal

Surface per-page preprocessing progress to the frontend so users see "⟳ Preprocessing… 40 / 500 pages" instead of a static spinner during long uploads.

## Architecture

Progress flows through four layers without coupling them:

```
OCRService (per-page signal)
  → pipeline.run() (maintains counter, calls on_progress callback)
    → _run_pipeline (writes pages_done/total_pages into jobs dict)
      → GET /documents/status (includes fields in response)
        → sidebar.js (renders counter in card)
```

No extra compute — OCRService already iterates pages one at a time. The signal is a no-arg callable added as an optional parameter.

## Components

### 1. `document_process/services.py` — `OCRService.extract()`

Add `on_page_done: Callable[[], None] | None = None` as a keyword-only parameter. After each page is processed (inside the existing `for page in pages` loop), call `on_page_done()` if provided. Default `None` means no-op — all existing call sites work unchanged.

```python
def extract(
    self,
    pages: list[PageContext],
    *,
    on_page_done: Callable[[], None] | None = None,
) -> tuple[list[OCRPageResult], list[ProcessingIssue]]:
    ...
    for page in pages:
        # existing OCR logic
        ...
        if on_page_done:
            on_page_done()
```

### 2. `document_process/pipeline.py` — `DocumentPreprocessingPipeline.run()`

Add `on_progress: Callable[[int, int], None] | None = None` as a keyword-only parameter. The pipeline owns the absolute counter and wires the signal:

```python
pages_done = 0

def _on_page_done() -> None:
    nonlocal pages_done
    pages_done += 1
    if on_progress:
        on_progress(pages_done, total_pages)

# Pass signal into each batch's OCR call:
ocr_batch, ocr_issues = self.ocr.extract(batch, on_page_done=_on_page_done)
```

The `on_progress(pages_done, total_pages)` callback is called once per page during the OCR phase. Layout and association don't emit progress (they're fast relative to OCR).

`preprocess_document()` (the module-level convenience function) forwards `on_progress` through to `pipeline.run()`.

### 3. `api/routers/documents.py` — `_run_pipeline`

Provide the callback that writes into the in-memory jobs dict:

```python
def on_progress(pages_done: int, total_pages: int) -> None:
    jobs[document_id]["pages_done"] = pages_done
    jobs[document_id]["total_pages"] = total_pages

result = preprocess_document(dest, settings=settings, force=True,
                             document_id=document_id,
                             on_progress=on_progress)
```

Update `GET /documents/status/{document_id}` to include `pages_done` and `total_pages` in the response when the job is in memory:

```json
{
  "document_id": "...",
  "status": "preprocessing",
  "pages_done": 40,
  "total_pages": 500,
  "error": null,
  "chunk_count": null,
  "page_count": null
}
```

Both fields are `null` when not yet known (before the first page completes) and when status is `indexing` or `ready`.

### 4. `api/static/sidebar.js`

`poll()` already destructures the status response — add `pages_done` and `total_pages`:

```js
.then(({ status, error, chunk_count, page_count, pages_done, total_pages }) => {
```

In `updateCardStatus`, update the preprocessing branch to show the counter when available:

```js
if (status === 'preprocessing') {
  const progress = (pages_done != null && total_pages != null)
    ? ` ${pages_done} / ${total_pages} pages`
    : '…';
  if (pipeline) pipeline.innerHTML = `
    <div class="pipeline-step pipeline-step--active">⟳ Preprocessing${progress}</div>
    <div class="pipeline-step pipeline-step--pending">○ Indexing</div>
  `;
}
```

`updateCardStatus` must also accept `pages_done` and `total_pages` as parameters — update its signature and all call sites (`addOrUpdateCard`, `poll`).

## Data Flow

1. User uploads PDF → `POST /documents/upload` returns `{status: "preprocessing"}`
2. Frontend starts polling `GET /documents/status/{id}` every 3 s
3. Background thread runs `_run_pipeline` → `preprocess_document` → `pipeline.run()`
4. After each page's OCR: `on_page_done()` → `on_progress(n, total)` → `jobs[id]["pages_done"] = n`
5. Next poll returns `{status: "preprocessing", pages_done: 25, total_pages: 500}`
6. Frontend renders "⟳ Preprocessing 25 / 500 pages"
7. On completion: `jobs[id]["status"] = "indexing"` → frontend shows indexing step
8. On ready: counter disappears, "● Ready · N chunks · M pages" shown

## Error Handling

- `on_page_done` exceptions must not kill the pipeline — wrap the call: `try: on_page_done() except Exception: pass`
- If `total_pages` is 0 (empty PDF edge case), omit the counter entirely rather than showing "0 / 0 pages"
- `pages_done` and `total_pages` absent from response → frontend falls back to "⟳ Preprocessing…" (no counter) — existing behaviour preserved

## Testing

- **`OCRService`**: pass a mock `on_page_done`, assert it was called once per page
- **`pipeline.run()`**: pass a mock `on_progress`, assert it was called with correct `(pages_done, total_pages)` values as pages increment, using the existing mock-service test harness from `test_pipeline_batching.py`
- **`_run_pipeline`**: assert `jobs[document_id]` gains `pages_done`/`total_pages` keys after the callback fires — use a fake `preprocess_document` that calls `on_progress` once
- **`document_status` endpoint**: assert response includes `pages_done`/`total_pages` when job has those keys
- **No frontend unit tests** — JS rendering is verified manually

## Files Changed

| File | Change |
|------|--------|
| `document_process/services.py` | Add `on_page_done` param to `OCRService.extract()` |
| `document_process/pipeline.py` | Add `on_progress` param to `run()` and `preprocess_document()`; wire signal |
| `api/routers/documents.py` | Provide callback in `_run_pipeline`; add fields to status response |
| `api/static/sidebar.js` | Read and render `pages_done`/`total_pages` in card |
| `tests/unit/test_services.py` | Test `on_page_done` callback |
| `tests/unit/test_pipeline_batching.py` | Test `on_progress` callback |
| `tests/unit/test_documents_api.py` | Test status response includes new fields |
