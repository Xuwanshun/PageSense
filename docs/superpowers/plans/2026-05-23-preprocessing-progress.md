# Preprocessing Page Progress Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Show "⟳ Preprocessing… 40 / 500 pages" in the frontend document card as each page completes OCR, by threading a no-arg callback from OCRService up through the pipeline to the API jobs dict.

**Architecture:** `OCRService.extract()` gains an optional `on_page_done` no-arg callable called after each page. `pipeline.run()` wraps it in a closure that increments a counter and fires an `on_progress(pages_done, total_pages)` callback supplied by `_run_pipeline` in the API layer. The status endpoint forwards `pages_done`/`total_pages` from the jobs dict to the frontend, which renders the counter.

**Tech Stack:** Python (typing.Callable), FastAPI, vanilla JS

---

## File Map

| File | Change |
|------|--------|
| `document_process/services.py` | Add `on_page_done` param to `OCRService.extract()` |
| `document_process/pipeline.py` | Add `on_progress` param to `run()` and `preprocess_document()`; wire counter closure |
| `api/routers/documents.py` | Provide callback in `_run_pipeline`; include `pages_done`/`total_pages` in status response |
| `api/static/sidebar.js` | Read and render `pages_done`/`total_pages` in card |
| `tests/unit/test_services.py` | Test `on_page_done` called once per page |
| `tests/unit/test_pipeline_batching.py` | Test `on_progress` receives correct cumulative counts |
| `tests/unit/test_documents_api.py` | Test status response includes `pages_done`/`total_pages` |

---

## Task 1: Add `on_page_done` callback to `OCRService.extract()`

**Files:**
- Modify: `document_process/services.py:162`
- Test: `tests/unit/test_services.py`

- [ ] **Step 1: Write the failing test**

Add to the bottom of `tests/unit/test_services.py`:

```python
def test_ocr_extract_calls_on_page_done_once_per_page():
    """on_page_done must be called exactly once per page processed."""
    from unittest.mock import MagicMock
    from document_process.services import OCRService
    from document_process.models import PageContext
    from pathlib import Path

    # Build two fake PageContext objects — OCRService reads the image path,
    # so we need real files. Use tmp_path via a helper or just mock predict().
    import tempfile, os
    with tempfile.TemporaryDirectory() as d:
        paths = []
        for i in range(2):
            p = Path(d) / f"page_{i+1}.png"
            p.write_bytes(b"fake")
            paths.append(p)

        pages = [
            PageContext(page_number=i+1, width=100.0, height=200.0, page_image_path=paths[i])
            for i in range(2)
        ]

        svc = OCRService()
        # Patch _get_paddle_ocr so no real Paddle is needed
        fake_result = MagicMock()
        fake_result.json = {"res": {"rec_texts": [], "rec_scores": [], "rec_boxes": [], "dt_polys": []}}
        fake_predictor = MagicMock()
        fake_predictor.predict.return_value = [fake_result]

        callback = MagicMock()

        with patch("document_process.services._get_paddle_ocr", return_value=fake_predictor):
            svc.extract(pages, on_page_done=callback)

        assert callback.call_count == 2
```

Add `from unittest.mock import patch` at the top of the test file if not already imported (check first).

- [ ] **Step 2: Run to confirm it fails**

```bash
/Users/longzhuang/Documents/RAG-Agent-for-PDF-reading/.venv/bin/pytest tests/unit/test_services.py::test_ocr_extract_calls_on_page_done_once_per_page -v
```

Expected: `FAILED` — `extract()` does not accept `on_page_done`.

- [ ] **Step 3: Implement the change in `document_process/services.py`**

Change line 162: replace the `extract` signature and add the callback call after each page's `results.append(...)`:

```python
def extract(
    self,
    pages: list[PageContext],
    *,
    on_page_done: Callable[[], None] | None = None,
) -> tuple[list[OCRPageResult], list[ProcessingIssue]]:
    logger.info("Running PaddleOCR text extraction on %s page(s)", len(pages))
    ocr = _get_paddle_ocr()
    results: list[OCRPageResult] = []
    issues: list[ProcessingIssue] = []
    for page in pages:
        try:
            payload = ocr.predict(str(page.page_image_path))[0].json["res"]
        except Exception as exc:
            raise RuntimeError(
                f"PaddleOCR text extraction failed on page {page.page_number}: {type(exc).__name__}: {exc}"
            ) from exc

        items: list[OCRTextItem] = []
        rec_texts = payload.get("rec_texts") or []
        rec_scores = payload.get("rec_scores") or []
        rec_boxes = payload.get("rec_boxes") or []
        dt_polys = payload.get("dt_polys") or []
        for index, text in enumerate(rec_texts, start=1):
            cleaned = str(text).strip()
            if not cleaned:
                continue
            bbox = _bbox_from_ocr_payload(rec_boxes, dt_polys, index - 1)
            if bbox is None or not bbox.is_valid():
                continue
            score = rec_scores[index - 1] if index - 1 < len(rec_scores) else None
            items.append(
                OCRTextItem(
                    item_id=f"p{page.page_number}_ocr_{index}",
                    page_number=page.page_number,
                    text=cleaned,
                    bbox=bbox,
                    confidence=float(score) if score is not None else None,
                    source="paddleocr",
                )
            )

        if not items:
            issues.append(
                ProcessingIssue(
                    code="ocr_no_text",
                    message="PaddleOCR did not return any text for this page.",
                    level="warning",
                    page_number=page.page_number,
                )
            )

        results.append(
            OCRPageResult(
                page_number=page.page_number,
                width=page.width,
                height=page.height,
                items=items,
                text_source="paddleocr_ppocrv5_mobile",
                page_image_path=str(page.page_image_path),
            )
        )
        try:
            if on_page_done is not None:
                on_page_done()
        except Exception:
            pass
    return results, issues
```

`Callable` is already imported at the top of `services.py` via `from typing import ...` — verify and add if missing.

- [ ] **Step 4: Run the new test**

```bash
/Users/longzhuang/Documents/RAG-Agent-for-PDF-reading/.venv/bin/pytest tests/unit/test_services.py::test_ocr_extract_calls_on_page_done_once_per_page -v
```

Expected: `PASSED`.

- [ ] **Step 5: Run full suite to confirm no regressions**

```bash
/Users/longzhuang/Documents/RAG-Agent-for-PDF-reading/.venv/bin/pytest tests/unit/ 2>&1 | tail -5
```

Expected: same pass count as before (222 passed, 1 pre-existing failure in `test_settings_defaults`).

- [ ] **Step 6: Commit**

```bash
git add document_process/services.py tests/unit/test_services.py
git commit -m "feat: add on_page_done callback to OCRService.extract for progress tracking"
```

---

## Task 2: Wire `on_progress` callback through `pipeline.run()` and `preprocess_document()`

**Files:**
- Modify: `document_process/pipeline.py:65` (`run` signature), `pipeline.py:95` (ocr call), `pipeline.py:256` (`preprocess_document` signature and last line)
- Test: `tests/unit/test_pipeline_batching.py`

- [ ] **Step 1: Write the failing test**

Add to the bottom of `tests/unit/test_pipeline_batching.py`:

```python
@patch("document_process.pipeline.build_chunks", return_value=[])
@patch("document_process.pipeline.build_visual_summaries", return_value=[])
@patch("document_process.pipeline.build_document_artifacts")
@patch("document_process.pipeline.export_artifacts")
def test_on_progress_receives_cumulative_page_counts(
    mock_export, mock_build_doc, mock_vis, mock_chunks, tmp_settings, tmp_path
):
    """on_progress must be called with (pages_done, total_pages) for each page,
    accumulating across batches."""
    mock_export.return_value = tmp_path / "processed" / "document.json"
    (tmp_path / "processed").mkdir(parents=True, exist_ok=True)
    mock_build_doc.return_value = (MagicMock(), MagicMock())

    settings = tmp_settings(preprocess_page_batch_size=2)
    pipeline = _build_pipeline(settings, tmp_path, num_pages=4)

    progress_calls = []
    pipeline.run(
        tmp_path / "test.pdf",
        document_id="test-doc",
        on_progress=lambda done, total: progress_calls.append((done, total)),
    )

    # 4 pages → on_progress called 4 times
    assert len(progress_calls) == 4
    # pages_done increments from 1 to 4
    assert [c[0] for c in progress_calls] == [1, 2, 3, 4]
    # total_pages is always 4
    assert all(c[1] == 4 for c in progress_calls)
```

Note: `_build_pipeline` mock's `ocr_mock.extract` uses a side effect that returns ocr pages for the batch. To make `on_page_done` fire, the mock must call it. Update `_build_pipeline`'s `ocr_extract` side effect:

```python
def ocr_extract(batch_pages, *, on_page_done=None):
    nums = [p.page_number for p in batch_pages]
    if on_page_done is not None:
        for _ in batch_pages:
            on_page_done()
    return [o for o in ocr_pages if o.page_number in nums], []
ocr_mock.extract.side_effect = ocr_extract
```

Update the `_build_pipeline` function's `ocr_extract` side effect to match this signature (it currently only takes `batch_pages`). Make this change to `_build_pipeline` now, before writing the test.

- [ ] **Step 2: Run to confirm it fails**

```bash
/Users/longzhuang/Documents/RAG-Agent-for-PDF-reading/.venv/bin/pytest tests/unit/test_pipeline_batching.py::test_on_progress_receives_cumulative_page_counts -v
```

Expected: `FAILED` — `run()` does not accept `on_progress`.

- [ ] **Step 3: Implement in `document_process/pipeline.py`**

**Change 1:** Update `run()` signature at line 65:

```python
def run(
    self,
    source_path: Path,
    *,
    document_id: str | None = None,
    on_progress: Callable[[int, int], None] | None = None,
) -> PreprocessingResult:
```

Add `from collections.abc import Callable` or `from typing import Callable` at the top of `pipeline.py` if `Callable` is not already imported. Check the existing imports first.

**Change 2:** Add the counter closure inside `run()`, just before the batch loop (after `next_block_index = 1`):

```python
        pages_done = 0

        def _on_page_done() -> None:
            nonlocal pages_done
            pages_done += 1
            if on_progress is not None and total_pages > 0:
                on_progress(pages_done, total_pages)
```

**Change 3:** Update the ocr call at line 95 to pass the signal:

```python
            ocr_batch, ocr_issues = self.ocr.extract(batch, on_page_done=_on_page_done)
```

**Change 4:** Update `preprocess_document()` signature at line 256 to accept and forward `on_progress`:

```python
def preprocess_document(
    source_name_or_path: str | Path,
    *,
    settings: Settings | None = None,
    document_id: str | None = None,
    force: bool = False,
    on_progress: Callable[[int, int], None] | None = None,
) -> PreprocessingResult:
```

**Change 5:** Update the last line of `preprocess_document()` at line 285 to forward it:

```python
    pipeline = DocumentPreprocessingPipeline(resolved_settings, loader=loader)
    return pipeline.run(source_path, document_id=resolved_id, on_progress=on_progress)
```

- [ ] **Step 4: Run the new test**

```bash
/Users/longzhuang/Documents/RAG-Agent-for-PDF-reading/.venv/bin/pytest tests/unit/test_pipeline_batching.py::test_on_progress_receives_cumulative_page_counts -v
```

Expected: `PASSED`.

- [ ] **Step 5: Run full suite**

```bash
/Users/longzhuang/Documents/RAG-Agent-for-PDF-reading/.venv/bin/pytest tests/unit/ 2>&1 | tail -5
```

Expected: same pass count + 1 new test passing.

- [ ] **Step 6: Commit**

```bash
git add document_process/pipeline.py tests/unit/test_pipeline_batching.py
git commit -m "feat: add on_progress callback to pipeline.run() and preprocess_document()"
```

---

## Task 3: Wire callback in `_run_pipeline` and expose fields in status endpoint

**Files:**
- Modify: `api/routers/documents.py:51` (`_run_pipeline` body), `api/routers/documents.py:294` (status response)
- Test: `tests/unit/test_documents_api.py`

- [ ] **Step 1: Write the failing tests**

Add to the bottom of `tests/unit/test_documents_api.py`:

```python
def test_status_includes_pages_done_when_preprocessing(client):
    """Status response must include pages_done and total_pages from the jobs dict."""
    app = client.app
    app.state.jobs["progressing_doc"] = {
        "status": "preprocessing",
        "error": None,
        "chunk_count": None,
        "page_count": None,
        "pages_done": 40,
        "total_pages": 500,
        "source_filename": "big.pdf",
    }
    r = client.get("/documents/status/progressing_doc")
    assert r.status_code == 200
    body = r.json()
    assert body["pages_done"] == 40
    assert body["total_pages"] == 500


def test_status_pages_done_null_when_not_set(client):
    """pages_done and total_pages must be null when not yet written to jobs dict."""
    app = client.app
    app.state.jobs["new_doc"] = {
        "status": "preprocessing",
        "error": None,
        "chunk_count": None,
        "page_count": None,
        "source_filename": "new.pdf",
    }
    r = client.get("/documents/status/new_doc")
    assert r.status_code == 200
    body = r.json()
    assert body["pages_done"] is None
    assert body["total_pages"] is None
```

- [ ] **Step 2: Run to confirm they fail**

```bash
/Users/longzhuang/Documents/RAG-Agent-for-PDF-reading/.venv/bin/pytest tests/unit/test_documents_api.py::test_status_includes_pages_done_when_preprocessing tests/unit/test_documents_api.py::test_status_pages_done_null_when_not_set -v
```

Expected: `FAILED` — status response does not yet include these fields.

- [ ] **Step 3: Update `_run_pipeline` in `api/routers/documents.py`**

Add the callback definition inside `_run_pipeline`, just before the `preprocess_document` call (line 51). Replace:

```python
            jobs[document_id]["status"] = "preprocessing"
            result = preprocess_document(dest, settings=settings, force=True, document_id=document_id)
```

With:

```python
            jobs[document_id]["status"] = "preprocessing"

            def _on_progress(pages_done: int, total_pages: int) -> None:
                jobs[document_id]["pages_done"] = pages_done
                jobs[document_id]["total_pages"] = total_pages

            result = preprocess_document(
                dest,
                settings=settings,
                force=True,
                document_id=document_id,
                on_progress=_on_progress,
            )
```

- [ ] **Step 4: Update the status response in `api/routers/documents.py`**

Find the `document_status` endpoint at line 294. Replace the `return JSONResponse(...)` block inside the `if document_id in jobs:` branch:

```python
        return JSONResponse(
            {
                "document_id": document_id,
                "status": job["status"],
                "error": job.get("error"),
                "chunk_count": job.get("chunk_count"),
                "page_count": job.get("page_count"),
                "pages_done": job.get("pages_done"),
                "total_pages": job.get("total_pages"),
            }
        )
```

- [ ] **Step 5: Run the new tests**

```bash
/Users/longzhuang/Documents/RAG-Agent-for-PDF-reading/.venv/bin/pytest tests/unit/test_documents_api.py::test_status_includes_pages_done_when_preprocessing tests/unit/test_documents_api.py::test_status_pages_done_null_when_not_set -v
```

Expected: both `PASSED`.

- [ ] **Step 6: Run full suite**

```bash
/Users/longzhuang/Documents/RAG-Agent-for-PDF-reading/.venv/bin/pytest tests/unit/ 2>&1 | tail -5
```

Expected: same pass count + 2 new tests.

- [ ] **Step 7: Commit**

```bash
git add api/routers/documents.py tests/unit/test_documents_api.py
git commit -m "feat: expose pages_done/total_pages in status endpoint and wire progress callback"
```

---

## Task 4: Render page progress in the frontend card

**Files:**
- Modify: `api/static/sidebar.js`

No automated tests — verify manually by running the dev server.

- [ ] **Step 1: Update `poll()` to destructure new fields**

In `sidebar.js` at the `.then(...)` block inside `function poll(document_id)` (around line 75), change:

```js
.then(({ status, error, chunk_count, page_count }) => {
    if (!card) return;
    updateCardStatus(card, document_id, status, chunk_count, page_count, error);
```

To:

```js
.then(({ status, error, chunk_count, page_count, pages_done, total_pages }) => {
    if (!card) return;
    updateCardStatus(card, document_id, status, chunk_count, page_count, error, pages_done, total_pages);
```

- [ ] **Step 2: Update `addOrUpdateCard` call sites**

Search `sidebar.js` for all calls to `updateCardStatus` and `addOrUpdateCard`. Add `pages_done` and `total_pages` parameters where missing, passing `null` as default:

In `addOrUpdateCard` (around line 90–135):

```js
function addOrUpdateCard(document_id, source_filename, status, chunk_count, page_count) {
    if (docCards.has(document_id)) {
        updateCardStatus(docCards.get(document_id), document_id, status, chunk_count, page_count, null, null, null);
```

In `initDocumentList` where it calls `updateCardStatus` for existing docs, add `null, null` for the two new params.

- [ ] **Step 3: Update `updateCardStatus` signature and preprocessing branch**

Change the function signature (around line 135):

```js
function updateCardStatus(li, _document_id, status, chunk_count, page_count, error, pages_done, total_pages) {
```

Update the `preprocessing` branch inside `updateCardStatus`:

```js
if (status === 'preprocessing') {
  const progress = (pages_done != null && total_pages != null && total_pages > 0)
    ? ` ${pages_done} / ${total_pages} pages`
    : '…';
  if (pipeline) pipeline.innerHTML = `
    <div class="pipeline-step pipeline-step--active">⟳ Preprocessing${progress}</div>
    <div class="pipeline-step pipeline-step--pending">○ Indexing</div>
  `;
}
```

- [ ] **Step 4: Manual verification**

Start the API server locally (requires a valid `.env`):

```bash
/Users/longzhuang/Documents/RAG-Agent-for-PDF-reading/.venv/bin/python main.py --serve
```

Upload a PDF and watch the card. It should show:
- Initially: "⟳ Preprocessing…" (before first page completes)
- After each poll: "⟳ Preprocessing 25 / N pages", "⟳ Preprocessing 50 / N pages", etc.
- After OCR phase: "⟳ Indexing…"
- When done: "● Ready · X chunks · Y pages"

If Paddle is not installed locally, verify by injecting a fake job directly:

Open browser console on the running frontend and run:

```js
fetch('/documents/status/fake', {headers: {'Authorization': 'Bearer <your-token>'}})
```

Or by directly setting the jobs dict via the debug endpoint if available.

- [ ] **Step 5: Commit**

```bash
git add api/static/sidebar.js
git commit -m "feat: show pages_done/total_pages progress counter in preprocessing card"
```

---

## Self-Review

**Spec coverage:**
- `OCRService.extract()` `on_page_done` — Task 1 ✓
- `pipeline.run()` `on_progress` closure with counter — Task 2 ✓
- `preprocess_document()` forwards `on_progress` — Task 2 ✓
- `_run_pipeline` provides callback writing to jobs dict — Task 3 ✓
- Status endpoint returns `pages_done`/`total_pages` — Task 3 ✓
- Frontend renders counter — Task 4 ✓
- Error handling: callback wrapped in try/except — Task 1 Step 3 ✓
- Fallback when `pages_done` is null → "…" — Task 4 Step 3 ✓
- `total_pages == 0` guard — Task 2 Step 3 (`if total_pages > 0`) ✓

**Placeholder scan:** None found.

**Type consistency:**
- `on_page_done: Callable[[], None] | None` — Tasks 1 and 2 match
- `on_progress: Callable[[int, int], None] | None` — Tasks 2 and 3 match
- `pages_done: int`, `total_pages: int` — Tasks 2, 3, 4 consistent
