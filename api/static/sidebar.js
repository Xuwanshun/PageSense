// sidebar.js — document library: upload, polling, doc list, selection

const POLL_INTERVAL_MS = 3000;

/**
 * @param {{ onSelectionChange: (ids: Set<string>) => void }} opts
 * @returns {{ loadExisting: (docs: Array) => void }}
 */
export function initSidebar({ onSelectionChange, authedFetch }) {
  const selectedIds = new Set();
  const selectedNames = new Map(); // document_id -> source_filename
  const pollingTimers = new Map(); // document_id -> intervalId
  const docCards = new Map();       // document_id -> li element
  const docNames = new Map();       // document_id -> source_filename

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

    authedFetch('/documents/upload', { method: 'POST', body: formData })
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
    authedFetch(`/documents/status/${document_id}`)
      .then((r) => r.json())
      .then(({ status, error, chunk_count, page_count, pages_done, total_pages }) => {
        const card = docCards.get(document_id);
        if (!card) return;
        updateCardStatus(card, document_id, status, chunk_count, page_count, error, pages_done, total_pages);
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
      updateCardStatus(docCards.get(document_id), document_id, status, chunk_count, page_count, null, null, null);
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
        selectedNames.set(document_id, docNames.get(document_id) || document_id);
        li.classList.add('doc-card--selected');
      } else {
        selectedIds.delete(document_id);
        selectedNames.delete(document_id);
        li.classList.remove('doc-card--selected');
      }
      updateSummary();
      onSelectionChange(new Set(selectedIds), new Map(selectedNames));
    });

    listEl.appendChild(li);
    docCards.set(document_id, li);
    docNames.set(document_id, source_filename);

    if (status === 'ready') enableCard(li, document_id);
    updateCardStatus(li, document_id, status, chunk_count, page_count, null, null, null);
  }

  function updateCardStatus(li, _document_id, status, chunk_count, page_count, error, pages_done, total_pages) {
    const pipeline = li.querySelector('.doc-card__pipeline');

    if (status === 'preprocessing') {
      const progress = (pages_done != null && total_pages != null && total_pages > 0)
        ? ` ${pages_done} / ${total_pages} pages`
        : '…';
      if (pipeline) pipeline.innerHTML = `
        <div class="pipeline-step pipeline-step--active">⟳ Preprocessing${progress}</div>
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
    selectedNames.set(document_id, docNames.get(document_id) || document_id);
    li.classList.add('doc-card--selected');
    updateSummary();
    onSelectionChange(new Set(selectedIds), new Map(selectedNames));
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
