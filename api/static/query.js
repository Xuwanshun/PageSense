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
