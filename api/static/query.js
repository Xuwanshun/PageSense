// query.js — query bar sync, chat interface, Ask form

/**
 * @param {{ getSelectedIds: () => Set<string>, authedFetch: Function,
 *           onConversationCreated: (id: string) => void }} opts
 */
export function initQuery({ getSelectedIds, authedFetch, onConversationCreated }) {
  const queryTagsEl = document.getElementById('query-tags');
  const chatHistoryEl = document.getElementById('chat-history');
  const questionInput = document.getElementById('question-input');
  const askBtn = document.getElementById('ask-btn');
  const askErrorEl = document.getElementById('ask-error');

  // Tracks the current conversation so follow-up questions append to it.
  // Reset to null by startNewChat() or when a history item is loaded.
  let currentConversationId = null;

  // ── Query bar ──────────────────────────────────────────────
  function updateQueryBar(ids, names) {
    queryTagsEl.innerHTML = '';
    if (ids.size === 0) {
      queryTagsEl.innerHTML = '<span class="query-bar__empty">No documents selected</span>';
      askBtn.disabled = true;
      return;
    }
    ids.forEach((id) => {
      const tag = document.createElement('span');
      tag.className = 'query-tag';
      tag.textContent = (names && names.get(id)) || id;
      queryTagsEl.appendChild(tag);
    });
    askBtn.disabled = false;
  }

  function onSelectionChange(ids, names) {
    updateQueryBar(ids, names);
  }

  // ── Render helpers ─────────────────────────────────────────
  function renderUserBubble(content) {
    const div = document.createElement('div');
    div.className = 'chat-question';
    div.textContent = content;
    return div;
  }

  function renderAssistantBubble(content, sources) {
    const aWrap = document.createElement('div');
    aWrap.className = 'chat-answer';

    const aText = document.createElement('div');
    aText.className = 'chat-answer__text';
    aText.textContent = content || '(no answer)';
    aWrap.appendChild(aText);

    if (sources && sources.length) {
      const sourcesEl = document.createElement('div');
      sourcesEl.className = 'chat-answer__sources';
      sources.forEach((src) => {
        const chip = document.createElement('span');
        chip.className = 'source-chip';
        const file = src.source_filename || src.document_id || '?';
        const page = src.page_number != null ? ` p.${src.page_number}` : '';
        chip.textContent = `${file}${page}`;
        sourcesEl.appendChild(chip);
      });
      aWrap.appendChild(sourcesEl);
    }
    return aWrap;
  }

  function appendTurn(question, result) {
    const empty = chatHistoryEl.querySelector('.chat-history__empty');
    if (empty) empty.remove();

    const turn = document.createElement('div');
    turn.className = 'chat-turn';
    turn.appendChild(renderUserBubble(question));
    turn.appendChild(renderAssistantBubble(result.answer, result.sources));
    chatHistoryEl.appendChild(turn);
    chatHistoryEl.scrollTop = chatHistoryEl.scrollHeight;
  }

  // ── Load a historical conversation into the chat panel ─────
  // Called by conversations.js when the user clicks a history item.
  function loadConversation(msgs) {
    chatHistoryEl.innerHTML = '';
    currentConversationId = null;

    if (!msgs || msgs.length === 0) {
      chatHistoryEl.innerHTML = '<p class="chat-history__empty">Select documents on the left, then ask a question.</p>';
      return;
    }

    // Pair up user + assistant messages into turns
    for (let i = 0; i < msgs.length; i++) {
      const msg = msgs[i];
      if (msg.role === 'user') {
        const next = msgs[i + 1];
        const turn = document.createElement('div');
        turn.className = 'chat-turn';
        turn.appendChild(renderUserBubble(msg.content));
        if (next && next.role === 'assistant') {
          turn.appendChild(renderAssistantBubble(next.content, next.sources));
          i++; // skip the assistant message — already consumed
        }
        chatHistoryEl.appendChild(turn);
      }
    }

    chatHistoryEl.scrollTop = chatHistoryEl.scrollHeight;

    // Continue this conversation if the user asks another question
    if (msgs.length > 0) {
      // Find the conversation_id embedded in any assistant message's context.
      // We get it from the URL we used to fetch: store it via markActive.
      // app.js sets currentConversationId via setConversationId().
    }
  }

  // Called by app.js after selecting a conversation from the history list
  function setConversationId(id) {
    currentConversationId = id;
  }

  // Called by the "New Chat" button in app.js
  function startNewChat() {
    currentConversationId = null;
    chatHistoryEl.innerHTML = '<p class="chat-history__empty">Select documents on the left, then ask a question.</p>';
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
      const body = {
        question,
        top_k: 4,
        doc_filter: [...ids],
        conversation_id: currentConversationId,  // null = start new conversation
      };

      const r = await authedFetch('/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
      const data = await r.json();
      if (!r.ok) {
        showError(data.detail || 'Query failed.');
        return;
      }

      // Backend always returns the conversation_id (new or existing).
      // Store it so the next question appends to the same conversation.
      if (data.conversation_id && !currentConversationId) {
        currentConversationId = data.conversation_id;
        // Tell the sidebar to refresh the history list and highlight this convo.
        onConversationCreated(data.conversation_id);
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
  return { onSelectionChange, loadConversation, setConversationId, startNewChat };
}
