// conversations.js — sidebar chat history list

/**
 * Manages the conversation list in the sidebar.
 *
 * @param {{ authedFetch: Function, onSelect: (messages: Array) => void }} opts
 * @returns {{ refresh: () => void, markActive: (id: string) => void }}
 */
export function initConversations({ authedFetch, onSelect }) {
  const listEl = document.getElementById('convo-list');
  const emptyEl = document.getElementById('convo-list-empty');

  let activeId = null;

  // ── Load + render ─────────────────────────────────────────
  async function refresh() {
    let data;
    try {
      const r = await authedFetch('/conversations');
      data = await r.json();
    } catch {
      return;
    }

    const convos = data.conversations || [];
    listEl.innerHTML = '';

    if (convos.length === 0) {
      emptyEl.hidden = false;
      listEl.appendChild(emptyEl);
      return;
    }

    emptyEl.hidden = true;

    convos.forEach((convo) => {
      const li = document.createElement('li');
      li.className = 'convo-card';
      if (convo.id === activeId) li.classList.add('convo-card--active');
      li.dataset.id = convo.id;

      const title = document.createElement('div');
      title.className = 'convo-card__title';
      title.textContent = convo.title || 'Untitled';

      const preview = document.createElement('div');
      preview.className = 'convo-card__preview';
      preview.textContent = convo.last_message
        ? (convo.last_role === 'assistant' ? 'AI: ' : 'You: ') + convo.last_message
        : '—';

      const date = document.createElement('div');
      date.className = 'convo-card__date';
      date.textContent = formatDate(convo.created_at);

      const deleteBtn = document.createElement('button');
      deleteBtn.className = 'convo-card__delete';
      deleteBtn.title = 'Delete conversation';
      deleteBtn.textContent = '✕';
      deleteBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        deleteConvo(convo.id, li);
      });

      li.appendChild(title);
      li.appendChild(preview);
      li.appendChild(date);
      li.appendChild(deleteBtn);

      li.addEventListener('click', () => loadConvo(convo.id, li));
      listEl.appendChild(li);
    });
  }

  // ── Load a conversation's messages into the chat panel ────
  async function loadConvo(id) {
    let data;
    try {
      const r = await authedFetch(`/conversations/${id}/messages`);
      if (!r.ok) return;
      data = await r.json();
    } catch {
      return;
    }

    activeId = id;
    listEl.querySelectorAll('.convo-card').forEach((el) => {
      el.classList.toggle('convo-card--active', el.dataset.id === id);
    });

    // Pass both the id and messages so app.js can set the active conversation
    onSelect(id, data.messages || []);
  }

  // ── Delete ────────────────────────────────────────────────
  async function deleteConvo(id, li) {
    if (!confirm('Delete this conversation?')) return;
    try {
      await authedFetch(`/conversations/${id}`, { method: 'DELETE' });
    } catch {
      return;
    }
    li.remove();
    if (activeId === id) {
      activeId = null;
      onSelect([]);
    }
    if (listEl.querySelectorAll('.convo-card').length === 0) {
      emptyEl.hidden = false;
      listEl.appendChild(emptyEl);
    }
  }

  // ── Helpers ───────────────────────────────────────────────
  function formatDate(iso) {
    const d = new Date(iso);
    const now = new Date();
    const diffDays = Math.floor((now - d) / 86400000);
    if (diffDays === 0) return 'Today';
    if (diffDays === 1) return 'Yesterday';
    if (diffDays < 7) return `${diffDays} days ago`;
    return d.toLocaleDateString();
  }

  // ── Public API ────────────────────────────────────────────
  return {
    refresh,
    markActive(id) {
      activeId = id;
      listEl.querySelectorAll('.convo-card').forEach((el) => {
        el.classList.toggle('convo-card--active', el.dataset.id === id);
      });
    },
  };
}
