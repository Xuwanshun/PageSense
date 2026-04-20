import { initSidebar } from './sidebar.js';
import { initQuery } from './query.js';

// Access token lives here only — never in localStorage or sessionStorage
let _token = null;

export async function authedFetch(url, options = {}) {
  const headers = { ...(options.headers || {}) };
  if (_token) headers['Authorization'] = `Bearer ${_token}`;
  const r = await fetch(url, { ...options, headers });
  if (r.status === 401) {
    const ok = await tryRefresh();
    if (ok) {
      headers['Authorization'] = `Bearer ${_token}`;
      return fetch(url, { ...options, headers });
    }
    redirectToLogin();
  }
  return r;
}

async function tryRefresh() {
  try {
    const r = await fetch('/auth/refresh', { method: 'POST' });
    if (!r.ok) return false;
    _token = (await r.json()).access_token;
    return true;
  } catch {
    return false;
  }
}

function redirectToLogin() {
  _token = null;
  window.location.href = '/login';
}

async function init() {
  const once = sessionStorage.getItem('__auth_token_once');
  if (once) {
    sessionStorage.removeItem('__auth_token_once');
    _token = once;
  } else {
    const ok = await tryRefresh();
    if (!ok) { redirectToLogin(); return; }
  }

  const selectedIds = new Set();

  const query = initQuery({ getSelectedIds: () => new Set(selectedIds), authedFetch });

  const sidebar = initSidebar({
    onSelectionChange: (ids, names) => {
      selectedIds.clear();
      ids.forEach((id) => selectedIds.add(id));
      query.onSelectionChange(new Set(selectedIds), names);
    },
    authedFetch,
  });

  authedFetch('/documents')
    .then((r) => r.json())
    .then(({ documents }) => sidebar.loadExisting(documents || []))
    .catch(() => {});

  const logoutBtn = document.getElementById('logout-btn');
  if (logoutBtn) {
    logoutBtn.addEventListener('click', async () => {
      await fetch('/auth/logout', { method: 'POST' });
      redirectToLogin();
    });
  }
}

init();
