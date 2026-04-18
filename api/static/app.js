// app.js — thin coordinator: wires sidebar and query via shared selectedIds

import { initSidebar } from './sidebar.js';
import { initQuery } from './query.js';

const selectedIds = new Set();

const query = initQuery({
  getSelectedIds: () => new Set(selectedIds),
});

const sidebar = initSidebar({
  onSelectionChange: (ids, names) => {
    selectedIds.clear();
    ids.forEach((id) => selectedIds.add(id));
    query.onSelectionChange(new Set(selectedIds), names);
  },
});

// Load documents that already exist on the server (e.g. from previous runs)
fetch('/documents')
  .then((r) => r.json())
  .then(({ documents }) => sidebar.loadExisting(documents || []))
  .catch(() => {}); // server may not have any yet — that's fine
