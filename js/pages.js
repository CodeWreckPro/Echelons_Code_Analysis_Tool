// Configure these values to match your deployment
const VERCEL_ENDPOINT = 'https://echelons-vercel-api.vercel.app/api/dispatch';
const PAGES_BASE = window.location.origin + window.location.pathname.replace(/\/index\.html$/, '').replace(/\/$/, '');

function toOwnerRepo(url) {
  const u = url.replace(/\.git$/, '').replace(/\/$/, '');
  const m = u.match(/^https:\/\/github\.com\/([^\/]+)\/([^\/]+)$/i);
  return m ? { owner: m[1], repo: m[2] } : null;
}

async function dispatch(repoUrl) {
  const resp = await fetch(VERCEL_ENDPOINT, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ repo_url: repoUrl })
  });
  if (!resp.ok) throw new Error(await resp.text());
  return resp.json();
}

async function pollInsights(owner, repo, { tries = 60, intervalMs = 5000 } = {}) {
  const url = `${PAGES_BASE}/insights/${owner}/${repo}.json`;
  for (let i = 0; i < tries; i++) {
    const r = await fetch(url, { cache: 'no-store' });
    if (r.ok) {
      return r.json();
    }
    await new Promise(res => setTimeout(res, intervalMs));
  }
  throw new Error('Timed out waiting for insights.json');
}

async function pollSuggestions(owner, repo, { tries = 60, intervalMs = 5000 } = {}) {
  const url = `${PAGES_BASE}/insights/${owner}/${repo}-suggestions.json`;
  for (let i = 0; i < tries; i++) {
    const r = await fetch(url, { cache: 'no-store' });
    if (r.ok) {
      return r.json();
    }
    await new Promise(res => setTimeout(res, intervalMs));
  }
  throw new Error('Timed out waiting for suggestions.json');
}

// --- Interactive JSON Viewer ---
function createNode({ key, value, path, root = false }) {
  const node = document.createElement('div');
  node.className = 'json-node';
  node.setAttribute('role', 'treeitem');
  node.setAttribute('tabindex', '0');
  node.dataset.path = path.join('.');

  const type = value === null ? 'null' : Array.isArray(value) ? 'array' : typeof value;
  const hasChildren = type === 'object' || type === 'array';

  if (hasChildren) {
    const toggle = document.createElement('button');
    toggle.className = 'toggle';
    toggle.setAttribute('aria-label', 'Toggle');
    toggle.setAttribute('aria-expanded', String(root));
    toggle.textContent = root ? '−' : '+';
    node.appendChild(toggle);
    toggle.addEventListener('click', (e) => {
      e.stopPropagation();
      const expanded = toggle.getAttribute('aria-expanded') === 'true';
      toggle.setAttribute('aria-expanded', String(!expanded));
      toggle.textContent = expanded ? '+' : '−';
      children.hidden = expanded;
    });
  }

  const keyEl = document.createElement('span');
  keyEl.className = 'json-key';
  keyEl.textContent = key !== null ? `${key}` : (type === 'array' ? '[ ]' : '{ }');
  node.appendChild(keyEl);

  if (!hasChildren) {
    const delim = document.createElement('span');
    delim.className = 'json-delim';
    delim.textContent = key !== null ? ': ' : '';
    node.appendChild(delim);

    const valEl = document.createElement('span');
    const valType = value === null ? 'null' : typeof value;
    valEl.className = `json-value ${valType}`;
    valEl.textContent = typeof value === 'string' ? `"${value}"` : String(value);
    node.appendChild(valEl);
  }

  // Copy button for segment
  const copyBtn = document.createElement('button');
  copyBtn.className = 'copy-segment';
  copyBtn.textContent = 'Copy';
  copyBtn.setAttribute('aria-label', 'Copy segment');
  copyBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    try {
      const segment = serializeSegment(value);
      navigator.clipboard.writeText(segment);
    } catch {}
  });
  node.appendChild(copyBtn);

  let children = document.createElement('div');
  children.className = 'json-children';
  children.hidden = !root;
  if (hasChildren) {
    const entries = Array.isArray(value) ? value.map((v, i) => [String(i), v]) : Object.entries(value);
    for (const [k, v] of entries) {
      children.appendChild(createNode({ key: k, value: v, path: [...path, k] }));
    }
    node.appendChild(children);
  }

  // Keyboard navigation
  node.addEventListener('keydown', (e) => {
    const focusable = [...node.parentElement.querySelectorAll('.json-node')];
    const idx = focusable.indexOf(node);
    if (e.key === 'ArrowDown') {
      e.preventDefault();
      const next = focusable[idx + 1];
      if (next) next.focus();
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      const prev = focusable[idx - 1];
      if (prev) prev.focus();
    } else if (e.key === 'ArrowRight' && hasChildren) {
      const toggle = node.querySelector('.toggle');
      if (toggle && toggle.getAttribute('aria-expanded') === 'false') toggle.click();
    } else if (e.key === 'ArrowLeft' && hasChildren) {
      const toggle = node.querySelector('.toggle');
      if (toggle && toggle.getAttribute('aria-expanded') === 'true') toggle.click();
    }
  });

  return node;
}

function renderJsonViewer(container, data) {
  container.innerHTML = '';
  const root = createNode({ key: null, value: data, path: [], root: true });
  container.appendChild(root);
}

function highlightMatches(container, query) {
  const nodes = container.querySelectorAll('.json-node');
  nodes.forEach(n => n.classList.remove('highlight'));
  if (!query) return;
  const lower = query.toLowerCase();
  nodes.forEach(n => {
    const text = n.textContent.toLowerCase();
    if (text.includes(lower)) {
      n.classList.add('highlight');
      const toggle = n.querySelector('.toggle');
      const children = n.querySelector('.json-children');
      if (toggle && children && children.hidden) toggle.click();
    }
  });
}

function serializeSegment(value) {
  try {
    return typeof value === 'string' ? value : JSON.stringify(value, null, 2);
  } catch {
    return String(value);
  }
}

function setupPanel({ viewerId, rawId, toggleId, copyId, data, render }) {
  const viewer = document.getElementById(viewerId);
  const raw = document.getElementById(rawId);
  const toggle = document.getElementById(toggleId);
  const copy = document.getElementById(copyId);

  render(viewer, data);
  raw.textContent = JSON.stringify(data, null, 2);

  toggle.addEventListener('click', () => {
    const pressed = toggle.getAttribute('aria-pressed') === 'true';
    toggle.setAttribute('aria-pressed', String(!pressed));
    const showRaw = !pressed;
    raw.hidden = !showRaw;
    viewer.hidden = showRaw;
  });
  copy.addEventListener('click', () => navigator.clipboard.writeText(raw.textContent));
}

// --- Insights Tiles Renderer ---
function renderTileContent(value) {
  const box = document.createElement('div');
  if (Array.isArray(value)) {
    const count = document.createElement('div');
    count.className = 'badge';
    count.textContent = `${value.length} items`;
    box.appendChild(count);
    const ul = document.createElement('ul');
    ul.className = 'list';
    value.slice(0, 10).forEach((item) => {
      const li = document.createElement('li');
      li.textContent = typeof item === 'object' ? JSON.stringify(item) : String(item);
      ul.appendChild(li);
    });
    box.appendChild(ul);
  } else if (value && typeof value === 'object') {
    const entries = Object.entries(value);
    entries.slice(0, 12).forEach(([k, v]) => {
      const row = document.createElement('div');
      row.className = 'kv-item';
      const key = document.createElement('span');
      key.className = 'key';
      key.textContent = k;
      const val = document.createElement('span');
      val.className = 'val';
      val.textContent = typeof v === 'object' ? JSON.stringify(v) : String(v);
      row.appendChild(key);
      row.appendChild(val);
      box.appendChild(row);
    });
  } else {
    const p = document.createElement('div');
    p.textContent = typeof value === 'string' ? value : String(value);
    box.appendChild(p);
  }
  return box;
}

function createLabelCell(name) {
  const cell = document.createElement('div');
  cell.className = 'row-label';
  const tile = document.createElement('div');
  tile.className = 'label-tile';
  const vt = document.createElement('div');
  vt.className = 'vertical-text';
  vt.textContent = name;
  tile.appendChild(vt);
  cell.appendChild(tile);
  return cell;
}

function createTrackCell() {
  const cell = document.createElement('div');
  cell.className = 'row-track';
  return cell;
}

function addTiles(track, group) {
  for (const [key, value] of group) {
    const tile = document.createElement('div');
    tile.className = 'tile';
    const title = document.createElement('h3');
    title.textContent = key;
    tile.appendChild(title);
    tile.appendChild(renderTileContent(value));
    track.appendChild(tile);
  }
}

function renderInsightsTiles(container, data) {
  container.innerHTML = '';
  const makeRow = (name, group) => {
    const label = createLabelCell(name);
    const track = createTrackCell();
    addTiles(track, group);
    container.appendChild(label);
    container.appendChild(track);
  };

  // Dashboard row
  if (data.dashboard && typeof data.dashboard === 'object') {
    const entries = Object.entries(data.dashboard).filter(([k]) => k !== 'generated_at');
    makeRow('Dashboard', entries);
  }

  // subsystem_health row (array -> tiles per item)
  if (Array.isArray(data.subsystem_health)) {
    const group = data.subsystem_health.map((item, idx) => [`item_${idx + 1}`, item]);
    makeRow('subsystem_health', group);
  }

  // refactor_alerts row (array -> tiles per alert)
  if (Array.isArray(data.refactor_alerts)) {
    const group = data.refactor_alerts.map((item, idx) => [`alert_${idx + 1}`, item]);
    makeRow('refactor_alerts', group);
  }

  // metrics row (object -> tiles per key)
  if (data.metrics && typeof data.metrics === 'object') {
    makeRow('metrics', Object.entries(data.metrics));
  }

  // predictions row (object with arrays)
  if (data.predictions && typeof data.predictions === 'object') {
    const group = [];
    for (const [k, v] of Object.entries(data.predictions)) {
      group.push([k, v]);
    }
    makeRow('predictions', group);
  }

  // risk_areas row
  if (Array.isArray(data.risk_areas)) {
    const group = data.risk_areas.map((item, idx) => [`risk_${idx + 1}`, item]);
    makeRow('risk_areas', group);
  }
}

document.getElementById('analyze-form').addEventListener('submit', async (e) => {
  e.preventDefault();
  const input = document.getElementById('repo-url').value.trim();
  const parsed = toOwnerRepo(input);
  const status = document.getElementById('status');

  // Reset viewers
  document.getElementById('insights-viewer').innerHTML = '';
  document.getElementById('insights-raw').textContent = '';
  document.getElementById('suggestions-viewer').innerHTML = '';
  document.getElementById('suggestions-raw').textContent = '';

  if (!parsed) {
    status.textContent = 'Please enter a valid GitHub repo URL: https://github.com/owner/repo.git';
    return;
  }

  status.textContent = 'Dispatching workflow...';

  try {
    await dispatch(input);
    status.textContent = 'Workflow queued. Waiting for results...';
    const data = await pollInsights(parsed.owner, parsed.repo);
    const suggestions = await pollSuggestions(parsed.owner, parsed.repo);
    status.textContent = 'Analysis complete.';

    setupPanel({
      viewerId: 'insights-viewer',
      rawId: 'insights-raw',
      toggleId: 'insights-toggle',
      copyId: 'insights-copy',
      data,
      render: renderInsightsTiles,
    });

    setupPanel({
      viewerId: 'suggestions-viewer',
      rawId: 'suggestions-raw',
      toggleId: 'suggestions-toggle',
      copyId: 'suggestions-copy',
      data: suggestions,
      render: renderJsonViewer,
    });
  } catch (err) {
    status.textContent = `Error: ${err.message}`;
  }
});