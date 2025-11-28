// Configure these values to match your deployment
const VERCEL_ENDPOINT = 'https://echelons-vercel-api.vercel.app/api/dispatch';
const PAGES_BASE = window.location.origin + window.location.pathname.replace(/\/index\.html$/, '').replace(/\/$/, '');
let currentSearchTerm = '';

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

function setupPanel({ viewerId, rawId, toggleId, copyId, data, render, searchId, onSearch }) {
  const viewer = document.getElementById(viewerId);
  const raw = document.getElementById(rawId);
  const toggle = document.getElementById(toggleId);
  const copy = document.getElementById(copyId);
  const search = searchId ? document.getElementById(searchId) : null;

  render(viewer, data);
  raw.textContent = JSON.stringify(data, null, 2);

  toggle.addEventListener('click', () => {
    const pressed = toggle.getAttribute('aria-pressed') === 'true';
    toggle.setAttribute('aria-pressed', String(!pressed));
    const showRaw = !pressed;
    if (showRaw && (!raw.textContent || raw.textContent.trim() === '')) {
      try { raw.textContent = JSON.stringify(data, null, 2); }
      catch { raw.textContent = 'Failed to render JSON'; }
    }
    raw.hidden = !showRaw;
    viewer.hidden = showRaw;
  });
  copy.addEventListener('click', () => navigator.clipboard.writeText(raw.textContent));

  if (search && typeof onSearch === 'function') {
    search.addEventListener('input', () => {
      const q = search.value.trim();
      currentSearchTerm = q;
      onSearch(viewer, q);
    });
  }
}

// --- Insights Tiles Renderer ---
function renderTileContent(value) {
  const box = document.createElement('div');
  if (Array.isArray(value)) {
    const badge = document.createElement('div');
    badge.className = 'badge';
    badge.textContent = `${value.length} items`;
    box.appendChild(badge);
    const fragment = document.createDocumentFragment();
    value.slice(0, 10).forEach((item) => {
      const block = document.createElement('div');
      block.className = 'kv-block';
      if (item && typeof item === 'object') {
        for (const [k, v] of Object.entries(item)) {
          const line = document.createElement('div');
          line.className = 'kv-line';
          const key = document.createElement('span');
          key.className = 'key';
          key.textContent = `${k}: `;
          const val = document.createElement('span');
          val.className = 'val';
          val.textContent = typeof v === 'object' ? JSON.stringify(v) : String(v);
          line.appendChild(key);
          line.appendChild(val);
          block.appendChild(line);
        }
      } else {
        const line = document.createElement('div');
        line.className = 'kv-line';
        line.textContent = String(item);
        block.appendChild(line);
      }
      fragment.appendChild(block);
    });
    box.appendChild(fragment);
  } else if (value && typeof value === 'object') {
    for (const [k, v] of Object.entries(value)) {
      const line = document.createElement('div');
      line.className = 'kv-line';
      const key = document.createElement('span');
      key.className = 'key';
      key.textContent = `${k}: `;
      const val = document.createElement('span');
      val.className = 'val';
      val.textContent = typeof v === 'object' ? JSON.stringify(v) : String(v);
      line.appendChild(key);
      line.appendChild(val);
      box.appendChild(line);
    }
  } else {
    const p = document.createElement('div');
    p.className = 'kv-line';
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
  vt.textContent = splitNameForVertical(name);
  tile.appendChild(vt);
  cell.appendChild(tile);
  return cell;
}

function createTrackCell() {
  const cell = document.createElement('div');
  cell.className = 'row-track';
  let hoverTimer;
  cell.addEventListener('mouseenter', () => {
    if (hoverTimer) { clearTimeout(hoverTimer); hoverTimer = null; }
    cell.classList.add('track-hover');
  });
  cell.addEventListener('mouseleave', () => {
    if (hoverTimer) { clearTimeout(hoverTimer); }
    hoverTimer = setTimeout(() => { cell.classList.remove('track-hover'); }, 2000);
  });
  return cell;
}

function addTiles(track, group) {
  for (const [key, value] of group) {
    const tile = document.createElement('div');
    tile.className = 'tile';
    tile.tabIndex = 0;
    const title = document.createElement('h3');
    title.textContent = key;
    tile.appendChild(title);
    tile.appendChild(renderTileContent(value));
    track.appendChild(tile);

    // Attach value for modal and search
    tile.__value = value;
    tile.__title = key;

    tile.addEventListener('click', () => openTileModal(key, value));
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
  if (Array.isArray(data.subsystem_health) && data.subsystem_health.length > 0) {
    const group = data.subsystem_health.map((item, idx) => [`item_${idx + 1}`, item]);
    makeRow('subsystem_health', group);
  }

  // refactor_alerts row (array -> tiles per alert)
  if (Array.isArray(data.refactor_alerts) && data.refactor_alerts.length > 0) {
    const group = data.refactor_alerts.map((item, idx) => [`alert_${idx + 1}`, item]);
    makeRow('refactor_alerts', group);
  }

  // metrics row (object -> tiles per key)
  if (data.metrics && typeof data.metrics === 'object') {
    const entries = Object.entries(data.metrics);
    if (entries.length > 0) makeRow('metrics', entries);
  }

  // predictions row (object with arrays)
  if (data.predictions && typeof data.predictions === 'object') {
    const group = [];
    for (const [k, v] of Object.entries(data.predictions)) {
      if (Array.isArray(v) && v.length === 0) continue;
      if (v && typeof v === 'object' && !Array.isArray(v) && Object.keys(v).length === 0) continue;
      group.push([k, v]);
    }
    if (group.length > 0) makeRow('predictions', group);
  }

  // risk_areas row
  if (Array.isArray(data.risk_areas) && data.risk_areas.length > 0) {
    const group = data.risk_areas.map((item, idx) => [`risk_${idx + 1}`, item]);
    makeRow('risk_areas', group);
  }
}

// --- Modal behavior ---
function openTileModal(title, value) {
  const overlay = document.createElement('div');
  overlay.className = 'modal';
  overlay.setAttribute('role', 'dialog');
  overlay.setAttribute('aria-modal', 'true');

  const modal = document.createElement('div');
  modal.className = 'modal-content';

  const header = document.createElement('div');
  header.className = 'modal-header';
  const h = document.createElement('h3');
  h.className = 'modal-title';
  h.textContent = title;
  const close = document.createElement('button');
  close.className = 'modal-close';
  close.textContent = 'Close';

  const body = document.createElement('div');
  body.className = 'modal-body';
  const rendered = renderFullTextContent(value);
  body.appendChild(rendered);
  if (currentSearchTerm && currentSearchTerm.trim()) {
    applyTextHighlights(rendered, currentSearchTerm);
  }

  const copyAll = document.createElement('button');
  copyAll.className = 'modal-copy';
  copyAll.textContent = 'Copy All';
  copyAll.addEventListener('click', () => {
    const lines = [];
    body.querySelectorAll('.kv-line').forEach(line => {
      const keyEl = line.querySelector('.key');
      const valEl = line.querySelector('.val');
      if (keyEl && valEl) {
        lines.push(`${keyEl.textContent}${valEl.textContent}`);
      } else if (valEl) {
        lines.push(valEl.textContent);
      } else {
        const txt = line.textContent.replace(/^\s*Copy\s*/,'').trim();
        if (txt) lines.push(txt);
      }
    });
    const payload = `${title}\n\n${lines.join('\n')}`;
    navigator.clipboard.writeText(payload);
  });

  const actions = document.createElement('div');
  actions.className = 'modal-actions';
  actions.appendChild(copyAll);
  actions.appendChild(close);
  header.appendChild(h);
  header.appendChild(actions);
  modal.appendChild(header);
  modal.appendChild(body);
  overlay.appendChild(modal);
  document.body.appendChild(overlay);

  const dispose = () => {
    document.body.removeChild(overlay);
  };
  close.addEventListener('click', dispose);
  overlay.addEventListener('click', (e) => { if (e.target === overlay) dispose(); });
  document.addEventListener('keydown', function escHandler(e) {
    if (e.key === 'Escape') {
      document.removeEventListener('keydown', escHandler);
      dispose();
    }
  });
  close.focus();
}

// --- Search behavior ---
function clearHighlights(container) {
  container.querySelectorAll('.tile.highlight').forEach(el => el.classList.remove('highlight'));
  container.querySelectorAll('.tile.match').forEach(el => el.classList.remove('match'));
  container.querySelectorAll('.bar.match').forEach(el => el.classList.remove('match'));
  container.querySelectorAll('.match-text').forEach(span => {
    const parent = span.parentNode;
    if (!parent) return;
    span.replaceWith(document.createTextNode(span.textContent));
    parent.normalize();
  });
}

function searchInsightsTiles(container, query) {
  clearHighlights(container);
  if (!query) return;
  const tiles = Array.from(container.querySelectorAll('.tile'));
  const lower = query.toLowerCase();
  tiles.forEach(t => {
    const text = `${t.__title || ''} ${t.textContent || ''}`.toLowerCase();
    if (text.includes(lower)) {
      t.classList.add('match');
      applyTextHighlights(t, query);
    }
  });
}

function applyTextHighlights(root, term) {
  if (!term) return;
  const lc = term.toLowerCase();
  const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, {
    acceptNode(node) {
      const val = (node.nodeValue || '').trim();
      if (!val) return NodeFilter.FILTER_SKIP;
      return val.toLowerCase().includes(lc) ? NodeFilter.FILTER_ACCEPT : NodeFilter.FILTER_SKIP;
    }
  });
  const nodes = [];
  while (walker.nextNode()) nodes.push(walker.currentNode);
  nodes.forEach(node => {
    const text = node.nodeValue;
    const idx = text.toLowerCase().indexOf(lc);
    if (idx === -1) return;
    const before = text.slice(0, idx);
    const match = text.slice(idx, idx + term.length);
    const after = text.slice(idx + term.length);
    const frag = document.createDocumentFragment();
    if (before) frag.appendChild(document.createTextNode(before));
    const mark = document.createElement('span');
    mark.className = 'match-text';
    mark.textContent = match;
    frag.appendChild(mark);
    if (after) frag.appendChild(document.createTextNode(after));
    node.parentNode.replaceChild(frag, node);
  });
}

// --- Full text renderer for modal ---
function renderFullTextContent(value) {
  const box = document.createElement('div');
  const appendKV = (parent, k, v) => {
    const line = document.createElement('div');
    line.className = 'kv-line seg-line';
    const copyBtn = document.createElement('button');
    copyBtn.className = 'copy-segment';
    copyBtn.textContent = 'Copy';
    copyBtn.addEventListener('click', (e) => {
      e.stopPropagation();
      navigator.clipboard.writeText(`${k}: ${typeof v === 'object' ? JSON.stringify(v) : String(v)}`);
    });
    const key = document.createElement('span');
    key.className = 'key';
    key.textContent = `${k}: `;
    const val = document.createElement('span');
    val.className = 'val';
    val.textContent = typeof v === 'object' ? JSON.stringify(v) : String(v);
    line.appendChild(copyBtn);
    line.appendChild(key);
    line.appendChild(val);
    parent.appendChild(line);
  };

  if (Array.isArray(value)) {
    value.forEach((item, idx) => {
      const block = document.createElement('div');
      block.className = 'kv-block';
      if (item && typeof item === 'object') {
        for (const [k, v] of Object.entries(item)) appendKV(block, k, v);
      } else {
        appendKV(block, `item_${idx + 1}`, item);
      }
      box.appendChild(block);
    });
  } else if (value && typeof value === 'object') {
    for (const [k, v] of Object.entries(value)) appendKV(box, k, v);
  } else {
    appendKV(box, 'value', value);
  }
  return box;
}

// --- Suggestions bars renderer ---
function renderSuggestionsBars(container, data) {
  container.innerHTML = '';
  const renderBar = (scope, items) => {
    const bar = document.createElement('div');
    bar.className = 'bar';
    bar.tabIndex = 0;
    bar.__title = scope;
    const title = document.createElement('h3');
    title.className = 'bar-title';
    title.textContent = scope;
    const count = document.createElement('span');
    count.className = 'bar-count';
    count.textContent = `${items.length} items`;
    bar.appendChild(title);
    bar.appendChild(count);
    bar.addEventListener('click', () => openSuggestionsModal(scope, items));
    container.appendChild(bar);
  };

  const renderValueBar = (scope, value) => {
    const bar = document.createElement('div');
    bar.className = 'bar';
    const title = document.createElement('h3');
    title.className = 'bar-title';
    title.textContent = scope;
    const val = document.createElement('span');
    val.className = 'bar-count';
    val.textContent = typeof value === 'object' ? JSON.stringify(value) : String(value);
    bar.appendChild(title);
    bar.appendChild(val);
    bar.addEventListener('click', () => openSuggestionsValueModal(scope, value));
    container.appendChild(bar);
  };

  // Shape 1: { suggestions: [...] }
  if (Array.isArray(data?.suggestions)) {
    const groups = new Map();
    data.suggestions.forEach(s => {
      const scope = s.scope || s.metadata?.scope || 'unknown_scope';
      if (!groups.has(scope)) groups.set(scope, []);
      groups.get(scope).push(s);
    });
    for (const [scope, items] of groups.entries()) { if (items.length > 0) renderBar(scope, items); }
    return;
  }

  // Shape 2: array of suggestions
  if (Array.isArray(data)) {
    const groups = new Map();
    data.forEach(s => {
      const scope = s.scope || s.metadata?.scope || 'unknown_scope';
      if (!groups.has(scope)) groups.set(scope, []);
      groups.get(scope).push(s);
    });
    for (const [scope, items] of groups.entries()) { if (items.length > 0) renderBar(scope, items); }
    return;
  }

  // Shape 3: object of arrays keyed by scope (e.g., project_scope, subsystem_scope, file_scope)
  if (data && typeof data === 'object') {
    const entries = Object.entries(data);
    if (entries.length === 0) return;
    entries.forEach(([scope, items]) => {
      if (Array.isArray(items)) {
        if (items.length > 0) renderBar(scope, items);
        return;
      }
      if (items && typeof items === 'object' && Array.isArray(items.suggestions)) {
        if (items.suggestions.length > 0) renderBar(scope, items.suggestions);
        return;
      }
      // Primitive or object value: render as value bar
      if (items !== undefined && items !== null && !(Array.isArray(items) && items.length === 0)) {
        renderValueBar(scope, items);
      }
    });
    return;
  }
}

// --- Suggestions search/highlight ---
function searchSuggestionsBars(container, query) {
  if (!query) { clearHighlights(container); return; }
  clearHighlights(container);
  const bars = Array.from(container.querySelectorAll('.bar'));
  const lower = query.toLowerCase();
  const matches = bars.filter(b => {
    const text = `${b.__title || ''} ${b.textContent || ''}`.toLowerCase();
    return text.includes(lower);
  });
  if (matches.length === 0) return;

  const containerTop = container.scrollTop;
  const scored = matches.map(b => {
    const titleMatch = (b.__title || '').toLowerCase().includes(lower) ? 1 : 0;
    const rowTop = b.offsetTop || 0;
    const verticalDistance = Math.abs(rowTop - containerTop);
    return { bar: b, score: titleMatch, dist: verticalDistance };
  });
  scored.sort((a, b) => {
    if (b.score !== a.score) return b.score - a.score;
    return a.dist - b.dist;
  });
  const match = scored[0].bar;

  container.scrollTo({ top: match.offsetTop, behavior: 'smooth' });
  match.classList.add('highlight');
  applyTextHighlights(match, query);
  match.focus();
}

function openSuggestionsModal(scope, items) {
  const overlay = document.createElement('div');
  overlay.className = 'modal';
  overlay.setAttribute('role', 'dialog');
  overlay.setAttribute('aria-modal', 'true');

  const modal = document.createElement('div');
  modal.className = 'modal-content';

  const header = document.createElement('div');
  header.className = 'modal-header';
  const h = document.createElement('h3');
  h.className = 'modal-title';
  h.textContent = scope;
  const close = document.createElement('button');
  close.className = 'modal-close';
  close.textContent = 'Close';

  const body = document.createElement('div');
  body.className = 'modal-body';
  items.forEach((s, idx) => {
    const block = document.createElement('div');
    block.className = 'kv-block';
    const print = (k, v) => {
      if (v === undefined || v === null) return;
      const line = document.createElement('div');
      line.className = 'kv-line seg-line';
      const copyBtn = document.createElement('button');
      copyBtn.className = 'copy-segment';
      copyBtn.textContent = 'Copy';
      copyBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        navigator.clipboard.writeText(`${k}: ${typeof v === 'object' ? JSON.stringify(v) : String(v)}`);
      });
      const key = document.createElement('span');
      key.className = 'key';
      key.textContent = `${k}: `;
      const val = document.createElement('span');
      val.className = 'val';
      val.textContent = typeof v === 'object' ? JSON.stringify(v) : String(v);
      line.appendChild(copyBtn);
      line.appendChild(key);
      line.appendChild(val);
      block.appendChild(line);
    };
    print('id', s.id);
    print('title', s.title);
    print('priority', s.priority);
    print('confidence_score', s.confidence_score);
    print('estimated_roi', s.estimated_roi);
    print('estimated_effort', s.estimated_effort);
    if (s.categories) print('categories', s.categories);
    if (s.locations) print('locations', s.locations);
    if (s.actions) print('actions', s.actions);
    if (s.rationale) print('rationale', s.rationale);
    if (s.horizon) print('horizon', s.horizon);
    if (s.metadata) print('metadata', s.metadata);
    body.appendChild(block);
  });

  const copyAll = document.createElement('button');
  copyAll.className = 'modal-copy';
  copyAll.textContent = 'Copy All';
  copyAll.addEventListener('click', () => {
    const lines = [];
    body.querySelectorAll('.kv-line').forEach(line => {
      const keyEl = line.querySelector('.key');
      const valEl = line.querySelector('.val');
      if (keyEl && valEl) {
        lines.push(`${keyEl.textContent}${valEl.textContent}`);
      } else if (valEl) {
        lines.push(valEl.textContent);
      } else {
        const txt = line.textContent.replace(/^\s*Copy\s*/,'').trim();
        if (txt) lines.push(txt);
      }
    });
    const payload = `${scope}\n\n${lines.join('\n')}`;
    navigator.clipboard.writeText(payload);
  });

  const actions = document.createElement('div');
  actions.className = 'modal-actions';
  actions.appendChild(copyAll);
  actions.appendChild(close);
  header.appendChild(h);
  header.appendChild(actions);
  modal.appendChild(header);
  modal.appendChild(body);
  overlay.appendChild(modal);
  if (currentSearchTerm && currentSearchTerm.trim()) {
    applyTextHighlights(body, currentSearchTerm);
  }
  document.body.appendChild(overlay);

  const dispose = () => { document.body.removeChild(overlay); };
  close.addEventListener('click', dispose);
  overlay.addEventListener('click', (e) => { if (e.target === overlay) dispose(); });
  document.addEventListener('keydown', function escHandler(e) {
    if (e.key === 'Escape') { document.removeEventListener('keydown', escHandler); dispose(); }
  });
  close.focus();
}

function openSuggestionsValueModal(scope, value) {
  const overlay = document.createElement('div');
  overlay.className = 'modal';
  overlay.setAttribute('role', 'dialog');
  overlay.setAttribute('aria-modal', 'true');

  const modal = document.createElement('div');
  modal.className = 'modal-content';

  const header = document.createElement('div');
  header.className = 'modal-header';
  const h = document.createElement('h3');
  h.className = 'modal-title';
  h.textContent = scope;
  const close = document.createElement('button');
  close.className = 'modal-close';
  close.textContent = 'Close';

  const body = document.createElement('div');
  body.className = 'modal-body';
  const line = document.createElement('div');
  line.className = 'kv-line seg-line';
  const copyBtn = document.createElement('button');
  copyBtn.className = 'copy-segment';
  copyBtn.textContent = 'Copy';
  copyBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    navigator.clipboard.writeText(typeof value === 'object' ? JSON.stringify(value) : String(value));
  });
  const val = document.createElement('span');
  val.className = 'val';
  val.textContent = typeof value === 'object' ? JSON.stringify(value) : String(value);
  line.appendChild(copyBtn);
  line.appendChild(val);
  body.appendChild(line);

  const copyAll = document.createElement('button');
  copyAll.className = 'modal-copy';
  copyAll.textContent = 'Copy All';
  copyAll.addEventListener('click', () => {
    const valEl = body.querySelector('.kv-line .val');
    const text = valEl ? valEl.textContent : (typeof value === 'object' ? JSON.stringify(value) : String(value));
    const payload = `${scope}\n\n${text}`;
    navigator.clipboard.writeText(payload);
  });

  const actions = document.createElement('div');
  actions.className = 'modal-actions';
  actions.appendChild(copyAll);
  actions.appendChild(close);
  header.appendChild(h);
  header.appendChild(actions);
  modal.appendChild(header);
  modal.appendChild(body);
  overlay.appendChild(modal);
  if (currentSearchTerm && currentSearchTerm.trim()) {
    applyTextHighlights(body, currentSearchTerm);
  }
  document.body.appendChild(overlay);

  const dispose = () => { document.body.removeChild(overlay); };
  close.addEventListener('click', dispose);
  overlay.addEventListener('click', (e) => { if (e.target === overlay) dispose(); });
  document.addEventListener('keydown', function escHandler(e) { if (e.key === 'Escape') { document.removeEventListener('keydown', escHandler); dispose(); } });
  close.focus();
}
function splitNameForVertical(name) {
  if (!name) return '';
  if (name.includes('_')) {
    const parts = name.split('_');
    if (parts.length >= 2) return parts.slice(0, 2).join('\n');
  }
  if (name.includes(' ')) {
    const parts = name.split(' ');
    if (parts.length >= 2) return parts.slice(0, 2).join('\n');
  }
  if (name.length > 10) {
    const mid = Math.floor(name.length / 2);
    return `${name.slice(0, mid)}\n${name.slice(mid)}`;
  }
  return name;
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
      searchId: 'insights-search',
      onSearch: searchInsightsTiles,
    });

    setupPanel({
      viewerId: 'suggestions-viewer',
      rawId: 'suggestions-raw',
      toggleId: 'suggestions-toggle',
      copyId: 'suggestions-copy',
      data: suggestions,
      render: renderSuggestionsBars,
      searchId: 'suggestions-search',
      onSearch: searchSuggestionsBars,
    });
  } catch (err) {
    status.textContent = `Error: ${err.message}`;
  }
});