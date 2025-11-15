// Configure these values to match your deployment
const VERCEL_ENDPOINT = 'https://YOUR-VERCEL-PROJECT.vercel.app/api/dispatch';
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

document.getElementById('analyze-form').addEventListener('submit', async (e) => {
  e.preventDefault();
  const input = document.getElementById('repo-url').value.trim();
  const parsed = toOwnerRepo(input);
  const status = document.getElementById('status');
  const output = document.getElementById('output');
  output.textContent = '';

  if (!parsed) {
    status.textContent = 'Please enter a valid GitHub repo URL: https://github.com/owner/repo.git';
    return;
  }

  status.textContent = 'Dispatching workflow...';

  try {
    await dispatch(input);
    status.textContent = 'Workflow queued. Waiting for results...';
    const data = await pollInsights(parsed.owner, parsed.repo);
    status.textContent = 'Analysis complete.';
    output.textContent = JSON.stringify(data, null, 2);
  } catch (err) {
    status.textContent = `Error: ${err.message}`;
  }
});