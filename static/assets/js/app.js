/* Frontend logic for Resume Screening Tool with progress & dedupe */
const jdText = document.getElementById('jdText');
const jdFile = document.getElementById('jdFile');
const resumeFiles = document.getElementById('resumeFiles');
const fileList = document.getElementById('fileList');
const dropZone = document.getElementById('dropZone');

const mustHaves = document.getElementById('mustHaves');
const niceToHaves = document.getElementById('niceToHaves');

const wSim = document.getElementById('wSim');
const wCov = document.getElementById('wCov');
const wBon = document.getElementById('wBon');

const btnScore = document.getElementById('btnScore');
const btnExport = document.getElementById('btnExport');

const resultsArea = document.getElementById('resultsArea');
const resultsTable = document.getElementById('resultsTable');
const emptyState = document.getElementById('emptyState');

const detailModal = document.getElementById('detailModal');
const modalTitle = document.getElementById('modalTitle');
const modalBody = document.getElementById('modalBody');

const progressWrap = document.getElementById('progressWrap');
const progressBar = document.getElementById('progressBar');
const progressLabel = document.getElementById('progressLabel');
const progressPct = document.getElementById('progressPct');

let lastResults = null;

function human(n) { return Number(n).toFixed(1); }
function badge(text, variant='slate') {
  const colors = {
    green: 'bg-green-100 text-green-800',
    red: 'bg-rose-100 text-rose-800',
    amber: 'bg-amber-100 text-amber-800',
    slate: 'bg-slate-100 text-slate-800'
  };
  return `<span class="inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium ${colors[variant]||colors.slate}">${text}</span>`;
}

// ---- Drag & drop ----
dropZone.addEventListener('dragover', (e) => { e.preventDefault(); dropZone.classList.add('ring-2','ring-indigo-500'); });
dropZone.addEventListener('dragleave', () => { dropZone.classList.remove('ring-2','ring-indigo-500'); });
dropZone.addEventListener('drop', (e) => { e.preventDefault(); dropZone.classList.remove('ring-2','ring-indigo-500'); addResumeFiles(Array.from(e.dataTransfer.files||[])); });
resumeFiles.addEventListener('change', (e) => { addResumeFiles(Array.from(e.target.files||[])); });

function addResumeFiles(files) {
  // merge and de-duplicate by (name,size,lastModified)
  const current = Array.from(resumeFiles.files || []);
  const merged = [...current, ...files];
  const seen = new Set();
  const dt = new DataTransfer();
  for (const f of merged) {
    const key = `${f.name}::${f.size}::${f.lastModified}`;
    if (seen.has(key)) continue;
    seen.add(key);
    dt.items.add(f);
  }
  resumeFiles.files = dt.files;
  renderFileList();
}
function renderFileList() {
  const names = Array.from(resumeFiles.files || []).map(f => `• ${f.name}`);
  fileList.textContent = names.join('\n');
}

// ---- Modal ----
detailModal.addEventListener('click', (e) => { if (e.target.hasAttribute('data-close')) closeModal(); });
function openModal() { detailModal.classList.remove('hidden'); detailModal.classList.add('flex'); }
function closeModal() { detailModal.classList.add('hidden'); detailModal.classList.remove('flex'); }

// ---- Export CSV ----
btnExport.addEventListener('click', () => {
  if (!lastResults) return;
  const rows = [['Rank','Candidate','Total','Similarity','Coverage','MissingMustHaves','MatchedKeywords']];
  lastResults.results.forEach((r, idx) => {
    rows.push([
      String(idx+1),
      r.candidate_name,
      String(r.total),
      String(r.sim),
      String(r.coverage),
      (r.missing_must_haves||[]).join('|'),
      (r.matched_keywords||[]).join('|')
    ]);
  });
  const csv = rows.map(row => row.map(v => `"${String(v).replace(/"/g,'""')}"`).join(',')).join('\n');
  const blob = new Blob([csv], {type: 'text/csv;charset=utf-8;'});
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url; a.download = 'resume_screening_results.csv';
  document.body.appendChild(a); a.click(); document.body.removeChild(a);
  URL.revokeObjectURL(url);
});

// ---- Progress helpers ----
function setProgress(pct, label) {
  progressWrap.classList.remove('hidden');
  progressBar.style.width = `${Math.max(0, Math.min(100, pct))}%`;
  progressPct.textContent = `${Math.round(Math.max(0, Math.min(100, pct)))}%`;
  if (label) progressLabel.textContent = label;
}
function hideProgress() {
  progressBar.style.width = '0%';
  progressPct.textContent = '0%';
  progressLabel.textContent = '';
  progressWrap.classList.add('hidden');
}
function loading(isLoading) {
  btnScore.disabled = isLoading;
  btnScore.textContent = isLoading ? 'Scoring…' : 'Run Screening';
  const disabled = !lastResults || !lastResults.results || !lastResults.results.length;
  btnExport.disabled = disabled;
  btnExport.classList.toggle('cursor-not-allowed', disabled);
  btnExport.classList.toggle('bg-slate-200', disabled);
  btnExport.classList.toggle('bg-emerald-600', !disabled);
  btnExport.classList.toggle('text-white', !disabled);
}

// ---- Incremental scoring (real % progress) ----
btnScore.addEventListener('click', async () => {
  const files = Array.from(resumeFiles.files || []);
  if (!files.length) { alert('Please add at least one resume file.'); return; }

  const initFd = new FormData();
  const text = jdText.value.trim();
  if (text) initFd.append('jd_text', text);
  if (jdFile.files[0]) initFd.append('jd_file', jdFile.files[0]);
  if (mustHaves.value.trim()) initFd.append('must_haves', mustHaves.value.trim());
  if (niceToHaves.value.trim()) initFd.append('nice_to_haves', niceToHaves.value.trim());
  initFd.append('weight_sim', wSim.value || '0.60');
  initFd.append('weight_cov', wCov.value || '0.30');
  initFd.append('weight_bonus', wBon.value || '0.10');

  loading(true);
  setProgress(5, 'Preparing job…');

  try {
    // Step 1: init job (get token)
    let res = await fetch('/api/score_init', { method: 'POST', body: initFd });
    // fallback to old endpoint if not available
    if (res.status === 404) return await legacyBulkScore(files, initFd);

    const initData = await res.json();
    if (!res.ok) throw new Error(initData?.detail || 'Init failed');
    const token = initData.job_token;
    const total = files.length;
    const out = [];

    // Step 2: score each resume (N requests => real progress)
    for (let i = 0; i < total; i++) {
      const f = files[i];
      setProgress(5 + (95*(i/total)), `Scoring ${f.name} (${i+1}/${total})`);
      const fd = new FormData();
      fd.append('job_token', token);
      fd.append('resume', f);
      const r = await fetch('/api/score_one', { method: 'POST', body: fd });
      const jr = await r.json();
      if (!r.ok) throw new Error(jr?.detail || 'Score failed');
      out.push(jr.result);
      setProgress(5 + (95*((i+1)/total)), `Scoring ${f.name} (${i+1}/${total})`);
    }

    // Sort, show results
    out.sort((a,b) => b.total - a.total);
    lastResults = {
      keywords_used: initData.keywords_used,
      must_haves: initData.must_haves,
      nice_to_haves: initData.nice_to_haves,
      results: out
    };
    renderResults(lastResults);
  } catch (err) {
    console.error(err);
    alert(err.message || 'Error during scoring.');
  } finally {
    loading(false);
    setTimeout(hideProgress, 600); // small delay so the user sees 100%
  }
});

// Fallback: original bulk endpoint (no %; still dedup on backend)
async function legacyBulkScore(files, initFd) {
  setProgress(30, 'Scoring (bulk endpoint)…');
  // Rebuild payload expected by /api/score
  const fd = new FormData();
  for (const [k,v] of initFd.entries()) {
    if (k === 'jd_file' || k === 'jd_text' || k.startsWith('weight_') || k.endsWith('_haves')) fd.append(k, v);
  }
  files.forEach(f => fd.append('resumes', f));
  const res = await fetch('/api/score', { method: 'POST', body: fd });
  const data = await res.json();
  if (!res.ok || data.error) throw new Error(data.error || 'Scoring failed');
  lastResults = data;
  renderResults(data);
}

function renderResults(data) {
  resultsTable.innerHTML = '';
  const rows = [];
  (data.results||[]).forEach((r, idx) => {
    const mh = (r.missing_must_haves||[]).length ? badge(`${r.missing_must_haves.length} missing`, 'red') : badge('All met', 'green');
    const scoreBadge = r.total >= 80 ? badge('Strong fit','green') : (r.total >= 60 ? badge('Possible','amber') : badge('Low fit','slate'));
    const tr = document.createElement('tr');
    tr.className = idx % 2 ? 'bg-white' : 'bg-slate-50/40';
    tr.innerHTML = `
      <td class="py-3 px-4">${idx+1}</td>
      <td class="py-3 px-4">
        <div class="font-medium">${r.candidate_name}</div>
        <div class="text-xs text-slate-500">${r.filename}</div>
      </td>
      <td class="py-3 px-4">
        <div class="font-semibold">${human(r.total)}</div>
        <div class="mt-1">${scoreBadge}</div>
      </td>
      <td class="py-3 px-4">${human(r.sim)}</td>
      <td class="py-3 px-4">${human(r.coverage)}</td>
      <td class="py-3 px-4">${mh}</td>
      <td class="py-3 px-4 text-right">
        <button class="px-3 py-1.5 rounded-lg bg-slate-900 text-white text-xs viewBtn">View</button>
      </td>
    `;
    tr.querySelector('.viewBtn').addEventListener('click', () => openDetails(r, idx+1, data));
    rows.push(tr);
  });
  rows.forEach(tr => resultsTable.appendChild(tr));
  const has = rows.length > 0;
  resultsArea.classList.toggle('hidden', !has);
  emptyState.classList.toggle('hidden', has);
  lastResults = data;
  loading(false);
}

function openDetails(r, rank, data) {
  modalTitle.textContent = `#${rank} — ${r.candidate_name}`;
  const mk = (r.matched_keywords || []).slice(0,20).map(k => badge(k, 'slate')).join(' ');
  const mhMiss = (r.missing_must_haves || []).map(k => badge(k, 'red')).join(' ') || badge('None', 'green');
  const nh = (r.matched_nice_to_haves || []).map(k => badge(k, 'green')).join(' ') || '—';
  const snips = (r.snippets || []).map(s => `<li class="leading-relaxed">"${escapeHtml(s)}"</li>`).join('') || '<li class="text-slate-500">No snippets available</li>';
  modalBody.innerHTML = `
    <div class="grid md:grid-cols-3 gap-4">
      <div class="p-3 rounded-xl bg-slate-50 border border-slate-200"><div class="text-xs uppercase text-slate-500">Total Score</div><div class="text-2xl font-bold">${human(r.total)}</div></div>
      <div class="p-3 rounded-xl bg-slate-50 border border-slate-200"><div class="text-xs uppercase text-slate-500">Similarity</div><div class="text-2xl font-bold">${human(r.sim)}</div></div>
      <div class="p-3 rounded-xl bg-slate-50 border border-slate-200"><div class="text-xs uppercase text-slate-500">Coverage</div><div class="text-2xl font-bold">${human(r.coverage)}</div></div>
    </div>
    <div class="grid md:grid-cols-2 gap-4">
      <div class="p-4 rounded-xl bg-white border border-slate-200"><div class="text-sm font-semibold mb-2">Matched Keywords</div><div class="space-x-1 space-y-1">${mk}</div></div>
      <div class="p-4 rounded-xl bg-white border border-slate-200"><div class="text-sm font-semibold mb-2">Missing Must-haves</div><div class="space-x-1 space-y-1">${mhMiss}</div></div>
    </div>
    <div class="p-4 rounded-xl bg-white border border-slate-200"><div class="text-sm font-semibold mb-2">Matched Nice-to-haves</div><div class="space-x-1 space-y-1">${nh}</div></div>
    <div class="p-4 rounded-xl bg-white border border-slate-200"><div class="text-sm font-semibold mb-2">Resume Snippets</div><ul class="list-disc pl-5">${snips}</ul></div>
  `;
  openModal();
}

function escapeHtml(s) {
  return String(s).replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;').replaceAll('"','&quot;').replaceAll("'",'&#039;');
}
