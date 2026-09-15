'use strict';

// ==========================================
// STATE
// ==========================================
let currentState = {};
try {
  currentState = JSON.parse(document.getElementById('initial-state-data').textContent);
} catch (e) {
  console.error('Could not parse initial state:', e);
}

const DEFAULT_OLLAMA_MODEL = 'qwen3-coder:30b';
const FALLBACK_OLLAMA_MODELS = ['qwen3-coder:30b', 'qwen2.5-coder:14b', 'deepseek-r1:14b', 'qwen2.5-coder:7b', 'mistral-small:24b'];
const CLI_MODEL_OPTIONS = [
  { val: 'claude', text: 'Claude CLI (Default)' },
  { val: 'claude:claude-opus-5', text: 'Claude Opus 5 (CLI)' },
  { val: 'claude:claude-sonnet-5', text: 'Claude Sonnet 5 (CLI)' },
  { val: 'gemini:gemini-3.8-flash-high', text: 'Gemini 3.8 Flash High (agy)' },
  { val: 'gemini:gemini-3.8-flash-medium', text: 'Gemini 3.8 Flash Medium (agy)' },
  { val: 'gemini:gemini-3.1-pro-high', text: 'Gemini 3.1 Pro High (agy)' },
];
const MODEL_PICKERS = [
  { id: 'run-model', defaultVal: DEFAULT_OLLAMA_MODEL, includeNone: false },
  { id: 'review-model', defaultVal: '', includeNone: true },
  { id: 'escalate-model', defaultVal: '', includeNone: true },
  { id: 'draft-model', defaultVal: DEFAULT_OLLAMA_MODEL, includeNone: false },
];
const PERSISTED_CONTROLS = ['run-model', 'review-model', 'escalate-model', 'draft-model', 'run-backend', 'run-limit'];
const MAX_LOG_ROWS = 2000;

let branches = [];
let activeReviewBranch = null;
let todoDetails = null;       // last checklist snapshot from the server
let todoBase = null;          // {content, hash} the editors were loaded from / last saved as
let editorDirty = false;
let lastStage = null;
let elapsedTimer = null;

const $ = id => document.getElementById(id);

function isRunning() {
  return !!currentState.is_running;
}

function escapeHtml(text) {
  return String(text ?? '')
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;').replace(/'/g, '&#39;');
}

function debounce(fn, ms) {
  let t = null;
  return (...args) => { clearTimeout(t); t = setTimeout(() => fn(...args), ms); };
}

// Local preferences: never required for correctness, so every access is guarded.
const prefs = {
  get(key, fallback) {
    try { const v = localStorage.getItem('forge.studio.' + key); return v === null ? fallback : JSON.parse(v); }
    catch (e) { return fallback; }
  },
  set(key, value) {
    try { localStorage.setItem('forge.studio.' + key, JSON.stringify(value)); } catch (e) { /* ignore */ }
  },
};

// ==========================================
// API, TOASTS, MODALS
// ==========================================
class ApiError extends Error {
  constructor(message, status, payload) {
    super(message);
    this.status = status;
    this.payload = payload || {};
  }
}

async function api(path, { method = 'GET', body } = {}) {
  const opts = { method, headers: {} };
  if (body !== undefined) {
    opts.headers['Content-Type'] = 'application/json';
    opts.body = JSON.stringify(body);
  }
  let res;
  try {
    res = await fetch(path, opts);
  } catch (e) {
    throw new ApiError('Studio server unreachable - is `forge studio` still running?', 0);
  }
  let data = {};
  try { data = await res.json(); } catch (e) { /* non-JSON error page */ }
  if (!res.ok || data.ok === false) {
    throw new ApiError(data.error || data.message || `Request failed (HTTP ${res.status})`, res.status, data);
  }
  return data;
}

function showToast(message, { kind = 'info', details = '', timeout } = {}) {
  const stack = $('toast-stack');
  const toast = document.createElement('div');
  toast.className = 'toast toast-' + kind;
  toast.innerHTML = `
    <div class="toast-row">
      <span class="toast-text"></span>
      <button class="toast-close" aria-label="Dismiss">×</button>
    </div>`;
  toast.querySelector('.toast-text').textContent = message;
  if (details) {
    const pre = document.createElement('pre');
    pre.className = 'toast-details';
    pre.textContent = details;
    toast.appendChild(pre);
  }
  const remove = () => toast.remove();
  toast.querySelector('.toast-close').addEventListener('click', remove);
  stack.appendChild(toast);
  const ms = timeout ?? (kind === 'error' ? (details ? 0 : 9000) : 3500);
  if (ms > 0) setTimeout(remove, ms);
  while (stack.children.length > 5) stack.firstElementChild.remove();
}

function showError(err, prefix) {
  const msg = (prefix ? prefix + ': ' : '') + (err && err.message ? err.message : String(err));
  const details = err && err.payload && typeof err.payload.details === 'string' ? err.payload.details : '';
  showToast(msg, { kind: 'error', details });
}

let modalResolve = null;

function closeModal(result) {
  $('modal-backdrop').hidden = true;
  const resolve = modalResolve;
  modalResolve = null;
  if (resolve) resolve(result);
}

// actions: [{label, value, kind: 'red'|'white'|undefined}]; resolves with the chosen value (null on dismiss).
function openModal({ title, body, actions, focusSelector }) {
  if (modalResolve) closeModal(null);
  $('modal-title').textContent = title;
  const bodyEl = $('modal-body');
  bodyEl.innerHTML = '';
  if (typeof body === 'string') {
    const p = document.createElement('p');
    p.textContent = body;
    bodyEl.appendChild(p);
  } else if (body) {
    bodyEl.appendChild(body);
  }
  const actionsEl = $('modal-actions');
  actionsEl.innerHTML = '';
  actions.forEach(a => {
    const b = document.createElement('button');
    b.textContent = a.label;
    if (a.kind) b.className = 'btn-' + a.kind;
    b.addEventListener('click', () => closeModal(typeof a.value === 'function' ? a.value() : a.value));
    actionsEl.appendChild(b);
  });
  $('modal-backdrop').hidden = false;
  const focusEl = (focusSelector && bodyEl.querySelector(focusSelector)) || actionsEl.lastElementChild;
  if (focusEl) setTimeout(() => focusEl.focus(), 0);
  return new Promise(resolve => { modalResolve = resolve; });
}

function confirmModal(title, message, confirmLabel = 'Confirm', danger = false) {
  return openModal({
    title, body: message,
    actions: [
      { label: 'Cancel', value: false },
      { label: confirmLabel, value: true, kind: danger ? 'red' : 'white' },
    ],
  }).then(v => v === true);
}

function promptModal(title, label, { value = '', placeholder = '', multiline = false } = {}) {
  const wrap = document.createElement('label');
  wrap.className = 'modal-field';
  const span = document.createElement('span');
  span.textContent = label;
  const input = document.createElement(multiline ? 'textarea' : 'input');
  if (!multiline) input.type = 'text';
  input.value = value;
  input.placeholder = placeholder;
  input.className = 'modal-input' + (multiline ? ' raw-textarea' : '');
  wrap.append(span, input);
  input.addEventListener('keydown', e => {
    if (e.key === 'Enter' && (!multiline || e.ctrlKey || e.metaKey)) {
      e.preventDefault();
      closeModal(input.value);
    }
  });
  return openModal({
    title, body: wrap, focusSelector: '.modal-input',
    actions: [
      { label: 'Cancel', value: null },
      { label: 'OK', value: () => input.value, kind: 'red' },
    ],
  });
}

document.addEventListener('keydown', e => {
  if (e.key === 'Escape' && !$('modal-backdrop').hidden) closeModal(null);
});
$('modal-backdrop').addEventListener('click', e => {
  if (e.target === $('modal-backdrop')) closeModal(null);
});

// ==========================================
// TABS & KEYBINDINGS
// ==========================================
function showTab(tabId) {
  document.querySelectorAll('.nav-tab').forEach(b => b.classList.remove('active'));
  document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
  $('tabbtn-' + tabId).classList.add('active');
  $('tab-' + tabId).classList.add('active');
  if (tabId === 'todo') loadTodoDetails();
  else if (tabId === 'worktrees') refreshBranches();
}

function setTodoView(mode) {
  document.querySelectorAll('.todo-view-btn').forEach(b => b.classList.remove('active'));
  document.querySelectorAll('.view-pane').forEach(p => p.classList.remove('active'));
  $('viewbtn-' + mode).classList.add('active');
  $('pane-' + mode).classList.add('active');
  if (mode === 'split') lintSpecNow();
}

document.addEventListener('keydown', e => {
  if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 's') {
    e.preventDefault();
    if ($('tab-todo').classList.contains('active')) saveRawTodo();
  }
});

window.addEventListener('beforeunload', e => {
  if (editorDirty) { e.preventDefault(); e.returnValue = ''; }
});

// ==========================================
// DASHBOARD / TELEMETRY
// ==========================================
function animateNumber(elementId, newValue) {
  const el = $(elementId);
  if (!el) return;
  const currentVal = parseInt(el.dataset.val || el.textContent.replace(/,/g, ''), 10) || 0;
  if (currentVal === newValue) return;
  el.dataset.val = newValue;
  const start = performance.now();
  function step(ts) {
    const progress = Math.min((ts - start) / 500, 1);
    const eased = 1 - Math.pow(1 - progress, 3);
    el.textContent = Math.floor(currentVal + (newValue - currentVal) * eased).toLocaleString();
    if (progress < 1) requestAnimationFrame(step);
    else el.textContent = newValue.toLocaleString();
  }
  requestAnimationFrame(step);
}

function updateStatusUI() {
  const running = isRunning();
  const stopping = running && currentState.stop_pending;
  const offline = currentState.status === 'offline';
  $('status-text').textContent = offline ? 'offline' : (stopping ? 'stopping' : (running ? 'running' : (currentState.status || 'idle')));
  $('status-pill').className = 'status-pill' + (running ? ' running' : '') + (stopping ? ' stopping' : '') + (offline ? ' offline' : '');
  $('led-indicator').className = 'led-dot' + (running ? ' running' : '');

  const btn = $('btn-toggle-run');
  if (stopping) {
    btn.textContent = '⏳ STOPPING AFTER ITEM…';
    btn.className = 'btn-stop';
    btn.disabled = true;
    btn.title = 'Stop is honoured between items - the current item will finish first.';
  } else if (running) {
    btn.textContent = '⏹ STOP LOOP';
    btn.className = 'btn-stop';
    btn.disabled = false;
    btn.title = 'Stop after the current item finishes';
  } else {
    btn.textContent = '▶ RUN LOOP';
    btn.className = 'btn-red';
    btn.disabled = offline;
    btn.title = '';
  }
  // Actions the server refuses mid-run are disabled up front, with the reason on hover.
  document.querySelectorAll('[data-idle-only]').forEach(el => {
    el.disabled = running;
    el.title = running ? 'Stop the runner first' : (el.dataset.idleTitle || '');
  });
}

function updateHud(state) {
  const tokens = state.tokens || {};
  $('hud-tps').innerHTML = escapeHtml(tokens.tps ?? 0) + ' <span style="font-size: 0.9rem; font-weight: 500; color: var(--text-muted)">t/s</span>';
  $('hud-peak-tps').textContent = tokens.peak_tps ?? 0;
  $('hud-tps-arc').style.strokeDashoffset = 125.66 - 125.66 * Math.min((tokens.tps || 0) / 100, 1);
  animateNumber('hud-total-tokens', tokens.total || 0);
  animateNumber('hud-prompt-tokens', tokens.prompt || 0);
  animateNumber('hud-comp-tokens', tokens.completion || 0);

  const byModel = Object.entries(tokens.by_model || {});
  const container = $('hud-tokens-models');
  if (byModel.length === 0) {
    container.innerHTML = '<span class="hud-empty">No model usage recorded yet</span>';
  } else {
    const totalAll = tokens.total || 1;
    byModel.sort((a, b) => (b[1].total || 0) - (a[1].total || 0));
    container.innerHTML = byModel.map(([name, u]) => {
      const pct = Math.min(100, Math.max(1, Math.round(((u.total || 0) / totalAll) * 100)));
      return `
        <div class="model-usage">
          <div class="model-usage-head">
            <span class="model-usage-name" title="${escapeHtml(name)}">${escapeHtml(name.split('/').pop())}</span>
            <span class="model-usage-total">${(u.total || 0).toLocaleString()} <span>(${pct}%)</span></span>
          </div>
          <div class="model-usage-split"><span>Prompt: ${(u.prompt || 0).toLocaleString()}</span><span>Eval: ${(u.completion || 0).toLocaleString()}</span></div>
          <div class="model-usage-bar"><div style="width: ${pct}%"></div></div>
        </div>`;
    }).join('');
  }

  const gpu = state.gpu || {};
  const memUsed = gpu.used_mb || 0;
  const memTotal = gpu.total_mb || 1;
  $('hud-gpu-mem-fill').style.width = Math.min((memUsed / memTotal) * 100, 100) + '%';
  $('hud-gpu-temp-fill').style.width = Math.min(((gpu.temp_c || 0) / 90) * 100, 100) + '%';
  if (!gpu.name || gpu.name === 'N/A') {
    $('hud-gpu-name').textContent = 'NO GPU DETECTED';
    $('hud-gpu-mem-text').textContent = '---';
    $('hud-gpu-temp-text').textContent = '---';
  } else {
    $('hud-gpu-name').textContent = `${gpu.name} (${gpu.power_w || 0}W)`;
    $('hud-gpu-mem-text').textContent = `${memUsed}M / ${memTotal}M`;
    $('hud-gpu-temp-text').textContent = `${gpu.temp_c || 0}°C`;
  }

  const ollama = state.ollama || {};
  $('hud-loaded-model').textContent = ollama.loaded_model || 'None';
  $('hud-ctx').textContent = (ollama.context || 0).toLocaleString();
}

const STEPPER_STAGES = ['idle', 'worktree', 'coding', 'validating', 'merged', 'parked'];

function updateStepper() {
  const task = currentState.current_task || {};
  const stage = task.stage || 'idle';
  STEPPER_STAGES.forEach(s => {
    $('step-' + s).classList.toggle('active', s === stage);
  });
  $('stepper-task-text').textContent = task.text && stage !== 'idle'
    ? `Item ${task.index}: ${task.text.split('\n')[0].substring(0, 70)}`
    : '';

  const note = $('stepper-note');
  if (stage === 'parked') {
    note.hidden = false;
    note.innerHTML = '';
    const span = document.createElement('span');
    span.textContent = `Item ${task.index} did not pass - its work is on a parked branch.`;
    const btn = document.createElement('button');
    btn.textContent = 'Review →';
    btn.addEventListener('click', () => showTab('worktrees'));
    note.append(span, btn);
  } else {
    note.hidden = true;
  }

  clearInterval(elapsedTimer);
  const inProgress = ['worktree', 'coding', 'validating'].includes(stage) && task.started_at;
  const tick = () => {
    if (!inProgress) { $('stepper-elapsed').textContent = ''; return; }
    const secs = Math.max(0, Math.floor((Date.now() - Date.parse(task.started_at)) / 1000));
    const m = Math.floor(secs / 60), s = secs % 60;
    $('stepper-elapsed').textContent = `${m}:${String(s).padStart(2, '0')}`;
  };
  tick();
  if (inProgress) elapsedTimer = setInterval(tick, 1000);
}

function updateUI(state) {
  updateStatusUI();
  updateHud(state);
  updateStepper();
  populateModelSelects((state.ollama && state.ollama.available_models) || []);
}

// ==========================================
// MODEL PICKERS (persisted per project)
// ==========================================
function prefsKey() {
  return 'controls.' + (currentState.project_dir || 'default');
}

function savedControls() {
  return prefs.get(prefsKey(), {});
}

function saveControl(id, value) {
  const saved = savedControls();
  saved[id] = value;
  prefs.set(prefsKey(), saved);
}

function populateModelSelects(availableModels) {
  const saved = savedControls();
  const models = availableModels.length ? availableModels : FALLBACK_OLLAMA_MODELS;
  MODEL_PICKERS.forEach(p => {
    const sel = $(p.id);
    if (!sel) return;
    const wanted = sel.dataset.userSelected ?? saved[p.id] ?? p.defaultVal;
    const sig = models.join(',') + '|' + wanted;
    if (sel.dataset.lastSignature === sig) return;

    let html = p.includeNone ? '<option value="">None (Disabled)</option>' : '';
    html += '<optgroup label="CLI & Cloud Engines">' +
      CLI_MODEL_OPTIONS.map(o => `<option value="${escapeHtml(o.val)}">${escapeHtml(o.text)}</option>`).join('') +
      '</optgroup>';
    const localModels = models.includes(wanted) || !wanted || CLI_MODEL_OPTIONS.some(o => o.val === wanted)
      ? models : [wanted, ...models];  // keep a saved model selectable even if Ollama is down
    html += '<optgroup label="Local Ollama Models">' +
      localModels.map(m => `<option value="${escapeHtml(m)}">${escapeHtml(m)}</option>`).join('') +
      '</optgroup>';
    sel.innerHTML = html;
    sel.value = wanted;
    if (sel.value !== wanted) sel.value = p.defaultVal;
    sel.dataset.lastSignature = sig;
  });
}

function restoreSimpleControls() {
  const saved = savedControls();
  if (saved['run-backend']) $('run-backend').value = saved['run-backend'];
  if (saved['run-limit']) $('run-limit').value = saved['run-limit'];
}

PERSISTED_CONTROLS.forEach(id => {
  $(id).addEventListener('change', () => {
    const el = $(id);
    if (el.tagName === 'SELECT') el.dataset.userSelected = el.value;
    saveControl(id, el.value);
  });
});

// ==========================================
// CONSOLE
// ==========================================
function appendLog(line) {
  const logs = $('terminal-logs');
  const msg = typeof line === 'string' ? line : (line.msg || '');
  const time = (line && line.time) || new Date().toTimeString().split(' ')[0];

  let cls = 'log-msg';
  if (/Merged|Validation passed/.test(msg)) cls += ' pass';
  if (/ERROR|error|failed|Parked|exited \([^0N]/.test(msg)) cls += ' error';
  if (/=== Item|Worktree/.test(msg)) cls += ' highlight';

  const row = document.createElement('div');
  row.className = 'log-line';
  row.innerHTML = `<span class="log-time">${escapeHtml(time)}</span><span class="${cls}">${escapeHtml(msg)}</span>`;

  const atBottom = logs.scrollHeight - logs.clientHeight <= logs.scrollTop + 25;
  logs.appendChild(row);
  while (logs.childElementCount > MAX_LOG_ROWS) logs.firstElementChild.remove();
  if ($('terminal-autoscroll').checked && atBottom) logs.scrollTop = logs.scrollHeight;
}

// ==========================================
// LIVE EVENTS
// ==========================================
const refreshTodoSoon = debounce(() => loadTodoDetails(), 250);
const refreshBranchesSoon = debounce(() => refreshBranches(), 400);

function initSSE() {
  const source = new EventSource('/api/events');
  const on = (name, fn) => source.addEventListener(name, e => {
    try { fn(JSON.parse(e.data)); } catch (err) { console.error('SSE ' + name, err); }
  });

  on('init', data => {
    const wasOffline = currentState.status === 'offline';
    currentState = data;
    // The server replays recent logs on every (re)connect; start clean so
    // a dropped connection doesn't duplicate the whole console.
    $('terminal-logs').innerHTML = '';
    (data.recent_logs || []).forEach(appendLog);
    updateUI(currentState);
    if (wasOffline) {
      showToast('Reconnected to Studio server', { kind: 'success' });
      refreshTodoSoon();
      refreshBranchesSoon();
    }
  });
  on('status', data => {
    Object.assign(currentState, data);
    updateStatusUI();
    renderQueueAndChecklist();
  });
  on('stage', task => {
    const previous = lastStage;
    currentState.current_task = task;
    lastStage = task.stage;
    updateStepper();
    renderQueueAndChecklist();
    refreshTodoSoon();
    if (['merged', 'parked', 'idle'].includes(task.stage) && previous !== task.stage) refreshBranchesSoon();
  });
  on('todo_updated', details => renderTodoDetails(details));
  on('branches_changed', () => refreshBranchesSoon());
  on('project_change', () => location.reload());
  on('tokens', tokens => { currentState.tokens = tokens; updateHud(currentState); });
  on('gpu', gpu => { currentState.gpu = gpu; updateHud(currentState); });
  on('ollama', ollama => {
    currentState.ollama = ollama;
    updateHud(currentState);
    populateModelSelects(ollama.available_models || []);
  });
  on('log', appendLog);

  source.onerror = () => {
    if (currentState.status !== 'offline') {
      currentState.status = 'offline';
      currentState.is_running = false;
      updateStatusUI();
    }
  };
}

// ==========================================
// RUN CONTROLS
// ==========================================
async function startOllama() {
  showToast('Starting Ollama & refreshing models…');
  try {
    const data = await api('/api/ollama/start', { method: 'POST' });
    currentState.ollama = Object.assign(currentState.ollama || {}, { available_models: data.models || [] });
    populateModelSelects(data.models || []);
    showToast(data.models && data.models.length ? `Ollama online - ${data.models.length} model(s)` : 'Ollama started, but reported no models', { kind: 'success' });
  } catch (err) {
    showError(err, 'Ollama');
  }
}

async function toggleRunner() {
  const btn = $('btn-toggle-run');
  btn.disabled = true;
  try {
    if (isRunning()) {
      const res = await api('/api/stop', { method: 'POST' });
      currentState.stop_pending = true;
      showToast(res.message || 'Stopping after the current item');
    } else {
      await api('/api/run', {
        method: 'POST',
        body: {
          backend: $('run-backend').value,
          model: $('run-model').value,
          review: $('review-model').value,
          fallback_model: $('escalate-model').value,
          max_items: parseInt($('run-limit').value, 10) || 1,
        },
      });
      currentState.is_running = true;
      showToast('Forge runner loop started', { kind: 'success' });
    }
  } catch (err) {
    showError(err, 'Runner');
  } finally {
    updateStatusUI();
  }
}

// ==========================================
// CHECKLIST (dashboard queue + full editor)
// ==========================================
const STATUS_META = {
  ' ': { cls: 'status-open', label: '[ ] PENDING' },
  'x': { cls: 'status-done', label: '[x] DONE' },
  'X': { cls: 'status-done', label: '[x] DONE' },
  '!': { cls: 'status-blocked', label: '[!] BLOCKED' },
  '?': { cls: 'status-review', label: '[?] NEEDS REVIEW' },
};
const isDone = it => it.status === 'x' || it.status === 'X';
const isStuck = it => it.status === '!' || it.status === '?';

function isActiveItem(it) {
  const task = currentState.current_task || {};
  return isRunning() && task.text && ['worktree', 'coding', 'validating'].includes(task.stage) && task.text === it.text;
}

async function loadTodoDetails() {
  try {
    renderTodoDetails(await api('/api/todo'));
  } catch (err) {
    showError(err, 'Loading checklist');
  }
}

function setEditorsContent(content) {
  $('raw-todo-editor').value = content;
  $('split-todo-editor').value = content;
  updateEditorMeta();
}

function updateEditorMeta() {
  const lines = ($('raw-todo-editor').value || '').split('\n').length;
  $('editor-line-count').textContent = lines + ' lines' + (editorDirty ? ' · ● unsaved' : '');
  $('editor-line-count').classList.toggle('unsaved', editorDirty);
}

function showSyncBanner(text) {
  $('todo-sync-text').textContent = text;
  $('todo-sync-banner').hidden = false;
}

function renderTodoDetails(details) {
  todoDetails = details;
  $('todo-display-filename').textContent = details.filename || 'TODO.md';
  $('todo-display-filepath').textContent = 'Path: ' + (details.filepath || '');
  $('badge-todo-count').textContent = (details.items || []).length;

  if (!editorDirty) {
    // Nothing unsaved: always track the file, including the runner's status writes.
    todoBase = { content: details.content || '', hash: details.hash };
    setEditorsContent(details.content || '');
    $('todo-sync-banner').hidden = true;
  } else if (todoBase && details.hash !== todoBase.hash) {
    // Never overwrite unsaved edits; say what happened instead.
    showSyncBanner(`${details.filename || 'TODO.md'} changed on disk (e.g. the runner marked an item) while you have unsaved edits.`);
  }
  renderQueueAndChecklist();
}

function renderQueueAndChecklist() {
  if (!todoDetails) return;
  const items = todoDetails.items || [];
  renderDashboardTodoQueue(items);
  renderChecklistCards(items);
}

function statusChip(it) {
  const meta = STATUS_META[it.status] || STATUS_META[' '];
  return `<span class="status-chip ${meta.cls}">${meta.label}</span>`;
}

function fileChips(it) {
  return (it.files || []).map(f => `<span class="file-chip">${escapeHtml(f)}</span>`).join('');
}

function renderDashboardTodoQueue(items) {
  const container = $('dashboard-todo-queue');
  const counts = { open: 0, done: 0, stuck: 0 };
  items.forEach(it => { if (isDone(it)) counts.done++; else if (isStuck(it)) counts.stuck++; else counts.open++; });
  $('queue-counts').textContent = items.length ? `${counts.open} open · ${counts.stuck} stuck · ${counts.done} done` : '';
  $('btn-requeue-dash').hidden = counts.stuck === 0;

  container.innerHTML = '';
  if (items.length === 0) {
    container.innerHTML = '<div class="empty-note">No items in checklist.</div>';
    return;
  }
  // What the runner still has left, in file order - not the whole file.
  const pending = items.filter(it => !isDone(it));
  if (pending.length === 0) {
    container.innerHTML = '<div class="empty-note">All items done.</div>';
    return;
  }
  pending.slice(0, 8).forEach(it => {
    const row = document.createElement('div');
    row.className = 'todo-queue-item' + (isActiveItem(it) ? ' is-active' : '') + (isStuck(it) ? ' is-stuck' : '');
    row.innerHTML = `
      <button class="todo-checkbox" data-action="done" title="Mark done"></button>
      <div style="flex: 1; min-width: 0;">
        <div class="todo-item-title">${isActiveItem(it) ? '<span class="live-tag">RUNNING</span>' : ''}${escapeHtml(it.first_line)}</div>
        <div class="todo-item-files">${statusChip(it)}${fileChips(it)}</div>
      </div>
      ${isStuck(it) ? '<button class="mini-btn" data-action="requeue" title="Back to [ ] for the next run">↺</button>' : ''}`;
    row.querySelector('[data-action="done"]').addEventListener('click', () => setItemStatus(it.line, 'x'));
    const rq = row.querySelector('[data-action="requeue"]');
    if (rq) rq.addEventListener('click', () => requeueItems([it.line]));
    container.appendChild(row);
  });
  if (pending.length > 8) {
    const more = document.createElement('button');
    more.className = 'queue-more';
    more.textContent = `+ ${pending.length - 8} more in the full editor →`;
    more.addEventListener('click', () => showTab('todo'));
    container.appendChild(more);
  }
}

function renderChecklistCards(items) {
  const container = $('checklist-items-container');
  container.innerHTML = '';
  if (items.length === 0) {
    container.innerHTML = '<div class="empty-note" style="padding: 3rem 0;">No tasks found. Click "+ Add Task" or use the AI Spec Architect.</div>';
    return;
  }
  items.forEach(it => {
    const meta = STATUS_META[it.status] || STATUS_META[' '];
    const parts = it.text.split('```');
    const card = document.createElement('div');
    card.className = `task-card ${meta.cls}` + (isActiveItem(it) ? ' is-active' : '');
    card.innerHTML = `
      <button class="todo-checkbox ${isDone(it) ? 'checked' : ''}" data-action="toggle" style="width: 20px; height: 20px; font-size: 0.8rem;" title="${isDone(it) ? 'Mark open' : 'Mark done'}">${isDone(it) ? '✓' : ''}</button>
      <div class="task-body">
        <div class="task-header">
          <span class="task-line-badge">LINE ${it.line} // TASK #${it.index}${isActiveItem(it) ? ' <span class="live-tag">RUNNING</span>' : ''}</span>
          <div style="display: flex; gap: 0.4rem; align-items: center;">
            ${statusChip(it)}
            ${isStuck(it) ? '<button class="mini-btn" data-action="requeue" title="Back to [ ] for the next run">↺ Requeue</button>' : ''}
            <button class="mini-btn" data-action="status">Status ▾</button>
          </div>
        </div>
        <div class="task-prompt">${escapeHtml(parts[0])}</div>
        ${parts.length >= 3 ? `<pre class="task-spec-code"><code>${escapeHtml(parts[1])}</code></pre>` : ''}
        <div class="todo-item-files" style="margin-top: 0.6rem;">${fileChips(it)}</div>
      </div>`;
    card.querySelector('[data-action="toggle"]').addEventListener('click', () => setItemStatus(it.line, isDone(it) ? ' ' : 'x'));
    card.querySelector('[data-action="status"]').addEventListener('click', () => chooseItemStatus(it));
    const rq = card.querySelector('[data-action="requeue"]');
    if (rq) rq.addEventListener('click', () => requeueItems([it.line]));
    container.appendChild(card);
  });
}

async function setItemStatus(line, status) {
  if (editorDirty) {
    showToast('Save or discard your unsaved checklist edits first - a status change rewrites the file.', { kind: 'error' });
    return;
  }
  try {
    const res = await api('/api/todo/toggle', { method: 'POST', body: { line, status } });
    renderTodoDetails(res.details);
  } catch (err) {
    showError(err, 'Status change');
  }
}

async function chooseItemStatus(it) {
  const choice = await openModal({
    title: `Set status - line ${it.line}`,
    body: it.first_line,
    actions: [
      { label: 'Cancel', value: null },
      { label: '[ ] Open', value: ' ' },
      { label: '[?] Review', value: '?' },
      { label: '[!] Blocked', value: '!' },
      { label: '[x] Done', value: 'x', kind: 'white' },
    ],
  });
  if (choice !== null && choice !== it.status) setItemStatus(it.line, choice);
}

async function requeueItems(lines) {
  if (isRunning()) {
    showToast('Stop the runner before requeuing items', { kind: 'error' });
    return;
  }
  if (editorDirty) {
    showToast('Save or discard your unsaved checklist edits first.', { kind: 'error' });
    return;
  }
  try {
    const res = await api('/api/todo/requeue-stuck', { method: 'POST', body: lines ? { lines } : {} });
    renderTodoDetails(res.details);
    showToast(res.count > 0 ? `Requeued ${res.count} item(s)` : 'No stuck items to requeue', { kind: res.count ? 'success' : 'info' });
  } catch (err) {
    showError(err, 'Requeue');
  }
}

async function requeueStuckItems() {
  const stuck = ((todoDetails && todoDetails.items) || []).filter(isStuck).length;
  if (stuck === 0) {
    showToast('No [!] or [?] items to requeue');
    return;
  }
  const ok = await confirmModal('Requeue stuck items',
    `Move ${stuck} blocked/needs-review item(s) back to [ ] so the next run retries them? Pick a different model first if the last one struggled.`,
    'Requeue ' + stuck);
  if (ok) requeueItems(null);
}

['raw-todo-editor', 'split-todo-editor'].forEach(id => {
  $(id).addEventListener('input', () => {
    const other = id === 'raw-todo-editor' ? 'split-todo-editor' : 'raw-todo-editor';
    $(other).value = $(id).value;
    editorDirty = !todoBase || $(id).value !== todoBase.content;
    updateEditorMeta();
    if ($('pane-split').classList.contains('active')) lintSoon();
  });
});

async function saveRawTodo(force = false) {
  if (isRunning()) {
    showToast('Stop the runner before saving the whole checklist - it may be writing status marks.', { kind: 'error' });
    return;
  }
  const content = $('raw-todo-editor').value;
  try {
    const res = await api('/api/todo', {
      method: 'POST',
      body: { content, base_hash: todoBase ? todoBase.hash : null, force: force === true },
    });
    editorDirty = false;
    renderTodoDetails(res.details);
    showToast(`${res.filename} saved`, { kind: 'success' });
    if ($('pane-split').classList.contains('active')) lintSpecNow();
  } catch (err) {
    if (err.status === 409 && err.payload.conflict) {
      showSyncBanner(err.message + ' Choose which version to keep.');
    } else {
      showError(err, 'Save');
    }
  }
}

async function reloadTodoFromDisk(discard = false) {
  if (editorDirty && !discard) {
    const ok = await confirmModal('Discard unsaved edits?', 'Reloading replaces your unsaved checklist edits with the file on disk.', 'Discard & reload', true);
    if (!ok) return;
  }
  editorDirty = false;
  $('todo-sync-banner').hidden = true;
  loadTodoDetails();
}

async function openAddTaskModal() {
  const task = await promptModal('Add checklist task', 'Task in Forge format - name files in backticks. Ctrl+Enter to add.', {
    placeholder: '- [ ] In `src/app.py`, implement feature', multiline: true,
  });
  if (!task || !task.trim()) return;
  let text = task.trim();
  if (!text.startsWith('- [')) text = '- [ ] ' + text;
  try {
    const res = await api('/api/todo/add', { method: 'POST', body: { task: text } });
    if (!editorDirty) renderTodoDetails(res.details);
    showToast('Task appended to checklist', { kind: 'success' });
  } catch (err) {
    showError(err, 'Add task');
  }
}

// ==========================================
// LINTER & SPEC ARCHITECT
// ==========================================
function renderLint(container, badge, res) {
  container.innerHTML = '';
  let fatals = 0;
  (res.results || []).forEach(it => {
    const hasFatal = it.issues.some(x => x.fatal);
    if (hasFatal) fatals++;
    const card = document.createElement('div');
    card.className = 'lint-card' + (hasFatal ? ' fatal' : '');
    const issues = it.issues.length === 0
      ? '<div class="lint-badge-clean">✓ Valid Task Shape</div>'
      : it.issues.map(iss => `<div class="${iss.fatal ? 'lint-badge-fatal' : 'lint-badge-clean'}">${iss.fatal ? '🔴 Fatal: ' : '🟡 Advisory: '}${escapeHtml(iss.message)}</div>`).join('');
    card.innerHTML = `
      <div style="font-size: 0.74rem; font-weight: 700; font-family: var(--font-mono);">
        <span style="color: var(--text-dim)">L${it.line}:</span> [${escapeHtml(it.status)}] ${escapeHtml(it.first_line)}
      </div>${issues}`;
    container.appendChild(card);
  });
  if (!res.total_items) {
    badge.textContent = 'No items';
    badge.style.color = 'var(--text-muted)';
  } else if (fatals > 0) {
    badge.textContent = fatals + ' Fatal Error' + (fatals > 1 ? 's' : '');
    badge.style.color = 'var(--red-bright)';
  } else {
    badge.textContent = '✓ All Tasks Valid';
    badge.style.color = '#ffffff';
  }
  return fatals;
}

async function lintSpecNow() {
  try {
    const res = await api('/api/lint-spec', { method: 'POST', body: { content: $('split-todo-editor').value } });
    renderLint($('linter-results-split'), $('linter-summary-badge'), res);
  } catch (err) {
    showError(err, 'Lint');
  }
}
const lintSoon = debounce(lintSpecNow, 600);

async function draftGoal() {
  const goal = $('goal-input').value.trim();
  if (!goal) {
    showToast('Enter a goal description first.', { kind: 'error' });
    return;
  }
  const btn = $('btn-draft-goal');
  btn.textContent = 'Architect reasoning…';
  btn.disabled = true;
  try {
    const res = await api('/api/draft-spec', { method: 'POST', body: { goal, model: $('draft-model').value } });
    $('draft-editor').value = res.draft;
    $('draft-model-label').textContent = res.model;
    $('draft-review').hidden = false;
    renderLint($('draft-lint-results'), $('draft-lint-badge'), res.lint || { results: [] });
    $('draft-editor').focus();
  } catch (err) {
    showError(err, 'Drafting');
  } finally {
    btn.textContent = 'Draft Tasks with AI';
    btn.disabled = false;
  }
}

async function lintDraft() {
  try {
    const res = await api('/api/lint-spec', { method: 'POST', body: { content: $('draft-editor').value } });
    return renderLint($('draft-lint-results'), $('draft-lint-badge'), res);
  } catch (err) {
    showError(err, 'Lint');
    return 0;
  }
}

function discardDraft() {
  $('draft-review').hidden = true;
  $('draft-editor').value = '';
}

async function acceptDraft() {
  const draft = $('draft-editor').value.trim();
  if (!draft) return;
  const fatals = await lintDraft();
  if (fatals > 0) {
    const ok = await confirmModal('Draft has fatal lint errors',
      `${fatals} drafted item(s) would run with key checks disabled. Append anyway?`, 'Append anyway', true);
    if (!ok) return;
  }
  try {
    const res = await api('/api/todo/add', { method: 'POST', body: { task: draft } });
    if (!editorDirty) renderTodoDetails(res.details);
    discardDraft();
    $('goal-input').value = '';
    setTodoView('checklist');
    showToast('Drafted items appended to checklist', { kind: 'success' });
  } catch (err) {
    showError(err, 'Append draft');
  }
}

// ==========================================
// PARKED BRANCHES & DIFF REVIEWER
// ==========================================
async function refreshBranches() {
  try {
    const data = await api('/api/branches');
    branches = data.branches || [];
  } catch (err) {
    showError(err, 'Loading branches');
    return;
  }
  $('badge-branches-count').textContent = branches.length;
  renderSidebarBranches();
  renderReviewBranches();
  if (activeReviewBranch && !branches.some(b => b.branch === activeReviewBranch)) clearReviewSelection();
}

function branchStatusChip(b) {
  const meta = STATUS_META[b.status];
  return meta ? `<span class="status-chip ${meta.cls}">${meta.label}</span>` : '<span class="status-chip">NO RECORD</span>';
}

function branchCardHtml(b) {
  return `
    <div class="branch-title">${escapeHtml(b.item_title || b.branch)}</div>
    <div class="branch-meta">${branchStatusChip(b)}${b.diffstat ? `<span class="branch-stat">${escapeHtml(b.diffstat)}</span>` : ''}</div>
    ${b.reason ? `<div class="branch-reason">${escapeHtml(b.reason)}</div>` : ''}
    ${b.item_title ? `<div class="branch-name">${escapeHtml(b.branch)}</div>` : ''}`;
}

function renderSidebarBranches() {
  const list = $('branch-list');
  list.innerHTML = '';
  if (branches.length === 0) {
    list.innerHTML = '<div class="empty-note">No parked branches. Clean checkout!</div>';
    return;
  }
  branches.forEach(b => {
    const card = document.createElement('div');
    card.className = 'branch-card';
    card.innerHTML = branchCardHtml(b) + `
      <div class="branch-actions">
        <button class="mini-btn" data-action="review">Review</button>
        <button class="mini-btn btn-red" data-action="merge" data-idle-only>Merge</button>
        <button class="mini-btn" data-action="discard" data-idle-only>Discard</button>
      </div>`;
    card.querySelector('[data-action="review"]').addEventListener('click', () => { showTab('worktrees'); selectReviewBranch(b.branch); });
    card.querySelector('[data-action="merge"]').addEventListener('click', () => mergeBranch(b.branch));
    card.querySelector('[data-action="discard"]').addEventListener('click', () => discardBranch(b.branch));
    list.appendChild(card);
  });
  updateStatusUI();
}

function renderReviewBranches() {
  const list = $('review-branch-list');
  list.innerHTML = '';
  if (branches.length === 0) {
    list.innerHTML = '<div class="empty-note" style="padding: 2rem 0;">No parked worktrees. Working tree clean!</div>';
    return;
  }
  branches.forEach(b => {
    const card = document.createElement('div');
    card.className = 'branch-card selectable' + (b.branch === activeReviewBranch ? ' selected' : '');
    card.tabIndex = 0;
    card.innerHTML = branchCardHtml(b);
    card.addEventListener('click', () => selectReviewBranch(b.branch));
    card.addEventListener('keydown', e => { if (e.key === 'Enter') selectReviewBranch(b.branch); });
    list.appendChild(card);
  });
}

function clearReviewSelection() {
  activeReviewBranch = null;
  $('diff-branch-title').textContent = 'Select a branch to review';
  $('diff-stat-badge').textContent = '';
  ['btn-critic', 'btn-merge-active', 'btn-discard-active'].forEach(id => { $(id).disabled = true; });
  $('critic-result-container').innerHTML = '';
  $('diff-code-view').textContent = 'Select a parked worktree branch from the left panel to inspect its diff and trigger an automated adversarial code review.';
  renderReviewBranches();
}

async function selectReviewBranch(branch) {
  activeReviewBranch = branch;
  const b = branches.find(x => x.branch === branch) || { branch };
  $('diff-branch-title').textContent = b.item_title || branch;
  $('diff-stat-badge').textContent = b.diffstat || '';
  $('critic-result-container').innerHTML = '';
  $('diff-code-view').textContent = 'Loading diff…';
  $('btn-critic').disabled = false;
  $('btn-merge-active').disabled = isRunning();
  $('btn-discard-active').disabled = isRunning();
  renderReviewBranches();
  try {
    const data = await api('/api/diff?branch=' + encodeURIComponent(branch));
    if (activeReviewBranch !== branch) return;  // a different branch was picked meanwhile
    renderDiff(data.diff || '(Empty diff)');
  } catch (err) {
    if (activeReviewBranch === branch) $('diff-code-view').textContent = 'Could not load diff: ' + err.message;
  }
}

function renderDiff(diffText) {
  const view = $('diff-code-view');
  view.innerHTML = diffText.split('\n').map(l => {
    let cls = '';
    if (l.startsWith('+++') || l.startsWith('---') || l.startsWith('diff') || l.startsWith('index')) cls = 'meta';
    else if (l.startsWith('+')) cls = 'add';
    else if (l.startsWith('-')) cls = 'del';
    else if (l.startsWith('@@')) cls = 'hunk';
    return `<span class="diff-line ${cls}">${escapeHtml(l) || ' '}</span>`;
  }).join('');
  view.scrollTop = 0;
}

async function runCritic() {
  if (!activeReviewBranch) return;
  const branch = activeReviewBranch;
  const model = $('review-model').value || $('run-model').value;
  const btn = $('btn-critic');
  btn.textContent = 'Critic analyzing…';
  btn.disabled = true;
  try {
    const res = await api('/api/critic', { method: 'POST', body: { branch, model } });
    if (activeReviewBranch !== branch) return;
    const cls = res.verdict === 'APPROVE' ? 'approve' : (res.verdict === 'REJECT' ? 'reject' : 'caution');
    $('critic-result-container').innerHTML = `
      <div class="critic-card verdict-${cls}">
        <div class="critic-verdict">VERDICT: ${escapeHtml(res.verdict)} <span>${escapeHtml(res.model || model)}</span></div>
        <div class="critic-body"></div>
      </div>`;
    $('critic-result-container').querySelector('.critic-body').textContent = res.critique;
  } catch (err) {
    showError(err, 'Critic');
  } finally {
    btn.textContent = '🤖 Critic Review';
    btn.disabled = !activeReviewBranch;
  }
}

async function mergeBranch(branch) {
  const b = branches.find(x => x.branch === branch) || {};
  const ok = await confirmModal('Merge parked branch',
    `Merge "${b.item_title || branch}" into the current branch? Its checklist item will be marked [x].`, 'Merge', true);
  if (!ok) return;
  try {
    const res = await api('/api/merge', { method: 'POST', body: { branch } });
    showToast('Merged ' + (b.item_title || branch) + (res.item_marked_done ? ' - item marked [x]' : ''), { kind: 'success' });
    if (activeReviewBranch === branch) clearReviewSelection();
  } catch (err) {
    showError(err, 'Merge');
  }
  refreshBranches();
}

async function discardBranch(branch, { skipConfirm = false } = {}) {
  const b = branches.find(x => x.branch === branch) || {};
  if (!skipConfirm) {
    const ok = await confirmModal('Discard parked branch',
      `Permanently delete ${branch}? The model's work on "${b.item_title || 'this item'}" is lost; the checklist item keeps its status so you can requeue it.`,
      'Delete branch', true);
    if (!ok) return false;
  }
  try {
    await api('/api/discard', { method: 'POST', body: { branch } });
    if (activeReviewBranch === branch) clearReviewSelection();
    if (!skipConfirm) showToast('Discarded ' + branch, { kind: 'success' });
    return true;
  } catch (err) {
    showError(err, 'Discard');
    return false;
  } finally {
    if (!skipConfirm) refreshBranches();
  }
}

async function discardAllBranches() {
  if (branches.length === 0) {
    showToast('No parked branches');
    return;
  }
  if (isRunning()) {
    showToast('Stop the runner first', { kind: 'error' });
    return;
  }
  const ok = await confirmModal('Discard all parked branches',
    `Permanently delete all ${branches.length} parked branch(es)? Their checklist items keep their status, so "Requeue Stuck" can retry them.`,
    `Delete ${branches.length}`, true);
  if (!ok) return;
  let done = 0;
  for (const b of [...branches]) {
    if (await discardBranch(b.branch, { skipConfirm: true })) done++;
  }
  showToast(`Discarded ${done} branch(es)`, { kind: 'success' });
  refreshBranches();
}

function mergeActiveBranch() { if (activeReviewBranch) mergeBranch(activeReviewBranch); }
function discardActiveBranch() { if (activeReviewBranch) discardBranch(activeReviewBranch); }

// ==========================================
// PROJECTS
// ==========================================
function recentProjects() {
  return prefs.get('recentProjects', []);
}

function rememberProject(path) {
  prefs.set('recentProjects', [path, ...recentProjects().filter(p => p !== path)].slice(0, 12));
}

async function fetchProjects() {
  let projects = [];
  try {
    projects = (await api('/api/projects')).projects || [];
  } catch (err) {
    showError(err, 'Loading projects');
  }
  const all = [...new Set([...projects, ...recentProjects(), currentState.project_dir].filter(Boolean))].sort();
  const sel = $('project-select');
  sel.innerHTML = '';
  all.forEach(p => {
    const opt = document.createElement('option');
    opt.value = p;
    opt.textContent = p.split('/').pop();
    opt.title = p;
    sel.appendChild(opt);
  });
  sel.value = currentState.project_dir;
}

async function switchProject(path) {
  if (!path || path === currentState.project_dir) return;
  if (editorDirty) {
    const ok = await confirmModal('Unsaved checklist edits', 'Switching projects discards your unsaved checklist edits.', 'Discard & switch', true);
    if (!ok) { $('project-select').value = currentState.project_dir; return; }
    editorDirty = false;
  }
  try {
    const res = await api('/api/project', { method: 'POST', body: { project_dir: path } });
    rememberProject(res.project_dir);
    location.reload();
  } catch (err) {
    showError(err, 'Switch project');
    $('project-select').value = currentState.project_dir;
  }
}

async function openProjectPathModal() {
  const p = await promptModal('Open project', 'Absolute path to a git repository', {
    value: currentState.project_dir || '', placeholder: '/home/you/projects/thing',
  });
  if (p && p.trim()) switchProject(p.trim());
}

$('project-select').addEventListener('change', e => switchProject(e.target.value));

// ==========================================
// INIT
// ==========================================
document.querySelectorAll('#tab-todo button[onclick^="requeueStuckItems"], #btn-requeue-dash').forEach(el => {
  el.dataset.idleOnly = '';
  el.dataset.idleTitle = el.title;
});

restoreSimpleControls();
if (currentState.project_dir) rememberProject(currentState.project_dir);
lastStage = (currentState.current_task || {}).stage || null;
updateUI(currentState);
initSSE();
fetchProjects();
refreshBranches();
loadTodoDetails();
