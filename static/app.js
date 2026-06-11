/* ── State ──────────────────────────────────────────────────────────────── */
const STATE = {
  groqApiKey: '',
  model: 'llama-3.3-70b-versatile',
  retrievalMode: 'tfidf',
  embeddingApiKey: '',
  topK: 4,
  documents: [],
  chatHistory: [],
  isLoading: false,
};

/* ── DOM Refs ───────────────────────────────────────────────────────────── */
const $ = id => document.getElementById(id);

const groqApiKeyEl     = $('groqApiKey');
const modelSelectEl    = $('modelSelect');
const retrievalModeEl  = $('retrievalMode');
const embeddingKeyEl   = $('embeddingApiKey');
const embeddingKeyGrp  = $('embeddingKeyGroup');
const embeddingKeyLbl  = $('embeddingKeyLabel');
const topKSliderEl     = $('topKSlider');
const topKValueEl      = $('topKValue');
const validateBtn      = $('validateBtn');
const validateStatus   = $('validateStatus');

const uploadZone       = $('uploadZone');
const fileInput        = $('fileInput');
const browseBtn        = $('browseBtn');
const uploadProgress   = $('uploadProgress');
const progressFill     = $('progressFill');
const progressLabel    = $('progressLabel');
const documentsList    = $('documentsList');
const emptyDocs        = $('emptyDocs');

const chatContainer    = $('chatContainer');
const messagesWrapper  = $('messagesWrapper');
const welcomeScreen    = $('welcomeScreen');
const chatInput        = $('chatInput');
const sendBtn          = $('sendBtn');
const charCount        = $('charCount');
const clearChatBtn     = $('clearChatBtn');

const toggleSidebar    = $('toggleSidebar');
const mobileSidebarToggle = $('mobileSidebarToggle');
const sidebar          = document.querySelector('.sidebar');

const headerBadgeText  = $('headerBadgeText');
const headerModelText  = $('headerModelText');
const badgeDot         = document.querySelector('.badge-dot');
const toggleKeyBtn     = $('toggleKeyVisibility');
const eyeIcon          = $('eyeIcon');

/* ── Local Storage ──────────────────────────────────────────────────────── */
function saveSettings() {
  sessionStorage.setItem('rag_groq_key',      STATE.groqApiKey);
  sessionStorage.setItem('rag_model',         STATE.model);
  sessionStorage.setItem('rag_retrieval',     STATE.retrievalMode);
  sessionStorage.setItem('rag_embed_key',     STATE.embeddingApiKey);
  sessionStorage.setItem('rag_topk',          STATE.topK);
}
function loadSettings() {
  STATE.groqApiKey      = sessionStorage.getItem('rag_groq_key')  || '';
  STATE.model           = sessionStorage.getItem('rag_model')     || 'llama-3.3-70b-versatile';
  STATE.retrievalMode   = sessionStorage.getItem('rag_retrieval') || 'tfidf';
  STATE.embeddingApiKey = sessionStorage.getItem('rag_embed_key') || '';
  STATE.topK            = parseInt(sessionStorage.getItem('rag_topk') || '4');

  groqApiKeyEl.value    = STATE.groqApiKey;
  modelSelectEl.value   = STATE.model;
  retrievalModeEl.value = STATE.retrievalMode;
  embeddingKeyEl.value  = STATE.embeddingApiKey;
  topKSliderEl.value    = STATE.topK;
  topKValueEl.textContent = STATE.topK;
  updateSliderBackground(topKSliderEl);
  updateHeaderModel();
  updateRetrievalUI();
}

/* ── Sidebar ────────────────────────────────────────────────────────────── */
toggleSidebar.addEventListener('click', () => sidebar.classList.toggle('collapsed'));
mobileSidebarToggle.addEventListener('click', () => sidebar.classList.toggle('collapsed'));

/* ── API Key Visibility Toggle ──────────────────────────────────────────── */
toggleKeyBtn.addEventListener('click', () => {
  const isPassword = groqApiKeyEl.type === 'password';
  groqApiKeyEl.type = isPassword ? 'text' : 'password';
  eyeIcon.innerHTML = isPassword
    ? '<path d="M17.94 17.94A10.07 10.07 0 0 1 12 20c-7 0-11-8-11-8a18.45 18.45 0 0 1 5.06-5.94"/><path d="M9.9 4.24A9.12 9.12 0 0 1 12 4c7 0 11 8 11 8a18.5 18.5 0 0 1-2.16 3.19"/><line x1="1" y1="1" x2="23" y2="23"/>'
    : '<path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/><circle cx="12" cy="12" r="3"/>';
});

/* ── Retrieval Mode UI ──────────────────────────────────────────────────── */
function updateRetrievalUI() {
  const mode = retrievalModeEl.value;
  if (mode === 'tfidf') {
    embeddingKeyGrp.style.display = 'none';
  } else if (mode === 'huggingface') {
    embeddingKeyGrp.style.display = 'flex';
    embeddingKeyLbl.textContent = 'HuggingFace API Key (optional)';
    embeddingKeyEl.placeholder = 'hf_... (optional)';
  } else if (mode === 'openai') {
    embeddingKeyGrp.style.display = 'flex';
    embeddingKeyLbl.textContent = 'OpenAI API Key';
    embeddingKeyEl.placeholder = 'sk-...';
  }
}
retrievalModeEl.addEventListener('change', updateRetrievalUI);

/* ── Slider ─────────────────────────────────────────────────────────────── */
function updateSliderBackground(el) {
  const min = parseFloat(el.min);
  const max = parseFloat(el.max);
  const val = parseFloat(el.value);
  const pct = ((val - min) / (max - min)) * 100;
  el.style.setProperty('--val', `${pct}%`);
}
topKSliderEl.addEventListener('input', e => {
  topKValueEl.textContent = e.target.value;
  updateSliderBackground(e.target);
});

/* ── Header update ──────────────────────────────────────────────────────── */
function updateHeaderModel() {
  const modelNames = {
    'llama-3.3-70b-versatile': 'Llama 3.3 70B',
    'llama-3.1-8b-instant':    'Llama 3.1 8B',
    'llama3-70b-8192':         'Llama 3 70B',
    'llama3-8b-8192':          'Llama 3 8B',
    'mixtral-8x7b-32768':      'Mixtral 8x7B',
    'gemma2-9b-it':            'Gemma 2 9B',
  };
  headerModelText.textContent = STATE.groqApiKey
    ? (modelNames[STATE.model] || STATE.model)
    : 'No API key';
}

/* ── Settings Validate & Save ───────────────────────────────────────────── */
validateBtn.addEventListener('click', async () => {
  const key = groqApiKeyEl.value.trim();
  if (!key) {
    showStatus('error', 'Please enter a Groq API key.');
    return;
  }
  STATE.groqApiKey      = key;
  STATE.model           = modelSelectEl.value;
  STATE.retrievalMode   = retrievalModeEl.value;
  STATE.embeddingApiKey = embeddingKeyEl.value.trim();
  STATE.topK            = parseInt(topKSliderEl.value);
  saveSettings();

  validateBtn.disabled = true;
  validateBtn.textContent = 'Validating…';
  validateStatus.className = 'status-msg';
  validateStatus.style.display = 'none';

  try {
    const res = await fetch('/validate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ groq_api_key: key, model: STATE.model }),
    });
    const data = await res.json();
    if (data.valid) {
      showStatus('success', `✓ Connected! Model: ${data.model || STATE.model}`);
      updateHeaderBadge(true);
    } else {
      showStatus('error', `✗ ${data.error || 'Invalid API key.'}`);
      updateHeaderBadge(false);
    }
  } catch (err) {
    showStatus('error', `Connection error: ${err.message}`);
  }

  updateHeaderModel();
  validateBtn.disabled = false;
  validateBtn.innerHTML = `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg> Validate & Save`;
});

function showStatus(type, msg) {
  validateStatus.className = `status-msg ${type}`;
  validateStatus.textContent = msg;
}
function updateHeaderBadge(valid) {
  if (valid) {
    badgeDot.classList.remove('inactive');
    headerBadgeText.textContent = 'Connected';
  } else {
    badgeDot.classList.add('inactive');
    headerBadgeText.textContent = 'Not connected';
  }
}

/* ── File Upload ────────────────────────────────────────────────────────── */
browseBtn.addEventListener('click', e => { e.stopPropagation(); fileInput.click(); });
uploadZone.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', e => handleFiles(e.target.files));

uploadZone.addEventListener('dragover', e => { e.preventDefault(); uploadZone.classList.add('drag-over'); });
uploadZone.addEventListener('dragleave', () => uploadZone.classList.remove('drag-over'));
uploadZone.addEventListener('drop', e => {
  e.preventDefault();
  uploadZone.classList.remove('drag-over');
  handleFiles(e.dataTransfer.files);
});

async function handleFiles(fileList) {
  if (!fileList || fileList.length === 0) return;

  for (const file of fileList) {
    await uploadFile(file);
  }
  fileInput.value = '';
}

const MAX_FILE_BYTES = 10 * 1024 * 1024; // 10 MB — matches server limit

async function uploadFile(file) {
  const allowed = ['.txt', '.md', '.pdf'];
  const ext = '.' + file.name.split('.').pop().toLowerCase();
  if (!allowed.includes(ext)) {
    showUploadError(`${file.name}: Unsupported file type.`);
    return;
  }

  // Client-side size guard — avoids wasting bandwidth before server rejects it
  if (file.size > MAX_FILE_BYTES) {
    showUploadError(`${file.name}: File too large (${(file.size / 1024 / 1024).toFixed(1)} MB). Maximum is 10 MB.`);
    return;
  }

  // Show progress
  uploadProgress.style.display = 'block';
  progressFill.style.width = '0%';
  progressLabel.textContent = `Uploading ${file.name}…`;

  // Animate progress bar
  let pct = 0;
  const interval = setInterval(() => {
    pct = Math.min(pct + Math.random() * 15, 85);
    progressFill.style.width = `${pct}%`;
  }, 200);

  const formData = new FormData();
  formData.append('file', file);
  formData.append('retrieval_mode', STATE.retrievalMode);
  formData.append('embedding_api_key', STATE.embeddingApiKey || '');

  try {
    const res = await fetch('/upload', { method: 'POST', body: formData });
    clearInterval(interval);

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || 'Upload failed.');
    }

    progressFill.style.width = '100%';
    progressLabel.textContent = 'Done!';
    setTimeout(() => { uploadProgress.style.display = 'none'; }, 1200);

    await loadDocuments();
  } catch (err) {
    clearInterval(interval);
    uploadProgress.style.display = 'none';
    showUploadError(err.message);
  }
}

function showUploadError(msg) {
  const toast = document.createElement('div');
  toast.className = 'error-bubble';
  toast.style.cssText = 'position:fixed;top:16px;right:16px;z-index:999;max-width:300px;animation:fadeInUp 0.3s ease-out;';
  toast.textContent = `⚠ ${msg}`;
  document.body.appendChild(toast);
  setTimeout(() => toast.remove(), 4000);
}

/* ── Documents List ─────────────────────────────────────────────────────── */
async function loadDocuments() {
  try {
    const res = await fetch('/documents');
    const data = await res.json();
    STATE.documents = data.documents || [];
    renderDocuments();
  } catch (err) {
    console.error('Failed to load documents:', err);
  }
}

function renderDocuments() {
  // Clear existing doc items (keep emptyDocs)
  const existing = documentsList.querySelectorAll('.doc-item');
  existing.forEach(el => el.remove());

  if (STATE.documents.length === 0) {
    emptyDocs.style.display = 'flex';
    return;
  }
  emptyDocs.style.display = 'none';

  STATE.documents.forEach(doc => {
    const ext = doc.filename.split('.').pop().toLowerCase();
    const iconClass = ['pdf','txt','md'].includes(ext) ? ext : 'txt';
    const extLabel = ext.toUpperCase();
    const chunks = doc.chunk_count;
    const chars  = formatNumber(doc.char_count);

    const item = document.createElement('div');
    item.className = 'doc-item';
    item.dataset.id = doc.id;
    item.innerHTML = `
      <div class="doc-icon ${iconClass}">${extLabel}</div>
      <div class="doc-meta">
        <div class="doc-name" title="${escHtml(doc.filename)}">${escHtml(doc.filename)}</div>
        <div class="doc-info">${chunks} chunk${chunks !== 1 ? 's' : ''} · ${chars} chars</div>
      </div>
      <button class="doc-delete" title="Remove document" data-id="${doc.id}">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
          <polyline points="3 6 5 6 21 6"/><path d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6"/>
          <path d="M10 11v6M14 11v6"/><path d="M9 6V4a1 1 0 0 1 1-1h4a1 1 0 0 1 1 1v2"/>
        </svg>
      </button>`;

    item.querySelector('.doc-delete').addEventListener('click', async e => {
      e.stopPropagation();
      await deleteDocument(doc.id);
    });
    documentsList.appendChild(item);
  });
}

async function deleteDocument(docId) {
  try {
    const res = await fetch(`/documents/${docId}`, { method: 'DELETE' });
    if (res.ok) await loadDocuments();
  } catch (err) {
    console.error('Delete error:', err);
  }
}

/* ── Chat ───────────────────────────────────────────────────────────────── */
chatInput.addEventListener('input', () => {
  // Auto-resize textarea
  chatInput.style.height = 'auto';
  chatInput.style.height = Math.min(chatInput.scrollHeight, 150) + 'px';

  // Char count
  const len = chatInput.value.length;
  charCount.textContent = `${len}/4000`;

  // Enable/disable send
  sendBtn.disabled = len === 0 || STATE.isLoading;
});

chatInput.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    if (!sendBtn.disabled) handleSend();
  }
});

sendBtn.addEventListener('click', handleSend);

clearChatBtn.addEventListener('click', () => {
  STATE.chatHistory = [];
  messagesWrapper.innerHTML = '';
  welcomeScreen.style.display = '';
  sendBtn.disabled = true;
  chatInput.value = '';
  chatInput.style.height = 'auto';
  charCount.textContent = '0/4000';
});

async function handleSend() {
  const query = chatInput.value.trim();
  if (!query || STATE.isLoading) return;

  if (!STATE.groqApiKey) {
    showToast('⚠ Please set your Groq API key in the settings panel first.', 'error');
    return;
  }

  // Hide welcome screen
  welcomeScreen.style.display = 'none';

  // Append user message
  appendMessage('user', query);
  chatInput.value = '';
  chatInput.style.height = 'auto';
  charCount.textContent = '0/4000';
  sendBtn.disabled = true;

  // Typing indicator
  STATE.isLoading = true;
  const typingEl = appendTypingIndicator();

  try {
    const res = await fetch('/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        query,
        groq_api_key:     STATE.groqApiKey,
        model:            STATE.model,
        retrieval_mode:   STATE.retrievalMode,
        embedding_api_key: STATE.embeddingApiKey || null,
        top_k:            STATE.topK,
        temperature:      0.3,
      }),
    });

    typingEl.remove();

    if (!res.ok) {
      const err = await res.json();
      appendBotMessage(null, null, err.detail || 'Something went wrong.');
    } else {
      const data = await res.json();
      appendBotMessage(data.answer, data.sources, null);
    }
  } catch (err) {
    typingEl.remove();
    appendBotMessage(null, null, `Network error: ${err.message}`);
  }

  STATE.isLoading = false;
  sendBtn.disabled = chatInput.value.trim().length === 0;
  scrollToBottom();
}

/* ── Message Rendering ──────────────────────────────────────────────────── */
function appendMessage(role, text) {
  const now = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  const div = document.createElement('div');
  div.className = `message ${role}`;

  if (role === 'user') {
    div.innerHTML = `
      <div class="avatar avatar-user">U</div>
      <div class="message-body">
        <div class="message-bubble">${escHtml(text).replace(/\n/g, '<br>')}</div>
        <div class="message-time">${now}</div>
      </div>`;
  }

  messagesWrapper.appendChild(div);
  scrollToBottom();
  return div;
}

function appendBotMessage(answer, sources, errorMsg) {
  const now = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  const div = document.createElement('div');
  div.className = 'message bot';

  let bodyHtml = '';

  if (errorMsg) {
    bodyHtml = `<div class="error-bubble">⚠ ${escHtml(errorMsg)}</div>`;
  } else {
    const formatted = formatMarkdown(answer || '');
    bodyHtml = `<div class="message-bubble">${formatted}</div>`;

    // Sources
    if (sources && sources.length > 0) {
      const srcItems = sources.map((s, i) => {
        const scorePercent = Math.round((s.score || 0) * 100);
        const scoreLabel = s.score > 0 ? `${scorePercent}%` : 'TF-IDF';
        return `
          <div class="source-card" data-idx="${i}">
            <div class="source-header">
              <span class="source-filename" title="${escHtml(s.filename)}">${escHtml(s.filename)}</span>
              <span class="source-score">${scoreLabel}</span>
            </div>
            <div class="source-chunk-info">Chunk ${s.chunk_idx + 1}</div>
            <div class="source-text">${escHtml(s.text)}</div>
          </div>`;
      }).join('');

      const sourceId = `src-${Date.now()}`;
      bodyHtml += `
        <button class="sources-toggle" data-target="${sourceId}">
          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="6 9 12 15 18 9"/></svg>
          View sources <span class="source-count">${sources.length}</span>
        </button>
        <div class="sources-list" id="${sourceId}" style="display:none;">${srcItems}</div>`;
    }
  }

  div.innerHTML = `
    <div class="avatar avatar-bot">
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <circle cx="12" cy="12" r="10"/><path d="M8 14s1.5 2 4 2 4-2 4-2"/><line x1="9" y1="9" x2="9.01" y2="9"/><line x1="15" y1="9" x2="15.01" y2="9"/>
      </svg>
    </div>
    <div class="message-body">
      ${bodyHtml}
      ${!errorMsg ? `<div class="message-time">${now} · ${STATE.model.split('-').slice(0,2).join('-')}</div>` : ''}
    </div>`;

  messagesWrapper.appendChild(div);

  // Sources toggle
  div.querySelectorAll('.sources-toggle').forEach(btn => {
    btn.addEventListener('click', () => {
      const target = document.getElementById(btn.dataset.target);
      const isOpen = target.style.display !== 'none';
      target.style.display = isOpen ? 'none' : 'block';
      btn.classList.toggle('open', !isOpen);
    });
  });

  // Source card expand
  div.querySelectorAll('.source-card').forEach(card => {
    card.addEventListener('click', () => card.classList.toggle('expanded'));
  });

  scrollToBottom();
  return div;
}

function appendTypingIndicator() {
  const div = document.createElement('div');
  div.className = 'message bot';
  div.innerHTML = `
    <div class="avatar avatar-bot">
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <circle cx="12" cy="12" r="10"/><path d="M8 14s1.5 2 4 2 4-2 4-2"/><line x1="9" y1="9" x2="9.01" y2="9"/><line x1="15" y1="9" x2="15.01" y2="9"/>
      </svg>
    </div>
    <div class="message-body">
      <div class="typing-indicator">
        <div class="typing-dot"></div>
        <div class="typing-dot"></div>
        <div class="typing-dot"></div>
      </div>
    </div>`;
  messagesWrapper.appendChild(div);
  scrollToBottom();
  return div;
}

/* ── Utilities ──────────────────────────────────────────────────────────── */
function scrollToBottom() {
  requestAnimationFrame(() => {
    chatContainer.scrollTop = chatContainer.scrollHeight;
  });
}

function escHtml(str) {
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

function formatNumber(n) {
  return n >= 1000 ? (n / 1000).toFixed(1) + 'k' : String(n);
}

function formatMarkdown(text) {
  // Very lightweight markdown formatting for common cases
  let html = escHtml(text);
  // Bold
  html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
  // Italic
  html = html.replace(/\*(.+?)\*/g, '<em>$1</em>');
  // Inline code
  html = html.replace(/`([^`]+)`/g, '<code>$1</code>');
  // Headers
  html = html.replace(/^### (.+)$/gm, '<h4 style="margin:8px 0 4px;font-size:0.9em;color:var(--accent-text)">$1</h4>');
  html = html.replace(/^## (.+)$/gm,  '<h3 style="margin:10px 0 5px;font-size:1em;color:var(--text-primary)">$1</h3>');
  html = html.replace(/^# (.+)$/gm,   '<h2 style="margin:12px 0 6px;font-size:1.1em;color:var(--text-primary)">$1</h2>');
  // Bullet lists
  html = html.replace(/^[*-] (.+)$/gm, '<li style="margin:2px 0;padding-left:4px">$1</li>');
  html = html.replace(/(<li.*<\/li>)/s, '<ul style="padding-left:18px;margin:6px 0">$1</ul>');
  // Numbered lists
  html = html.replace(/^\d+\. (.+)$/gm, '<li style="margin:2px 0;padding-left:4px">$1</li>');
  // Double newlines → paragraphs
  html = html.replace(/\n\n+/g, '</p><p>');
  // Single newlines
  html = html.replace(/\n/g, '<br>');
  return `<p>${html}</p>`;
}

function showToast(msg, type = 'error') {
  const toast = document.createElement('div');
  toast.className = type === 'error' ? 'error-bubble' : 'status-msg success';
  toast.style.cssText = 'position:fixed;top:16px;right:16px;z-index:9999;max-width:320px;animation:fadeInUp 0.3s ease-out;';
  toast.textContent = msg;
  document.body.appendChild(toast);
  setTimeout(() => toast.remove(), 5000);
}

/* ── HF Spaces Banner ───────────────────────────────────────────────────── */
async function maybeShowHFBanner() {
  try {
    const res = await fetch('/api/info');
    if (!res.ok) return;
    const info = await res.json();
    if (!info.is_huggingface) return;

    const banner = document.createElement('div');
    banner.id = 'hfBanner';
    banner.style.cssText = [
      'position:fixed', 'bottom:0', 'left:0', 'right:0', 'z-index:9000',
      'background:linear-gradient(90deg,#ff6f00,#ff9800)',
      'color:#fff', 'font-size:0.78rem', 'padding:8px 16px',
      'display:flex', 'align-items:center', 'gap:10px',
      'font-family:Inter,sans-serif', 'box-shadow:0 -2px 12px rgba(0,0,0,0.3)',
    ].join(';');
    banner.innerHTML = `
      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>
      <span>🤗 Running on <strong>Hugging Face Spaces</strong> — uploaded documents are stored in memory and will reset when the Space restarts.</span>
      <button onclick="this.parentElement.remove()" style="margin-left:auto;background:rgba(255,255,255,0.25);border:none;color:#fff;border-radius:4px;padding:2px 8px;cursor:pointer;font-size:0.8rem;">✕ Dismiss</button>
    `;
    document.body.appendChild(banner);
  } catch (_) {
    // /api/info unavailable — silently skip
  }
}

/* ── Init ───────────────────────────────────────────────────────────────── */
async function init() {
  loadSettings();
  await loadDocuments();
  await maybeShowHFBanner();

  // Update badge based on key presence
  if (STATE.groqApiKey) {
    updateHeaderBadge(true);
  } else {
    updateHeaderBadge(false);
    headerBadgeText.textContent = 'Set API key';
  }
}

init();
