// app.js — MedRAG frontend

// ── Panel navigation ────────────────────────────────────────
function showPanel(name) {
  document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.nav-btn').forEach(b => b.classList.remove('active'));
  document.getElementById(`panel-${name}`).classList.add('active');
  document.querySelectorAll('.nav-btn').forEach(b => {
    if (b.textContent.toLowerCase().includes(name)) b.classList.add('active');
  });
}

// ── Stats ───────────────────────────────────────────────────
async function loadStats() {
  try {
    const res = await fetch('/api/stats');
    const d   = await res.json();
    if (d.error) { document.getElementById('statsBox').textContent = 'Unavailable'; return; }
    document.getElementById('statsBox').innerHTML =
      `Chunks: <span class="sv">${d.total_chunks}</span><br>` +
      `Embed: <span class="sv">MiniLM-L6</span><br>` +
      `LLM: <span class="sv">flan-t5</span><br>` +
      `Index: <span class="sv">FAISS</span>`;
  } catch {
    document.getElementById('statsBox').textContent = 'Server offline';
  }
}
loadStats();

// ── Slider ──────────────────────────────────────────────────
document.getElementById('kSlider').addEventListener('input', function () {
  document.getElementById('kLabel').textContent = this.value;
});

// ── Suggested questions ─────────────────────────────────────
function fillQ(text) {
  document.getElementById('qInput').value = text;
  document.getElementById('qInput').focus();
  autoResize(document.getElementById('qInput'));
}

// ── Auto-resize textarea ─────────────────────────────────────
function autoResize(el) {
  el.style.height = 'auto';
  el.style.height = Math.min(el.scrollHeight, 120) + 'px';
}
document.getElementById('qInput').addEventListener('input', function () { autoResize(this); });
document.getElementById('qInput').addEventListener('keydown', function (e) {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendQuestion(); }
});

// ── Send question ────────────────────────────────────────────
async function sendQuestion() {
  const input = document.getElementById('qInput');
  const question = input.value.trim();
  if (!question) return;

  input.value = '';
  autoResize(input);
  document.getElementById('sendBtn').disabled = true;

  appendMessage('user', question);
  const thinkEl = appendThinking();

  const topK = +document.getElementById('kSlider').value;

  try {
    const res = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question, top_k: topK }),
    });
    const data = await res.json();
    thinkEl.remove();

    if (data.error) {
      appendMessage('assistant', `⚠️ Error: ${data.error}`);
    } else {
      appendAnswer(data);
    }
  } catch (err) {
    thinkEl.remove();
    appendMessage('assistant', `⚠️ Could not reach the server. Make sure Flask is running on port 5000.`);
  }

  document.getElementById('sendBtn').disabled = false;
  input.focus();
}

// ── DOM helpers ──────────────────────────────────────────────
const messages = document.getElementById('messages');

function appendMessage(role, text) {
  const wrap = document.createElement('div');
  wrap.className = `message ${role}`;
  wrap.innerHTML = `
    <div class="msg-avatar">${role === 'user' ? 'You' : '⚕'}</div>
    <div class="msg-content">
      <div class="bubble">${escHtml(text)}</div>
    </div>`;
  messages.appendChild(wrap);
  scrollBottom();
  return wrap;
}

function appendAnswer({ answer, sources, disclaimer }) {
  const wrap = document.createElement('div');
  wrap.className = 'message assistant';

  let sourcesHtml = '';
  if (sources && sources.length) {
    const chips = sources.map((s, i) => `
      <div class="source-chip">
        <strong>[${i+1}]</strong> ${escHtml(s.text)}
        <span class="score">${s.score}%</span>
        <span class="src-file">${escHtml(s.source)}</span>
      </div>`).join('');
    sourcesHtml = `
      <div class="sources-wrap">
        <button class="sources-toggle" onclick="toggleSources(this)">▸ ${sources.length} source${sources.length > 1 ? 's' : ''}</button>
        <div class="sources-list">${chips}</div>
      </div>`;
  }

  wrap.innerHTML = `
    <div class="msg-avatar">⚕</div>
    <div class="msg-content">
      <div class="bubble">${escHtml(answer)}</div>
      ${sourcesHtml}
      <div class="disclaimer-msg">${escHtml(disclaimer)}</div>
    </div>`;
  messages.appendChild(wrap);
  scrollBottom();
}

function appendThinking() {
  const wrap = document.createElement('div');
  wrap.className = 'message assistant';
  wrap.innerHTML = `
    <div class="msg-avatar">⚕</div>
    <div class="msg-content">
      <div class="thinking-dots"><span></span><span></span><span></span></div>
    </div>`;
  messages.appendChild(wrap);
  scrollBottom();
  return wrap;
}

function toggleSources(btn) {
  const list = btn.nextElementSibling;
  list.classList.toggle('open');
  const count = list.querySelectorAll('.source-chip').length;
  btn.textContent = list.classList.contains('open')
    ? `▾ ${count} source${count > 1 ? 's' : ''}`
    : `▸ ${count} source${count > 1 ? 's' : ''}`;
}

function scrollBottom() { messages.scrollTop = messages.scrollHeight; }

function escHtml(t) {
  return String(t)
    .replace(/&/g, '&amp;').replace(/</g, '&lt;')
    .replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

// ── Upload ───────────────────────────────────────────────────
function handleUpload(file) {
  if (!file) return;
  const status = document.getElementById('uploadStatus');
  status.style.display = 'block';
  status.className = 'upload-status';
  status.textContent = `Uploading ${file.name}…`;

  const fd = new FormData();
  fd.append('file', file);

  fetch('/api/upload', { method: 'POST', body: fd })
    .then(r => r.json())
    .then(d => {
      if (d.error) {
        status.className = 'upload-status err';
        status.textContent = `Error: ${d.error}`;
      } else {
        status.className = 'upload-status ok';
        status.textContent = `✓ ${d.message} (${d.chunks} chunks indexed)`;
        loadStats();
      }
    })
    .catch(() => {
      status.className = 'upload-status err';
      status.textContent = 'Upload failed. Is Flask running?';
    });
}

// Drag-and-drop
const dz = document.getElementById('dropZone');
dz.addEventListener('dragover', e => { e.preventDefault(); dz.style.borderColor = '#1d6ff2'; });
dz.addEventListener('dragleave', () => { dz.style.borderColor = ''; });
dz.addEventListener('drop', e => {
  e.preventDefault();
  dz.style.borderColor = '';
  handleUpload(e.dataTransfer.files[0]);
});
