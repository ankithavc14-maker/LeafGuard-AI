(() => {
  'use strict';

  const $ = (id) => document.getElementById(id);
  let token = localStorage.getItem('leafguard_token');
  let currentUser = null;
  let lastPrediction = null;
  let initialized = false;

  function escapeHtml(value) {
    return String(value ?? '').replace(/[&<>'"]/g, c => ({
      '&': '&amp;', '<': '&lt;', '>': '&gt;', "'": '&#39;', '"': '&quot;'
    }[c]));
  }

  function authHeaders(extra = {}) {
    return token ? { ...extra, Authorization: `Bearer ${token}` } : extra;
  }

  function setText(id, value) {
    const el = $(id);
    if (el) el.textContent = value;
  }

  function showAuth() {
    $('authScreen')?.classList.remove('hidden');
    $('appShell')?.classList.add('hidden');
  }

  function showApp() {
    $('authScreen')?.classList.add('hidden');
    $('appShell')?.classList.remove('hidden');
  }

  function logout(redirect = true) {
    token = null;
    currentUser = null;
    lastPrediction = null;
    localStorage.removeItem('leafguard_token');
    if (redirect) {
      showAuth();
      const loginEmail = $('loginEmail');
      const loginPassword = $('loginPassword');
      if (loginEmail) loginEmail.focus();
      if (loginPassword) loginPassword.value = '';
    }
  }

  async function api(url, options = {}) {
    const opts = { ...options, headers: authHeaders(options.headers || {}) };
    let response;
    try {
      response = await fetch(url, opts);
    } catch (err) {
      throw new Error('Unable to reach the LeafGuard server. Make sure FastAPI is running.');
    }

    let data = {};
    try { data = await response.json(); } catch (_) {}

    if (response.status === 401) {
      logout(false);
      showAuth();
      throw new Error(data.detail || 'Please sign in again.');
    }

    if (!response.ok) {
      throw new Error(data.detail || `Request failed (${response.status})`);
    }

    return data;
  }

  function applyUser() {
    if (!currentUser) return;
    setText('welcomeName', currentUser.name);
    setText('sideName', currentUser.name);
    setText('sideEmail', currentUser.email);
    setText('sideAvatar', currentUser.name?.charAt(0)?.toUpperCase() || 'U');
  }

  function showView(id) {
    const target = $(id);
    if (!target) return false;

    document.querySelectorAll('.view').forEach(v => v.classList.remove('active-view'));
    target.classList.add('active-view');
    document.querySelectorAll('.nav').forEach(n => n.classList.toggle('active', n.dataset.view === id));

    if (id === 'history') loadHistory();
    if (id === 'guide') loadGuide();
    if (id === 'statistics') loadStats();
    if (id === 'dashboard') {
      loadStats();
      loadRecent();
    }
    return true;
  }

  function focusUpload() {
    showView('dashboard');
    window.setTimeout(() => $('dropzone')?.scrollIntoView({ behavior: 'smooth', block: 'center' }), 80);
  }

  function showError(msg) {
    $('emptyResult')?.classList.remove('hidden');
    $('result')?.classList.add('hidden');
    if ($('emptyResult')) $('emptyResult').textContent = '⚠️ ' + msg;
  }

  function setupDrop(id) {
    const zone = $(id);
    if (!zone || zone.dataset.wired === 'true') return;
    zone.dataset.wired = 'true';

    ['dragenter', 'dragover'].forEach(name => zone.addEventListener(name, e => {
      e.preventDefault();
      e.stopPropagation();
      zone.classList.add('drag');
    }));

    ['dragleave', 'drop'].forEach(name => zone.addEventListener(name, e => {
      e.preventDefault();
      e.stopPropagation();
      zone.classList.remove('drag');
    }));

    zone.addEventListener('drop', e => {
      const file = e.dataTransfer?.files?.[0];
      if (file) uploadFile(file);
    });

    zone.addEventListener('click', e => {
      const input = $('fileInput');
      if (!input) return;
      if (e.target.closest('label, input, button')) return;
      input.click();
    });
  }

  async function uploadFile(file) {
    if (!token) {
      showAuth();
      return;
    }

    focusUpload();

    if (file.size > 5 * 1024 * 1024) {
      showError('Image is larger than 5MB.');
      return;
    }
    if (!['image/jpeg', 'image/png', 'image/webp'].includes(file.type)) {
      showError('Please upload JPG, PNG or WEBP.');
      return;
    }

    $('emptyResult')?.classList.remove('hidden');
    $('result')?.classList.add('hidden');
    setText('emptyResult', 'Analyzing image...');

    const fd = new FormData();
    fd.append('file', file, file.name || 'leaf.jpg');

    try {
      const data = await api('/api/predict', { method: 'POST', body: fd });
      lastPrediction = data;
      renderResult(data);
      await Promise.allSettled([loadStats(), loadRecent()]);
    } catch (err) {
      showError(err.message || 'Prediction failed.');
    }
  }

  function renderResult(d) {
    $('emptyResult')?.classList.add('hidden');
    $('result')?.classList.remove('hidden');

    const warning = d.needs_review ? `<div class="notice warning"><strong>⚠️ Needs review</strong><br>This is a low-confidence AI result (${d.confidence}% confidence, ${d.margin}% margin). Upload a clearer leaf image and verify the result before relying on it.</div>` : '';
    const cam = d.gradcam ? `<div class="cam"><h4>AI Attention Map</h4><img src="data:image/jpeg;base64,${d.gradcam}" alt="Grad-CAM explanation"><p class="muted">Highlighted regions show where the model focused for this prediction.</p></div>` : '';
    const alternatives = (d.top3 || []).map(x => `<span class="alt-pill">${escapeHtml(x.disease)} · ${escapeHtml(x.confidence)}%</span>`).join('');
    const badgeClass = d.status === 'Healthy' ? 'healthy' : (d.needs_review ? 'review' : '');
    const resultTitle = d.needs_review ? 'Possible match' : d.disease;

    const result = $('result');
    if (!result) return;

    result.innerHTML = `<div class="result-grid">
      <img class="result-image" src="${d.image_data || d.image_url}" alt="Uploaded leaf">
      <div>
        <span class="badge ${badgeClass}">${escapeHtml(resultTitle)}</span>
        <p class="result-plant">${escapeHtml(d.plant)} · ${escapeHtml(d.status)}</p>
        <div class="confidence">${escapeHtml(d.confidence)}%</div>
        <div class="bar"><i style="width:${Math.min(Number(d.confidence) || 0, 100)}%"></i></div>
        ${warning}
        <h4>About</h4><p>${escapeHtml(d.about)}</p>
        <h4>Treatment</h4><p>${escapeHtml(d.treatment)}</p>
        <h4>Prevention</h4><p>${escapeHtml(d.prevention)}</p>
        <h4>Top alternatives</h4><div class="alternatives">${alternatives || '<span class="muted">No alternatives available.</span>'}</div>
        <div class="result-actions">
          <button type="button" class="secondary" data-action="download-report" onclick="downloadReport()">Download Report</button>
          <button type="button" class="primary" data-action="new-prediction" onclick="newPrediction()">New Prediction</button>
        </div>
      </div>
    </div>${cam}`;
  }

  function newPrediction() {
    const input = $('fileInput');
    if (input) input.value = '';
    $('emptyResult')?.classList.remove('hidden');
    $('result')?.classList.add('hidden');
    setText('emptyResult', 'Upload an image to see the AI result.');
    focusUpload();
  }

  function downloadReport() {
    if (!lastPrediction) return;
    const d = lastPrediction;
    const text = `LEAFGUARD AI REPORT\n====================\nPlant: ${d.plant}\nResult: ${d.disease}\nStatus: ${d.status}\nConfidence: ${d.confidence}%\nMargin: ${d.margin}%\n\nAbout:\n${d.about}\n\nTreatment:\n${d.treatment}\n\nPrevention:\n${d.prevention}\n\nNote: AI decision-support only. Verify uncertain results with qualified local agricultural guidance.`;
    const a = document.createElement('a');
    const url = URL.createObjectURL(new Blob([text], { type: 'text/plain' }));
    a.href = url;
    a.download = 'leafguard-report.txt';
    document.body.appendChild(a);
    a.click();
    a.remove();
    window.setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  async function loadStats() {
    try {
      const s = await api('/api/stats');
      setText('totalStat', s.total);
      setText('healthyStat', s.healthy);
      setText('diseasedStat', s.diseased);
      setText('confidenceStat', `${s.average_confidence}%`);
      setText('statTotal', s.total);
      setText('statHealthy', s.healthy);
      setText('statDiseased', s.diseased);
      setText('statReview', s.needs_review);
      setText('ringValue', `${s.average_confidence}%`);
      const ring = $('ringValue')?.parentElement;
      if (ring) ring.style.background = `conic-gradient(#2d8b46 ${Math.min(Number(s.average_confidence) || 0, 100) * 3.6}deg,#e8efe9 0deg)`;
      setText('confidenceText', s.total ? `${s.total} saved prediction${s.total === 1 ? '' : 's'} in your account.` : 'No predictions yet.');
    } catch (_) {}
  }

  async function loadRecent() {
    try {
      const history = await api('/api/history');
      const recent = $('recent');
      if (!recent) return;
      recent.innerHTML = history.slice(0, 5).map(x => `<div class="recent-row"><div><b>${escapeHtml(x.disease)}</b><small>${escapeHtml(x.plant)} · ${escapeHtml(x.status)}</small></div><span>${escapeHtml(x.confidence)}%</span></div>`).join('') || '<p class="muted">No predictions yet.</p>';
    } catch (_) {}
  }

  async function loadHistory() {
    try {
      const history = await api('/api/history');
      const container = $('historyTable');
      if (!container) return;
      container.innerHTML = history.length ? `<div class="table-wrap"><table class="table"><thead><tr><th>Image</th><th>Disease</th><th>Plant</th><th>Status</th><th>Confidence</th><th>Date</th><th></th></tr></thead><tbody>${history.map(x => `<tr><td><img class="thumb" src="/api/image/${encodeURIComponent(x.id)}" data-auth-img="true"></td><td>${escapeHtml(x.disease)}</td><td>${escapeHtml(x.plant)}</td><td><span class="status-dot">${escapeHtml(x.status)}</span></td><td>${escapeHtml(x.confidence)}%</td><td>${escapeHtml(String(x.created_at).replace('T', ' '))}</td><td><button type="button" class="delete-btn" data-action="delete-prediction" data-id="${escapeHtml(x.id)}">Delete</button></td></tr>`).join('')}</tbody></table></div>` : '<p class="muted">No predictions yet.</p>';
      await authorizeImages();
    } catch (_) {}
  }

  async function authorizeImages() {
    for (const img of document.querySelectorAll('img[data-auth-img]')) {
      try {
        const r = await fetch(img.src, { headers: authHeaders() });
        if (!r.ok) continue;
        const b = await r.blob();
        const url = URL.createObjectURL(b);
        img.onload = () => URL.revokeObjectURL(url);
        img.src = url;
      } catch (_) {}
    }
  }

  async function deletePrediction(id) {
    if (!id || !window.confirm('Delete this prediction?')) return;
    try {
      await api(`/api/history/${encodeURIComponent(id)}`, { method: 'DELETE' });
      await Promise.allSettled([loadHistory(), loadStats(), loadRecent()]);
    } catch (err) {
      window.alert(err.message || 'Unable to delete prediction.');
    }
  }

  async function loadGuide() {
    try {
      const data = await api('/api/supported');
      setText('guideSummary', `${data.classes} classes across ${Object.keys(data.plants).length} crop species.`);
      const grid = $('guideGrid');
      if (!grid) return;
      grid.innerHTML = Object.entries(data.plants).map(([plant, diseases]) => `<div class="guide-card"><div class="guide-icon">🌿</div><h3>${escapeHtml(plant)}</h3><ul>${diseases.map(x => `<li>${escapeHtml(x)}</li>`).join('')}</ul></div>`).join('');
    } catch (_) {}
  }

  async function login(email, password) {
    const response = await fetch('/api/auth/login', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ email, password })
    });
    const data = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(data.detail || 'Login failed');
    token = data.access_token;
    localStorage.setItem('leafguard_token', token);
    currentUser = data.user;
    showApp();
    applyUser();
    await loadAll();
  }

  async function register(name, email, password) {
    const response = await fetch('/api/auth/register', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name, email, password })
    });
    const data = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(data.detail || 'Registration failed');
    token = data.access_token;
    localStorage.setItem('leafguard_token', token);
    currentUser = data.user;
    showApp();
    applyUser();
    await loadAll();
  }

  async function loadAll() {
    if (!token) { showAuth(); return; }
    try {
      currentUser = await api('/api/auth/me');
      showApp();
      applyUser();
      await Promise.allSettled([loadStats(), loadRecent(), loadGuide()]);
    } catch (_) {
      showAuth();
    }
  }

  function switchAuthTab(mode) {
    const login = $('loginForm');
    const reg = $('registerForm');
    const loginTab = $('loginTab');
    const regTab = $('registerTab');
    if (mode === 'register') {
      reg?.classList.remove('hidden'); login?.classList.add('hidden');
      regTab?.classList.add('active'); loginTab?.classList.remove('active');
    } else {
      login?.classList.remove('hidden'); reg?.classList.add('hidden');
      loginTab?.classList.add('active'); regTab?.classList.remove('active');
    }
    setText('authError', '');
  }

  function useDemoAccount() {
    const email = $('loginEmail'); const password = $('loginPassword');
    if (email) email.value = 'demo@leafguard.ai';
    if (password) password.value = 'LeafGuard123!';
    submitLogin();
  }

  function submitLogin(e) {
    if (e?.preventDefault) e.preventDefault();
    setText('authError', '');
    login($('loginEmail')?.value.trim() || '', $('loginPassword')?.value || '')
      .catch(err => setText('authError', err.message || 'Login failed'));
    return false;
  }

  function submitRegister(e) {
    if (e?.preventDefault) e.preventDefault();
    setText('authError', '');
    register($('regName')?.value.trim() || '', $('regEmail')?.value.trim() || '', $('regPassword')?.value || '')
      .catch(err => setText('authError', err.message || 'Registration failed'));
    return false;
  }

  function wireEvents() {
    if (initialized) return;
    initialized = true;

    // Auth tab switching, demo login, sign-in/register submit, sign-out, the
    // sidebar toggle, new-prediction and download-report all already have a
    // working onclick/onsubmit/onchange attribute in index.html (or in the
    // markup built by renderResult() below) that calls the matching function
    // exposed on window. Those are the single source of truth for those
    // clicks, so they're intentionally not re-wired here as well.

    // Bind auth actions directly so login/register work even if inline HTML
    // handlers are blocked or a browser has stale cached markup.
    $('loginForm')?.addEventListener('submit', submitLogin);
    $('registerForm')?.addEventListener('submit', submitRegister);
    $('loginTab')?.addEventListener('click', () => switchAuthTab('login'));
    $('registerTab')?.addEventListener('click', () => switchAuthTab('register'));
    $('demoBtn')?.addEventListener('click', useDemoAccount);

    document.querySelectorAll('.nav').forEach(nav => {
      nav.addEventListener('click', () => $('sidebar')?.classList.remove('open'));
    });

    $('fileInput')?.addEventListener('click', e => e.stopPropagation());
    setupDrop('dropzone');

    document.addEventListener('click', e => {
      const action = e.target.closest?.('[data-action]')?.dataset?.action;
      if (action === 'delete-prediction') deletePrediction(e.target.closest('[data-action]')?.dataset?.id);
    });
  }

  // Expose functions used by existing inline handlers and console testing.
  window.showView = showView;
  window.newPrediction = newPrediction;
  window.downloadReport = downloadReport;
  window.deletePrediction = deletePrediction;
  window.focusUpload = focusUpload;
  window.uploadFile = uploadFile;
  window.logout = logout;
  window.switchAuthTab = switchAuthTab;
  window.useDemoAccount = useDemoAccount;
  window.submitLogin = submitLogin;
  window.submitRegister = submitRegister;

  function initialize() {
    wireEvents();
    if (token) loadAll();
    else showAuth();
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initialize, { once: true });
  } else {
    initialize();
  }
})();
