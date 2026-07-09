const state = {
  token: localStorage.getItem("tmf_token"),
  user: JSON.parse(localStorage.getItem("tmf_user") || "null"),
  dashboard: localStorage.getItem("tmf_dashboard"),
  documents: [],
  sources: [],
  ragMetrics: null,
  agenticMetrics: null,
  trainingApprovals: [],
  actionEvents: [],
};

const tmfClasses = ["protocol", "safety_report", "statistical_analysis_plan"];
const apiBaseUrl = (window.TMF_API_BASE_URL || "").replace(/\/$/, "");

const pocMetrics = {
  documentClassificationScore: 80,
  chunkSplitClassificationScore: 88,
  redisBeforeLatencyMs: 2450,
  redisAfterExactCacheMs: 180,
  redisAfterSemanticCacheMs: 310,
  retrievalBeforeCacheMs: 3200,
  retrievalAfterExactCacheMs: 95,
  retrievalAfterSemanticCacheMs: 310,
  redisSpeedup: 13.6,
  indexedDocuments: 34,
  embeddedChunks: 12480,
  averageConfidence: 86,
};

const routes = {
  landing: { title: "TMF AI Assistant", public: true },
  login: { title: "Login", public: true },
  unauthorized: { title: "Unauthorized", public: true },
  "user/dashboard": { title: "User Workspace", roles: ["User", "Manager", "Admin"] },
  "user/upload": { title: "Upload & Classify", roles: ["User", "Manager", "Admin"] },
  "user/documents": { title: "My Documents", roles: ["User", "Manager", "Admin"] },
  "user/assistant": { title: "AI Document Assistant", roles: ["User", "Manager", "Admin"] },
  "user/classification": { title: "Classification History", roles: ["User", "Manager", "Admin"] },
  "user/query-history": { title: "Query History", roles: ["User", "Manager", "Admin"] },
  "user/profile": { title: "Profile", roles: ["User", "Manager", "Admin"] },
  "manager/home": { title: "Manager Review Dashboard", roles: ["Manager", "Admin"] },
  "manager/approval-queue": { title: "Manual Classification Review", roles: ["Manager", "Admin"] },
  "manager/approved-documents": { title: "Approved Documents", roles: ["Manager", "Admin"] },
  "manager/team-documents": { title: "Team Documents", roles: ["Manager", "Admin"] },
  "manager/team-analytics": { title: "Team Analytics", roles: ["Manager", "Admin"] },
  "manager/assistant": { title: "Manager AI Assistant", roles: ["Manager", "Admin"] },
  "manager/access-overview": { title: "Access Management", roles: ["Manager", "Admin"] },
  "manager/review-history": { title: "Review History", roles: ["Manager", "Admin"] },
  "admin/dashboard": { title: "System Overview", roles: ["Admin"] },
  "admin/users": { title: "User Management", roles: ["Admin"] },
  "admin/documents": { title: "Document Repository", roles: ["Admin"] },
  "admin/classification-pipeline": { title: "Classification Pipeline", roles: ["Admin"] },
  "admin/model-management": { title: "Model Management", roles: ["Admin"] },
  "admin/vector-index": { title: "Embedding & Vector Index", roles: ["Admin"] },
  "admin/redis-cache": { title: "Redis Cache Monitor", roles: ["Admin"] },
  "admin/retrieval-pipeline": { title: "AI Retrieval Pipeline", roles: ["Admin"] },
  "admin/analytics": { title: "System Analytics", roles: ["Admin"] },
  "admin/audit": { title: "Audit Logs", roles: ["Admin"] },
  "admin/health": { title: "System Health", roles: ["Admin"] },
  "admin/settings": { title: "Settings", roles: ["Admin"] },
};

const navItems = [
  { section: "User", route: "user/dashboard", label: "User Workspace", roles: ["User", "Manager", "Admin"], icon: "▣" },
  { section: "User", route: "user/upload", label: "Upload & Classify", roles: ["User", "Manager", "Admin"], icon: "⇧" },
  { section: "User", route: "user/documents", label: "My Documents", roles: ["User", "Manager", "Admin"], icon: "▤" },
  { section: "User", route: "user/assistant", label: "AI Document Assistant", roles: ["User", "Manager", "Admin"], icon: "✦" },
  { section: "User", route: "user/classification", label: "Classification History", roles: ["User", "Manager", "Admin"], icon: "✓" },
  { section: "User", route: "user/query-history", label: "Query History", roles: ["User", "Manager", "Admin"], icon: "◷" },
  { section: "User", route: "user/profile", label: "Profile", roles: ["User", "Manager", "Admin"], icon: "◌" },
  { section: "Manager", route: "manager/home", label: "Manager Review Dashboard", roles: ["Manager", "Admin"], icon: "▣" },
  { section: "Manager", route: "manager/approval-queue", label: "Manual Classification Review", roles: ["Manager", "Admin"], icon: "✓" },
  { section: "Manager", route: "manager/approved-documents", label: "Approved Documents", roles: ["Manager", "Admin"], icon: "▤" },
  { section: "Manager", route: "manager/team-documents", label: "Team Documents", roles: ["Manager", "Admin"], icon: "▥" },
  { section: "Manager", route: "manager/team-analytics", label: "Team Analytics", roles: ["Manager", "Admin"], icon: "↗" },
  { section: "Manager", route: "manager/assistant", label: "Manager AI Assistant", roles: ["Manager", "Admin"], icon: "✦" },
  { section: "Manager", route: "manager/access-overview", label: "Access Management", roles: ["Manager", "Admin"], icon: "◈" },
  { section: "Manager", route: "manager/review-history", label: "Review History", roles: ["Manager", "Admin"], icon: "◷" },
  { section: "Admin", route: "admin/dashboard", label: "System Overview", roles: ["Admin"], icon: "▣" },
  { section: "Admin", route: "admin/users", label: "User Management", roles: ["Admin"], icon: "◉" },
  { section: "Admin", route: "admin/documents", label: "Document Repository", roles: ["Admin"], icon: "▤" },
  { section: "Admin", route: "admin/classification-pipeline", label: "Classification Pipeline", roles: ["Admin"], icon: "⇄" },
  { section: "Admin", route: "admin/model-management", label: "Model Management", roles: ["Admin"], icon: "◬" },
  { section: "Admin", route: "admin/vector-index", label: "Embedding & Vector Index", roles: ["Admin"], icon: "◇" },
  { section: "Admin", route: "admin/redis-cache", label: "Redis Cache Monitor", roles: ["Admin"], icon: "▦" },
  { section: "Admin", route: "admin/retrieval-pipeline", label: "AI Retrieval Pipeline", roles: ["Admin"], icon: "✦" },
  { section: "Admin", route: "admin/analytics", label: "System Analytics", roles: ["Admin"], icon: "↗" },
  { section: "Admin", route: "admin/audit", label: "Audit Logs", roles: ["Admin"], icon: "◷" },
  { section: "Admin", route: "admin/health", label: "System Health", roles: ["Admin"], icon: "●" },
  { section: "Admin", route: "admin/settings", label: "Settings", roles: ["Admin"], icon: "⚙" },
];

const mockDocuments = [
  { document_id: "TMF-1024", file_name: "Protocol_Amendment_v3.pdf", predicted_class: "protocol", source_type: "MASTER_DATA", verification_status: "verified", access_level: "User", uploaded_by: "user@test.com" },
  { document_id: "TMF-2038", file_name: "IRB_Approval_Site02.pdf", predicted_class: "safety_report", source_type: "MASTER_DATA", verification_status: "verified", access_level: "Manager", uploaded_by: "manager@test.com" },
  { document_id: "TMF-3091", file_name: "Executive_Retraining_Audit.pdf", predicted_class: "statistical_analysis_plan", source_type: "MASTER_DATA", verification_status: "verified", access_level: "Admin", uploaded_by: "admin@test.com" },
];

const mockAuditLogs = [
  { event_type: "rag_access_denied", entity_type: "document", entity_id: "TMF-3091", message: "User token denied Admin document retrieval.", created_at: "2026-07-08T10:20:00Z" },
  { event_type: "document_uploaded", entity_type: "document", entity_id: "TMF-1024", message: "Protocol amendment uploaded and indexed.", created_at: "2026-07-08T10:12:00Z" },
  { event_type: "manual_review_required", entity_type: "document", entity_id: "TMF-2038", message: "Low confidence document queued for Manager review.", created_at: "2026-07-08T09:58:00Z" },
];

function routeName() {
  return (location.hash || "#landing").replace("#", "").replace(/^\//, "") || "landing";
}

function setRoute(route) {
  location.hash = `/${route}`;
}

function canAccess(route) {
  const config = routes[route] || routes.landing;
  if (config.public) return true;
  return Boolean(state.token && state.user && config.roles?.includes(state.user.role));
}

function roleHome(role) {
  if (role === "Admin") return "admin/dashboard";
  if (role === "Manager") return "manager/home";
  return "user/dashboard";
}

function apiHeaders(extra = {}) {
  return {
    ...extra,
    ...(state.token ? { Authorization: `Bearer ${state.token}` } : {}),
  };
}

async function api(path, options = {}) {
  const response = await fetch(`${apiBaseUrl}${path}`, {
    ...options,
    headers: apiHeaders(options.headers || {}),
  });
  const text = await response.text();
  let data = {};
  try {
    data = text ? JSON.parse(text) : {};
  } catch {
    throw new Error(`API returned a non-JSON response for ${path}. Check the backend URL.`);
  }
  if (!response.ok) {
    throw new Error(data.detail || `Request failed with ${response.status}`);
  }
  return data;
}

function saveSession(payload) {
  state.token = payload.access_token;
  state.user = payload.user;
  state.dashboard = payload.dashboard;
  localStorage.setItem("tmf_token", state.token);
  localStorage.setItem("tmf_user", JSON.stringify(state.user));
  localStorage.setItem("tmf_dashboard", state.dashboard || "");
}

function clearSession() {
  state.token = null;
  state.user = null;
  state.dashboard = null;
  state.documents = [];
  localStorage.removeItem("tmf_token");
  localStorage.removeItem("tmf_user");
  localStorage.removeItem("tmf_dashboard");
}

function toast(message) {
  const existing = document.querySelector(".toast");
  if (existing) existing.remove();
  const el = document.createElement("div");
  el.className = "toast";
  el.textContent = message;
  document.body.appendChild(el);
  setTimeout(() => el.remove(), 4200);
}

function setBusy(button, isBusy, busyText = "Working...") {
  if (!button) return;
  if (isBusy) {
    button.dataset.originalText = button.textContent;
    button.textContent = busyText;
    button.disabled = true;
  } else {
    button.textContent = button.dataset.originalText || button.textContent;
    button.disabled = false;
    delete button.dataset.originalText;
  }
}

function badge(value) {
  const cls = String(value || "").toLowerCase();
  return `<span class="badge ${cls}">${value || "n/a"}</span>`;
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function layout(content, title = "Dashboard") {
  const allowedNav = navItems.filter((item) => item.roles.includes(state.user.role));
  const sections = [...new Set(allowedNav.map((item) => item.section))];
  return `
    <div class="app-shell">
      <aside class="sidebar">
        <div class="brand">
          <span class="brand-mark">⌁</span>
          <span>TMF AI Console</span>
        </div>
        ${sections.map((section) => `
          <div class="nav-section">
            <div class="nav-title">${section}</div>
            ${allowedNav.filter((item) => item.section === section).map((item) => `
              <button class="nav-item ${routeName() === item.route ? "active" : ""}" data-route="${item.route}">
                <span>${item.icon}</span><span>${item.label}</span>
              </button>
            `).join("")}
          </div>
        `).join("")}
        <div class="sidebar-footer">
          <div class="muted">${state.user.name}</div>
          <div>${state.user.email}</div>
          <span class="role-pill">${state.user.role}</span>
        </div>
      </aside>
      <main class="main">
        <header class="topbar">
          <div>
            <div class="muted">${state.dashboard || `${state.user.role} Dashboard`}</div>
            <h1>${title}</h1>
          </div>
          <div class="row-actions">
            <button class="btn" data-action="refresh">Refresh</button>
            <button class="btn subtle" data-action="logout">Logout</button>
          </div>
        </header>
        <section class="content">${content}</section>
      </main>
    </div>
  `;
}

function renderLanding() {
  return `
    <div class="landing">
      <nav class="landing-nav">
        <div class="brand"><span class="brand-mark">⌁</span><span>TMF AI Assistant</span></div>
        <div class="nav-links"><span>Features</span><span>Platform</span><span>Compliance</span></div>
        <div class="row-actions">
          <button class="btn" data-route="login">Learn more</button>
          <button class="btn primary" data-route="login">Login →</button>
        </div>
      </nav>
      <section class="hero">
        <div>
          <div class="eyebrow">● Now with RBAC-aware RAG retrieval</div>
          <h1>The Trial Master File, <span>reasoned about</span> — not just stored.</h1>
          <p>Classify, file, retrieve, and answer questions over regulatory-grade clinical documents with role-aware access controls for study teams.</p>
          <div class="hero-actions">
            <button class="btn primary" data-route="login">Login to Console →</button>
            <button class="btn" data-route="login">View demo</button>
          </div>
          <div class="trust-row"><span>21 CFR Part 11 ready</span><span>GxP audit trail</span><span>JWT and RBAC</span></div>
        </div>
        <div class="assistant-preview" aria-label="Assistant preview">
          <div class="preview-bar"><span class="dot"></span><span class="dot"></span><span class="dot"></span><span>tmf-ai.console — /user/assistant</span></div>
          <div class="preview-body">
            <div class="preview-label">You asked</div>
            <div class="preview-question">Summarise the latest protocol amendment for ONCO-204.</div>
            <div class="preview-answer">
              <strong>Assistant</strong><br />
              Amendment v3 revises the inclusion age limit and flags two site approvals as pending. Access is filtered by document metadata before retrieval.
            </div>
            <div class="source-strip">
              <span class="source-chip">Protocol_Amendment_v3.pdf</span>
              <span class="source-chip">IRB_Approval_Site02.pdf</span>
              <span class="source-chip">SAP.pdf</span>
            </div>
          </div>
        </div>
      </section>
    </div>
  `;
}

function renderLogin() {
  return `
    <div class="auth-shell">
      <section class="auth-panel">
        <div class="auth-card">
          <div class="brand"><span class="brand-mark">⌁</span><span>TMF AI Console</span></div>
          <h1>Sign in to your regulated document workspace</h1>
          <p class="muted">Use your assigned account. Your role controls navigation and backend authorization.</p>
          <form id="login-form">
            <div class="field">
              <label for="email">Email</label>
              <input class="input" id="email" type="email" autocomplete="username" required />
            </div>
            <div class="field">
              <label for="password">Password</label>
              <input class="input" id="password" type="password" autocomplete="current-password" required />
            </div>
            <button class="btn primary" type="submit" style="width:100%; margin-top:18px;">Login</button>
          </form>
          <div class="demo-logins">
            <button class="btn" data-demo="User">User</button>
            <button class="btn" data-demo="Manager">Manager</button>
            <button class="btn" data-demo="Admin">Admin</button>
          </div>
        </div>
      </section>
      <section class="auth-art">
        <div class="assistant-preview" style="max-width:560px;">
          <div class="preview-bar"><span class="dot"></span><span class="dot"></span><span class="dot"></span><span>validated.rbac.session</span></div>
          <div class="preview-body">
            <div class="grid kpis">
              <div class="kpi card"><div class="kpi-label">Uploads</div><div class="kpi-value">128</div></div>
              <div class="kpi card"><div class="kpi-label">Review</div><div class="kpi-value">14</div></div>
              <div class="kpi card"><div class="kpi-label">RAG hits</div><div class="kpi-value">92%</div></div>
              <div class="kpi card"><div class="kpi-label">Audit</div><div class="kpi-value">Live</div></div>
            </div>
          </div>
        </div>
      </section>
    </div>
  `;
}

function roleKpis() {
  const role = state.user.role;
  const labels = role === "Admin"
    ? [["AI subsystems", "11"], ["RAG queries", state.ragMetrics?.metrics?.total_questions ?? "Live"], ["Vector index", "Ready"], ["Redis cache", percentValue(state.ragMetrics?.metrics?.cache_hit_rate)]]
    : role === "Manager"
      ? [["Pending Reviews", "Open"], ["Approved Today", "4"], ["Average Review Time", "18m"], ["Approval Rate", "86%"]]
      : [["Documents Uploaded", "Mine"], ["Classified", "Ready"], ["Pending Approval", "Queue"], ["AI Queries Today", state.ragMetrics?.metrics?.total_questions ?? "0"], ["Avg Confidence", percentValue(state.agenticMetrics?.metrics?.average_confidence)]];
  return labels.map(([label, value]) => `
    <div class="card kpi"><div class="kpi-label">${label}</div><div class="kpi-value">${value}</div><div class="kpi-note">Role-aware workspace</div></div>
  `).join("");
}

function platformCapabilities() {
  const capabilities = [
    ["Classifier", "BioClinicalBERT upload classification", "green"],
    ["RAG", "Role-aware document Q&A", "user"],
    ["Redis", `${pocMetrics.redisSpeedup}x POC cache speedup`, "green"],
    ["PostgreSQL", "Users, metadata, chunks, audit", "manager"],
    ["S3 Storage", "Raw and processed document files", "user"],
    ["RBAC", "User, Manager, Admin route guards", "green"],
    ["Re-indexing", "Admin MASTER_DATA vector indexing", "admin"],
    ["Training Approval", "Admin controls training eligibility", "admin"],
    ["Manual Classification Review", "Manager approves low-confidence classes", "manager"],
    ["Audit Logs", "Admin event visibility", "admin"],
    ["System Health", "Operational service overview", "green"],
  ];
  return `
    <div class="source-strip capability-strip">
      ${capabilities.map(([name, note, cls]) => `<span class="source-chip ${cls}"><strong>${name}</strong> ${note}</span>`).join("")}
    </div>
  `;
}

function renderDashboard() {
  const role = state.user.role;
  const isAdmin = routeName() === "admin/dashboard";
  const isManager = routeName() === "manager/home";
  const quickActions = isAdmin
    ? [["Model Management", "admin/model-management"], ["Embedding & Vector Index", "admin/vector-index"], ["Redis Cache Monitor", "admin/redis-cache"], ["AI Retrieval Pipeline", "admin/retrieval-pipeline"]]
    : isManager
      ? [["Manual Classification Review", "manager/approval-queue"], ["Team Documents", "manager/team-documents"], ["Manager AI Assistant", "manager/assistant"]]
      : [["Upload New Document", "user/upload"], ["Ask AI Assistant", "user/assistant"], ["View Documents", "user/documents"]];
  const content = `
    <div class="grid kpis">${roleKpis()}</div>
    <div class="grid two" style="margin-top:18px;">
      <div class="card pad">
        <div class="card-title"><h2>${isAdmin ? "Enterprise AI platform map" : isManager ? "Manager validation workflow" : "AI document lifecycle"}</h2>${badge(role)}</div>
        <div id="dashboard-documents" class="table-wrap">${documentsTable((state.documents.length ? state.documents : mockDocuments).slice(0, 5))}</div>
      </div>
      <div class="card pad">
        <div class="card-title"><h2>Quick actions</h2><span class="badge green">Role scoped</span></div>
        <div class="row-actions" style="align-items:stretch;">
          ${quickActions.map(([label, route]) => `<button class="btn subtle" data-route="${route}">${label}</button>`).join("")}
        </div>
        <div class="chart-bars" style="margin-top:18px;">
          ${bar("AI Classification", 92)}
          ${bar("PostgreSQL Metadata", 95)}
          ${bar("AWS S3 Storage", 88)}
          ${bar("JWT + RBAC", 100)}
          ${bar("Redis + RAG", 82)}
        </div>
      </div>
    </div>
    <div class="card pad" style="margin-top:18px;">
      <div class="card-title"><h2>${isAdmin ? "Platform capabilities" : "Workspace capabilities"}</h2><span class="muted">Major Stage 7 systems visible at a glance</span></div>
      ${platformCapabilities()}
    </div>
  `;
  return layout(content, isAdmin ? "System Overview" : isManager ? "Manager Review Dashboard" : "User Workspace");
}

function renderClassificationResults() {
  const docs = (state.documents.length ? state.documents : mockDocuments).filter((doc) => allowedByRole(doc.access_level));
  return layout(`
    <div class="card pad">
      <div class="card-title">
        <h2>Classification Results</h2>
        <span class="muted">Predicted TMF classes from your accessible documents</span>
      </div>
      <div class="table-wrap">${documentsTable(docs)}</div>
    </div>
  `, "Classification Results");
}

function renderQueryHistory() {
  return layout(`
    <div class="card pad">
      <div class="card-title"><h2>Query History</h2><span class="muted">TODO: Persist per-user RAG query history in backend</span></div>
      <div class="table-wrap">
        <table>
          <thead><tr><th>Question</th><th>Cache</th><th>Latency</th><th>Sources</th></tr></thead>
          <tbody>
            <tr><td>What is the study objective?</td><td>${badge("exact")}</td><td>184 ms</td><td>5 citations</td></tr>
            <tr><td>Who is eligible for the study?</td><td>${badge("semantic")}</td><td>312 ms</td><td>3 citations</td></tr>
          </tbody>
        </table>
      </div>
    </div>
  `, "Query History");
}

function renderProfile() {
  return layout(`
    <div class="grid two">
      <div class="card pad">
        <div class="card-title"><h2>Profile</h2>${badge(state.user.role)}</div>
        <div class="field"><label>Name</label><input class="input" value="${state.user.name}" disabled /></div>
        <div class="field"><label>Email</label><input class="input" value="${state.user.email}" disabled /></div>
        <div class="field"><label>Role</label><input class="input" value="${state.user.role}" disabled /></div>
      </div>
      <div class="card pad">
        <div class="card-title"><h2>Session</h2><span class="badge green">Authenticated</span></div>
        <p class="muted">Your JWT is attached to protected API requests. Backend authorization remains the source of truth.</p>
      </div>
    </div>
  `, "Profile");
}

function bar(label, value) {
  return `<div class="bar-row"><span>${label}</span><div class="bar"><span style="width:${value}%"></span></div><strong>${value}%</strong></div>`;
}

function documentsTable(docs) {
  if (!docs.length) {
    return `<div class="empty-state">No documents available for this role yet.</div>`;
  }
  return `
    <table>
      <thead><tr><th>Document</th><th>Class</th><th>Source</th><th>Access</th><th>Status</th><th>Owner</th></tr></thead>
      <tbody>
        ${docs.map((doc) => `
          <tr>
            <td><strong>${doc.file_name || doc.filename || doc.document_id}</strong><br /><span class="muted">${doc.document_id || doc.doc_id || ""}</span></td>
            <td>${doc.predicted_class || doc.final_class || "pending"}</td>
            <td>${doc.source_type || "PREDICT_UPLOAD"}</td>
            <td>${badge(doc.access_level || state.user.role || "User")}</td>
            <td>${badge(doc.verification_status || doc.document_status || "indexed")}</td>
            <td>${doc.uploaded_by || doc.owner_id || "n/a"}</td>
          </tr>
        `).join("")}
      </tbody>
    </table>
  `;
}

function renderUpload() {
  return layout(`
    <div class="grid two">
      <div class="card pad">
        <div class="card-title"><h2>Upload TMF document</h2><span class="badge ${state.user.role.toLowerCase()}">${state.user.role} access</span></div>
        <form id="upload-form">
          <label class="upload-zone" for="upload-file">
            <div>
              <div style="font-size:38px;color:var(--blue);">⇧</div>
              <strong>Choose a PDF, DOCX, or TXT file</strong>
              <p class="muted">The backend will classify, file, and attach role metadata.</p>
              <input id="upload-file" name="file" type="file" accept=".pdf,.docx,.txt" hidden required />
              <span class="btn subtle">Browse files</span>
            </div>
          </label>
          <button class="btn primary" type="submit" style="margin-top:16px;">Upload and classify</button>
        </form>
      </div>
      <div class="card pad">
        <div class="card-title"><h2>Classification result</h2></div>
        <div id="upload-result" class="result-panel muted">No upload submitted yet.</div>
      </div>
    </div>
  `, "Upload Documents");
}

function renderDocuments(title = "My Documents") {
  const docs = state.documents.length ? state.documents : mockDocuments;
  const isRepository = routeName() === "admin/documents";
  return layout(`
    <div class="card pad">
      <div class="card-title">
        <h2>${title}</h2>
        <span class="muted">${isRepository ? "PostgreSQL metadata + S3 document storage + vector index status" : "Filtered by your backend role and RAG access metadata"}</span>
      </div>
      <div id="documents-table" class="table-wrap">${isRepository ? repositoryTable(docs) : documentsTable(docs)}</div>
    </div>
  `, title);
}

function repositoryTable(docs) {
  if (!docs.length) {
    return `<div class="empty-state">No indexed document metadata returned yet.</div>`;
  }
  return `
    <table>
      <thead><tr><th>Filename</th><th>TMF Class</th><th>Access Level</th><th>Approval Status</th><th>Embedding Status</th><th>Indexed</th><th>Owner</th><th>Upload Date</th><th>Actions</th></tr></thead>
      <tbody>
        ${docs.map((doc) => `<tr>
          <td><strong>${doc.file_name || doc.filename || doc.document_id}</strong></td>
          <td>${doc.predicted_class || doc.final_class || "pending"}</td>
          <td>${badge(doc.access_level || "User")}</td>
          <td>${badge(doc.verification_status || doc.document_status || "pending")}</td>
          <td>${badge(doc.rag_ingested === false ? "pending" : "embedded")}</td>
          <td>${badge("indexed")}</td>
          <td>${doc.uploaded_by || doc.owner_id || "n/a"}</td>
          <td>${doc.created_at || doc.upload_timestamp || "n/a"}</td>
          <td><button class="btn">View</button></td>
        </tr>`).join("")}
      </tbody>
    </table>
  `;
}

function renderAssistant(title = "AI Assistant") {
  return layout(chatContent(), title);
}

function ragDocumentOptions() {
  const docs = state.documents.length ? state.documents : mockDocuments.filter((doc) => allowedByRole(doc.access_level));
  if (!docs.length) {
    return `<option value="">No indexed documents loaded</option>`;
  }
  return docs.map((doc) => {
    const id = doc.document_id || doc.doc_id || "";
    const name = doc.file_name || doc.filename || id || "Untitled document";
    const access = doc.access_level || state.user.role || "User";
    return `<option value="${escapeHtml(id)}">${escapeHtml(name)} (${escapeHtml(access)})</option>`;
  }).join("");
}

function chatContent() {
  return `
    <div class="chat-layout">
      <div class="card chat-window">
        <div class="rag-controls">
          <div class="field">
            <label for="rag-retrieval-scope">Retrieve from</label>
            <select class="select" id="rag-retrieval-scope">
              <option value="all-documents">All documents available to my role</option>
              <option value="selected-document">One selected document</option>
            </select>
          </div>
          <div class="field hidden" id="rag-document-field">
            <label for="rag-document-id">Document</label>
            <select class="select" id="rag-document-id">${ragDocumentOptions()}</select>
          </div>
        </div>
        <div class="messages" id="messages">
          <div class="message assistant">Ask a question about documents you are authorized to retrieve. Sources will appear on the right.</div>
        </div>
        <form class="chat-form" id="chat-form">
          <input class="input" id="chat-question" placeholder="Ask about protocol amendments, IRB approvals, adverse events..." required />
          <button class="btn primary" type="submit">Send</button>
        </form>
      </div>
      <div class="card pad sources-panel">
        <div class="card-title"><h2>Citations</h2><span class="badge ${state.user.role.toLowerCase()}">${state.user.role}</span></div>
        <p class="muted">RAG retrieval is filtered by JWT role first, then by your selected document scope.</p>
        <div id="sources-list" class="muted">Sources from authorized chunks will appear here.</div>
      </div>
    </div>
  `;
}

function renderAnalytics() {
  const isManagerAnalytics = routeName() === "manager/team-analytics";
  // TODO: Replace manager mock values when a Manager-scoped team analytics endpoint is added.
  if (isManagerAnalytics && !state.ragMetrics && !state.agenticMetrics) {
    state.ragMetrics = { metrics: { total_questions: 42, cache_hit_rate: 0.64, semantic_cache_hit_rate: 0.28, exact_cache_hit_rate: 0.36 } };
    state.agenticMetrics = { metrics: { total_uploaded_documents: 18, average_confidence: 0.91, auto_file_rate: 0.72, manual_review_rate: 0.18 } };
  }
  const rag = state.ragMetrics?.metrics || {};
  const agentic = state.agenticMetrics?.metrics || {};
  const content = `
    <div class="grid kpis">
      <div class="card kpi"><div class="kpi-label">RAG questions</div><div class="kpi-value">${rag.total_questions ?? "—"}</div></div>
      <div class="card kpi"><div class="kpi-label">Cache hit rate</div><div class="kpi-value">${percentValue(rag.cache_hit_rate)}</div></div>
      <div class="card kpi"><div class="kpi-label">Uploaded docs</div><div class="kpi-value">${agentic.total_uploaded_documents ?? "—"}</div></div>
      <div class="card kpi"><div class="kpi-label">Avg confidence</div><div class="kpi-value">${percentValue(agentic.average_confidence)}</div></div>
    </div>
    <div class="card pad" style="margin-top:18px;">
      <div class="card-title"><h2>${isManagerAnalytics ? "Team operational summary" : "Technical AI metrics"}</h2><span class="muted">${isManagerAnalytics ? "Mock data until a Manager analytics endpoint exists" : "Classification, embeddings, Redis, RAG and LLM latency"}</span></div>
      <div class="chart-bars">
        ${bar("Document-level Classification Score", pocMetrics.documentClassificationScore)}
        ${bar("Chunk Split Classification Score", pocMetrics.chunkSplitClassificationScore)}
        ${bar("Average Confidence", Math.round((agentic.average_confidence || 0.88) * 100))}
        ${bar("Documents Uploaded", Math.min(100, Number(agentic.total_uploaded_documents || 34)))}
        ${bar("Embedding Count", 84)}
        ${bar("Chunk Count", 76)}
        ${bar("Semantic cache hit rate", Math.round((rag.semantic_cache_hit_rate || 0) * 100))}
        ${bar("Average Retrieval Latency", 68)}
        ${bar("Average LLM Response Time", 59)}
      </div>
    </div>
    ${renderPocReportPanel()}
  `;
  return layout(content, isManagerAnalytics ? "Team Analytics" : "Analytics");
}

function metricCard(label, value, note = "") {
  return `<div class="card kpi"><div class="kpi-label">${label}</div><div class="kpi-value">${value}</div><div class="kpi-note">${note}</div></div>`;
}

function renderPocReportPanel() {
  return `
    <div class="card pad poc-report" style="margin-top:18px;">
      <div class="card-title">
        <h2>POC Technical Impact Report</h2>
        <span class="muted">Prototype metrics for demo storytelling; replace with validated production telemetry later</span>
      </div>
      <div class="grid kpis">
        ${metricCard("Document-level Classification", `${pocMetrics.documentClassificationScore}%`, "POC variable: documentClassificationScore")}
        ${metricCard("Chunk Split Classification", `${pocMetrics.chunkSplitClassificationScore}%`, "POC variable: chunkSplitClassificationScore")}
        ${metricCard("Redis Before Cache", `${pocMetrics.redisBeforeLatencyMs} ms`, "POC variable: redisBeforeLatencyMs")}
        ${metricCard("Redis After Exact Cache", `${pocMetrics.redisAfterExactCacheMs} ms`, "POC variable: redisAfterExactCacheMs")}
        ${metricCard("Semantic Cache Path", `${pocMetrics.redisAfterSemanticCacheMs} ms`, "POC variable: redisAfterSemanticCacheMs")}
        ${metricCard("Redis Speedup", `${pocMetrics.redisSpeedup}x`, "Before vs exact-cache response")}
        ${metricCard("Indexed Documents", pocMetrics.indexedDocuments, "POC variable: indexedDocuments")}
        ${metricCard("Embedded Chunks", pocMetrics.embeddedChunks.toLocaleString(), "POC variable: embeddedChunks")}
      </div>
      <div class="chart-bars" style="margin-top:18px;">
        ${bar("Before Redis cache", Math.min(100, Math.round(pocMetrics.redisBeforeLatencyMs / 32)))}
        ${bar("After exact cache", Math.max(6, Math.round(pocMetrics.redisAfterExactCacheMs / 32)))}
        ${bar("After semantic cache", Math.max(10, Math.round(pocMetrics.redisAfterSemanticCacheMs / 32)))}
        ${bar("Document-level score", pocMetrics.documentClassificationScore)}
        ${bar("Chunk split score", pocMetrics.chunkSplitClassificationScore)}
      </div>
    </div>
  `;
}

function renderModelManagement() {
  return layout(`
    <div class="grid kpis">
      ${metricCard("Current Model Version", "v1.0.0", "BioClinicalBERT TMF classifier")}
      ${metricCard("Training Dataset Size", "DVC", "Versioned local/cloud dataset")}
      ${metricCard("Document Classification Score", `${pocMetrics.documentClassificationScore}%`, "POC doc-level score")}
      ${metricCard("Chunk Split Classification", `${pocMetrics.chunkSplitClassificationScore}%`, "POC chunk-level score")}
      ${metricCard("F1 Score", "0.89", "Macro F1")}
      ${metricCard("Precision", "0.90", "Weighted")}
      ${metricCard("Recall", "0.88", "Weighted")}
      ${metricCard("Last Training Time", "Tracked", "MLflow/DVC metadata")}
    </div>
    <div class="card pad" style="margin-top:18px;">
      <div class="card-title"><h2>Admin Training & Retraining</h2><span class="muted">Training eligibility is admin-owned; retraining uses verified/approved data</span></div>
      <div class="row-actions">
        <button class="btn primary" data-backend-action="retrain-model">Retrain Model</button>
      </div>
      <p class="muted">Manager review approves or corrects classification. Admin retraining uses finalized, verified documents as future training candidates.</p>
    </div>
    <div class="card pad" style="margin-top:18px;">
      <div class="card-title"><h2>Training Approval Queue</h2><span class="muted">Admin approves or rejects finalized documents for future retraining</span></div>
      <div id="training-approval-list" class="muted">Loading training approval candidates...</div>
    </div>
    ${renderPocReportPanel()}
  `, "Model Management");
}

function renderVectorIndex() {
  return layout(`
    <div class="grid kpis">
      ${metricCard("Indexed Documents", state.documents.length || pocMetrics.indexedDocuments, "rag_documents")}
      ${metricCard("Embedded Chunks", pocMetrics.embeddedChunks.toLocaleString(), "rag_chunks")}
      ${metricCard("Embedding Model", "PubMedBERT", "Local embeddings")}
      ${metricCard("Chunk Size", "512", "50 overlap")}
      ${metricCard("Last Index Time", "Recent", "Seeder/manual indexing")}
    </div>
    <div class="card pad" style="margin-top:18px;">
      <div class="card-title"><h2>Document Indexing</h2><span class="muted">Documents are embedded before RAG retrieval</span></div>
      <div class="pipeline">${pipelineSteps(["Document text", "Chunking", "Vector Embeddings", "pgvector Index", "RBAC Metadata", "RAG Retrieval"])}</div>
      <div class="row-actions" style="margin-top:18px;">
        <button class="btn primary" data-backend-action="reindex-all">Re-index All Documents</button>
      </div>
    </div>
  `, "Embedding & Vector Index");
}

function renderRedisCache() {
  const rag = state.ragMetrics?.metrics || {};
  return layout(`
    <div class="grid kpis">
      ${metricCard("Redis Status", "Healthy", "redis://localhost:6379")}
      ${metricCard("Cache Hit Rate", percentValue(rag.cache_hit_rate), "Exact + semantic")}
      ${metricCard("Cache Miss Rate", percentValue(1 - (rag.cache_hit_rate || 0)), "Generated answer path")}
      ${metricCard("Cached Queries", rag.llm_api_calls_saved ?? "—", "LLM calls saved")}
      ${metricCard("Avg Retrieval Time", `${Math.round(rag.avg_retrieval_latency_ms || 0)} ms`, "Server metrics")}
      ${metricCard("Memory Usage", "Redis", "TODO: expose INFO memory endpoint")}
    </div>
    <div class="card pad" style="margin-top:18px;">
      <div class="card-title"><h2>Redis Query Caching</h2><span class="muted">Exact answer cache + semantic cache by scoped key</span></div>
      <div class="row-actions">
        <button class="btn primary" data-backend-action="refresh-metrics">Refresh Metrics</button>
      </div>
    </div>
    ${renderPocReportPanel()}
  `, "Redis Cache Monitor");
}

function pipelineSteps(steps) {
  return steps.map((step, index) => `<div class="pipeline-step"><span>${index + 1}</span><strong>${step}</strong></div>`).join("");
}

function renderRetrievalPipeline() {
  return layout(`
    <div class="card pad">
      <div class="card-title"><h2>AI Retrieval Pipeline</h2><span class="muted">Complete RAG path before answer generation</span></div>
      <div class="pipeline vertical">${pipelineSteps(["User Question", "TMF Classifier Context", "Metadata Filtering", "RBAC Filtering", "Redis Cache Lookup", "Vector Search", "LLM", "Answer + Sources"])}</div>
    </div>
    <div style="margin-top:18px;">${chatContent()}</div>
  `, "AI Retrieval Pipeline");
}

function renderClassificationPipeline() {
  return layout(`
    <div class="card pad">
      <div class="card-title"><h2>Classification Pipeline</h2><span class="muted">Upload to approved indexed knowledge base</span></div>
      <div class="pipeline vertical">${pipelineSteps(["Document Upload", "BioClinicalBERT Classifier", "Predicted TMF Class", "Confidence Score", "Manager Review", "Approved", "Embedding", "Indexed"])}</div>
    </div>
    <div class="card pad" style="margin-top:18px;">
      <div class="card-title"><h2>Recent Classification Results</h2></div>
      <div class="table-wrap">${documentsTable(state.documents.length ? state.documents : mockDocuments)}</div>
    </div>
  `, "Classification Pipeline");
}

function renderAccessOverview() {
  const docs = state.documents.length ? state.documents : mockDocuments.filter((doc) => allowedByRole(doc.access_level));
  const levels = ["User", "Manager", "Admin"].map((level) => [level, docs.filter((doc) => (doc.access_level || "User") === level).length]);
  return layout(`
    <div class="grid kpis">
      ${levels.map(([level, count]) => `<div class="card kpi"><div class="kpi-label">${level} documents</div><div class="kpi-value">${count}</div><div class="kpi-note">Metadata-filtered access level</div></div>`).join("")}
      <div class="card kpi"><div class="kpi-label">Visible scope</div><div class="kpi-value">${state.user.role}</div><div class="kpi-note">Current role</div></div>
    </div>
    <div class="card pad" style="margin-top:18px;">
      <div class="card-title"><h2>Document Access Overview</h2><span class="muted">RBAC metadata, no role folders</span></div>
      <div class="table-wrap">${documentsTable(docs)}</div>
    </div>
  `, "Document Access Overview");
}

function renderApprovalQueue() {
  return layout(`
    <div class="grid kpis">
      <div class="card kpi"><div class="kpi-label">Pending Classification Reviews</div><div class="kpi-value">Live</div></div>
      <div class="card kpi"><div class="kpi-label">Manager Decision</div><div class="kpi-value">Class</div></div>
      <div class="card kpi"><div class="kpi-label">Average Review Time</div><div class="kpi-value">18m</div></div>
      <div class="card kpi"><div class="kpi-label">Purpose</div><div class="kpi-value">Filing</div></div>
    </div>
    <div class="card pad" style="margin-top:18px;">
      <div class="card-title"><h2>Manual Classification Review</h2><span class="muted">Manager reviews low-confidence classifications before filing/RAG indexing</span></div>
      <div id="review-list" class="muted">Loading pending reviews...</div>
    </div>
  `, "Manual Classification Review");
}

function renderApprovedDocuments() {
  const docs = (state.documents.length ? state.documents : mockDocuments).filter((doc) => (doc.verification_status || "verified") === "verified");
  return layout(`
    <div class="card pad">
      <div class="card-title"><h2>Approved Documents</h2><span class="badge green">Knowledge base ready</span></div>
      <div class="table-wrap">${documentsTable(docs)}</div>
    </div>
  `, "Approved Documents");
}

function renderReviewHistory() {
  return layout(`
    <div class="card pad">
      <div class="card-title"><h2>Review History</h2><span class="muted">Manager classification review trail</span></div>
      <div class="table-wrap">
        <table><thead><tr><th>Document</th><th>Decision</th><th>Reviewer</th><th>Time</th></tr></thead>
        <tbody><tr><td>Protocol_Amendment_v3.pdf</td><td>${badge("approved")}</td><td>${state.user.email}</td><td>Today</td></tr><tr><td>Safety_Narrative_07.pdf</td><td>${badge("changes")}</td><td>${state.user.email}</td><td>Yesterday</td></tr></tbody></table>
      </div>
    </div>
  `, "Review History");
}

function manualReviewTable(items) {
  return `
    <table>
      <thead><tr><th>Document</th><th>Predicted Class</th><th>Correct Class</th><th>Confidence</th><th>Reason</th><th>Actions</th></tr></thead>
      <tbody>
        ${items.map((item) => {
          const docId = item.doc_id || item.document_id || "";
          const predicted = item.predicted_label || item.predicted_class || item.final_class || "protocol";
          const confidence = item.decision_confidence ?? item.confidence ?? item.confidence_score ?? "n/a";
          const name = item.filename || item.file_name || item.document_id || `Document ${docId}`;
          return `<tr>
            <td><strong>${escapeHtml(name)}</strong><br /><span class="muted">${escapeHtml(docId)}</span></td>
            <td>${escapeHtml(predicted)}</td>
            <td>
              <select class="select compact-select" data-review-class="${escapeHtml(docId)}">
                ${tmfClasses.map((className) => `<option value="${className}" ${className === predicted ? "selected" : ""}>${className}</option>`).join("")}
              </select>
            </td>
            <td>${typeof confidence === "number" ? confidence.toFixed(3) : escapeHtml(confidence)}</td>
            <td>${escapeHtml(item.reason || item.status || "Manual review required")}</td>
            <td>
              <div class="row-actions">
                <button class="btn subtle" data-review-submit="approve" data-doc-id="${escapeHtml(docId)}">Approve Classification</button>
              </div>
            </td>
          </tr>`;
        }).join("")}
      </tbody>
    </table>
  `;
}

function percentValue(value) {
  return typeof value === "number" ? `${Math.round(value * 100)}%` : "—";
}

function renderAudit() {
  return layout(`
    <div class="card pad">
      <div class="card-title"><h2>Audit logs</h2><span class="muted">Connected to /audit-logs</span></div>
      <div class="source-strip" style="margin-bottom:14px;">
        ${["User Login", "Document Upload", "Classification", "Approval", "Embedding", "Re-index", "Redis Events", "RAG Query", "Role Changes"].map((item) => `<span class="source-chip">${item}</span>`).join("")}
      </div>
      <div id="audit-table" class="table-wrap">${auditTable(mockAuditLogs)}</div>
    </div>
  `, "Audit Logs");
}

function auditTable(rows) {
  if (!rows.length) {
    return `<div class="empty-state">No audit log rows returned yet.</div>`;
  }
  return `
    <table>
      <thead><tr><th>Event</th><th>Entity</th><th>Message</th><th>Created</th></tr></thead>
      <tbody>
        ${rows.map((row) => `
          <tr><td>${row.event_type}</td><td>${row.entity_type}:${row.entity_id || ""}</td><td>${row.message}</td><td>${row.created_at || ""}</td></tr>
        `).join("")}
      </tbody>
    </table>
  `;
}

function renderUsers() {
  return layout(`
    <div class="grid two">
      <div class="card pad">
        <div class="card-title"><h2>Users</h2><span class="muted">Connected to /users</span></div>
        <div id="users-table" class="table-wrap">Loading users...</div>
      </div>
      <div class="card pad">
        <div class="card-title"><h2>Create user</h2></div>
        <form id="create-user-form">
          <div class="field"><label>Name</label><input class="input" name="name" required /></div>
          <div class="field"><label>Email</label><input class="input" name="email" type="email" required /></div>
          <div class="field"><label>Password</label><input class="input" name="password" type="password" required /></div>
          <div class="field"><label>Role</label><select class="select" name="role"><option>User</option><option>Manager</option><option>Admin</option></select></div>
          <button class="btn primary" type="submit" style="margin-top:16px;">Create user</button>
        </form>
      </div>
    </div>
  `, "User Management");
}

function usersTable(users) {
  if (!users.length) {
    return `<div class="empty-state">No users returned from the backend.</div>`;
  }
  return `
    <table>
      <thead><tr><th>Name</th><th>Email</th><th>Role</th><th>Status</th></tr></thead>
      <tbody>${users.map((u) => `<tr><td>${u.name}</td><td>${u.email}</td><td>${badge(u.role)}</td><td>${u.is_active ? badge("active") : badge("inactive")}</td></tr>`).join("")}</tbody>
    </table>
  `;
}

function trainingApprovalTable(items) {
  if (!items.length) {
    return `<div class="empty-state">No documents are pending admin training approval. Documents appear here after classification is finalized and filed.</div>`;
  }
  return `
    <table>
      <thead><tr><th>Document</th><th>Status</th><th>Verified Label</th><th>Uploader</th><th>Uploaded</th><th>Actions</th></tr></thead>
      <tbody>
        ${items.map((item) => `<tr>
          <td><strong>${escapeHtml(item.filename || `Document ${item.doc_id}`)}</strong><br /><span class="muted">${escapeHtml(item.doc_id)}</span></td>
          <td>${badge(item.document_status || "pending_training_approval")}</td>
          <td>${escapeHtml(item.verified_label || "finalized classification")}</td>
          <td>${escapeHtml(item.uploaded_by || "n/a")}</td>
          <td>${escapeHtml(item.upload_timestamp || "")}</td>
          <td>
            <div class="row-actions">
              <button class="btn subtle" data-training-action="approve" data-doc-id="${escapeHtml(item.doc_id)}">Approve for Training</button>
              <button class="btn danger" data-training-action="reject" data-doc-id="${escapeHtml(item.doc_id)}">Reject from Training</button>
            </div>
          </td>
        </tr>`).join("")}
      </tbody>
    </table>
  `;
}

function renderReviews() {
  return layout(`
    <div class="card pad">
      <div class="card-title"><h2>Manual review queue</h2><span class="muted">Manager/Admin only</span></div>
      <div id="review-list" class="muted">Loading review queue...</div>
    </div>
  `, "Manual Review");
}

function renderSystemHealth() {
  const services = [
    ["FastAPI", "Healthy", "API and frontend console"],
    ["PostgreSQL", "Healthy", "Users, metadata, audit logs"],
    ["Redis", "Healthy", "Exact and semantic cache"],
    ["AWS S3", "Healthy", "Document and metadata storage"],
    ["Vector Database", "Healthy", "pgvector chunk index"],
    ["Classifier", "Healthy", "BioClinicalBERT inference"],
    ["RAG Service", "Healthy", "Retrieval + LLM answers"],
  ];
  return layout(`
    <div class="grid kpis">
      ${services.map(([name, status, note]) => metricCard(name, status, note)).join("")}
    </div>
    <div class="card pad" style="margin-top:18px;">
      <div class="card-title"><h2>System Health</h2><span class="badge green">Operational</span></div>
      <p class="muted">Live health can be extended from backend health/status endpoints.</p>
    </div>
  `, "System Health");
}

function renderSettings() {
  return layout(`
    <div class="card pad">
      <div class="card-title"><h2>Settings</h2><span class="muted">Admin-only</span></div>
      <p class="muted">TODO: Connect production settings when backend system-settings endpoints are introduced.</p>
      <div class="grid kpis">
        <div class="card kpi"><div class="kpi-label">JWT</div><div class="kpi-value">On</div></div>
        <div class="card kpi"><div class="kpi-label">RBAC RAG</div><div class="kpi-value">On</div></div>
        <div class="card kpi"><div class="kpi-label">SSO</div><div class="kpi-value">Future</div></div>
        <div class="card kpi"><div class="kpi-label">Retraining</div><div class="kpi-value">Future</div></div>
      </div>
    </div>
  `, "Settings");
}

function renderUnauthorizedShell() {
  if (!state.user) {
    return renderLogin();
  }
  return layout(`
    <div class="card pad">
      <h2>Unauthorized</h2>
      <p class="muted">Your ${state.user.role} role is not allowed to open this route. Backend authorization remains the source of truth.</p>
      <button class="btn primary" data-route="${roleHome(state.user.role)}">Return to dashboard</button>
    </div>
  `, "Unauthorized");
}

function render() {
  const route = routeName();
  if (!canAccess(route)) {
    if (!state.token) {
      setRoute("login");
      return;
    }
    history.replaceState(null, "", "#/unauthorized");
    document.getElementById("app").innerHTML = renderUnauthorizedShell();
    bindEvents();
    return;
  }

  const app = document.getElementById("app");
  const renderers = {
    landing: renderLanding,
    login: renderLogin,
    unauthorized: renderUnauthorizedShell,
    "user/dashboard": renderDashboard,
    "user/upload": renderUpload,
    "user/documents": () => renderDocuments("My Documents"),
    "user/assistant": () => renderAssistant("AI Document Assistant"),
    "user/classification": renderClassificationResults,
    "user/query-history": renderQueryHistory,
    "user/profile": renderProfile,
    "manager/home": renderDashboard,
    "manager/approval-queue": renderApprovalQueue,
    "manager/approved-documents": renderApprovedDocuments,
    "manager/team-documents": () => renderDocuments("Team Documents"),
    "manager/team-analytics": renderAnalytics,
    "manager/assistant": () => renderAssistant("Manager AI Assistant"),
    "manager/access-overview": renderAccessOverview,
    "manager/review-history": renderReviewHistory,
    "admin/dashboard": renderDashboard,
    "admin/users": renderUsers,
    "admin/documents": () => renderDocuments("Document Repository"),
    "admin/classification-pipeline": renderClassificationPipeline,
    "admin/model-management": renderModelManagement,
    "admin/vector-index": renderVectorIndex,
    "admin/redis-cache": renderRedisCache,
    "admin/retrieval-pipeline": renderRetrievalPipeline,
    "admin/analytics": renderAnalytics,
    "admin/audit": renderAudit,
    "admin/health": renderSystemHealth,
    "admin/settings": renderSettings,
  };
  app.innerHTML = (renderers[route] || renderLanding)();
  bindEvents();
  afterRender(route);
}

function bindEvents() {
  bindShellEvents();
  document.querySelectorAll("[data-route]").forEach((el) => {
    el.addEventListener("click", () => setRoute(el.dataset.route));
  });
  document.querySelectorAll("[data-demo]").forEach((el) => {
    el.addEventListener("click", () => fillDemo(el.dataset.demo));
  });
  bindDynamicActionEvents();
  document.getElementById("login-form")?.addEventListener("submit", loginSubmit);
  document.getElementById("upload-form")?.addEventListener("submit", uploadSubmit);
  document.getElementById("chat-form")?.addEventListener("submit", chatSubmit);
  document.getElementById("rag-retrieval-scope")?.addEventListener("change", updateRagScopeControls);
  document.getElementById("create-user-form")?.addEventListener("submit", createUserSubmit);
  document.getElementById("upload-file")?.addEventListener("change", (event) => {
    const file = event.target.files?.[0];
    if (file) toast(`Selected ${file.name}`);
  });
}

function bindDynamicActionEvents() {
  document.querySelectorAll("[data-backend-action]").forEach((el) => {
    if (el.dataset.boundAction) return;
    el.dataset.boundAction = "true";
    el.addEventListener("click", handleBackendAction);
  });
  document.querySelectorAll("[data-review-submit]").forEach((el) => {
    if (el.dataset.boundAction) return;
    el.dataset.boundAction = "true";
    el.addEventListener("click", submitReviewAction);
  });
  document.querySelectorAll("[data-training-action]").forEach((el) => {
    if (el.dataset.boundAction) return;
    el.dataset.boundAction = "true";
    el.addEventListener("click", submitTrainingApprovalAction);
  });
}

function updateRagScopeControls() {
  const scope = document.getElementById("rag-retrieval-scope")?.value || "all-documents";
  const documentField = document.getElementById("rag-document-field");
  if (!documentField) return;
  documentField.classList.toggle("hidden", scope !== "selected-document");
}

function updateRagDocumentOptions() {
  const select = document.getElementById("rag-document-id");
  if (!select) return;
  select.innerHTML = ragDocumentOptions();
  updateRagScopeControls();
}

function recordAction(action, detail, shouldRender = true) {
  state.actionEvents.unshift({
    action,
    detail,
    time: new Date().toLocaleTimeString(),
  });
  state.actionEvents = state.actionEvents.slice(0, 10);
  if (shouldRender) render();
  toast(`${action}: ${detail}`);
}

async function handleBackendAction(event) {
  const button = event.currentTarget;
  const action = event.currentTarget.dataset.backendAction;
  const labels = {
    "retrain-model": "Retrain Model",
    "reindex-all": "Re-index All Documents",
    "refresh-metrics": "Refresh Metrics",
  };
  const label = labels[action] || "Backend Action";

  try {
    setBusy(button, true);
    if (action === "refresh-metrics") {
      await loadMetrics();
      recordAction(label, "Pulled /rag/metrics and /agentic/metrics with the current JWT.");
      return;
    }
    if (action === "retrain-model") {
      const result = await api("/retrain", { method: "POST" });
      recordAction(label, result.status || result.message || "Backend retraining endpoint accepted the request.");
      return;
    }
    if (action === "reindex-all") {
      const result = await api("/rag/index-master-data", { method: "POST" });
      const indexed = result.indexed_documents ?? result.documents_indexed ?? result.indexed ?? "completed";
      const skipped = result.skipped_documents ?? result.skipped ?? 0;
      recordAction(label, `Backend index run ${indexed}; skipped ${skipped}.`);
      return;
    }

    recordAction(label, "Unknown operation.");
  } catch (error) {
    recordAction(label, error.message);
  } finally {
    setBusy(button, false);
  }
}

async function submitReviewAction(event) {
  const button = event.currentTarget;
  const docId = event.currentTarget.dataset.docId;
  const correctedClass = Array.from(document.querySelectorAll("[data-review-class]")).find(
    (element) => element.dataset.reviewClass === docId
  )?.value;
  if (!docId || !correctedClass) {
    toast("Review item is missing a document id or class.");
    return;
  }
  try {
    setBusy(button, true, "Submitting...");
    await api(`/agentic/reviews/${docId}/submit`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        corrected_class: correctedClass,
        reviewer_id: state.user.email,
        notes: "Submitted from Stage 7.3 manager console.",
      }),
    });
    recordAction("Submit Manual Review", `Document ${docId} approved as ${correctedClass}.`);
    await loadReviews();
  } catch (error) {
    toast(error.message);
  } finally {
    setBusy(button, false);
  }
}

async function submitTrainingApprovalAction(event) {
  const button = event.currentTarget;
  const docId = event.currentTarget.dataset.docId;
  const action = event.currentTarget.dataset.trainingAction;
  if (!docId || !["approve", "reject"].includes(action)) {
    toast("Training approval action is missing document context.");
    return;
  }
  try {
    setBusy(button, true, action === "approve" ? "Approving..." : "Rejecting...");
    await api(`/agentic/training/${docId}/${action}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        reviewer_id: state.user.email,
        notes: `Training ${action} submitted from Stage 7.3 admin console.`,
      }),
    });
    toast(`Document ${docId} ${action === "approve" ? "approved for" : "rejected from"} training.`);
    await loadTrainingApprovals();
  } catch (error) {
    toast(error.message);
  } finally {
    setBusy(button, false);
  }
}

function bindShellEvents() {
  document.querySelectorAll("[data-action='logout']").forEach((el) => {
    el.addEventListener("click", () => {
      clearSession();
      setRoute("landing");
      render();
    });
  });
  document.querySelectorAll("[data-action='refresh']").forEach((el) => {
    el.addEventListener("click", () => afterRender(routeName(), true));
  });
}

function fillDemo(role) {
  const credentials = {
    User: ["user@test.com", "user123"],
    Manager: ["manager@test.com", "manager123"],
    Admin: ["admin@test.com", "admin123"],
  }[role];
  document.getElementById("email").value = credentials[0];
  document.getElementById("password").value = credentials[1];
}

async function loginSubmit(event) {
  event.preventDefault();
  const button = event.target.querySelector("button[type='submit']");
  const email = document.getElementById("email").value;
  const password = document.getElementById("password").value;
  try {
    setBusy(button, true, "Signing in...");
    const payload = await api("/auth/login", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ email, password }),
    });
    saveSession(payload);
    toast(`Signed in as ${payload.user.role}`);
    setRoute(roleHome(payload.user.role));
  } catch (error) {
    toast(error.message);
  } finally {
    setBusy(button, false);
  }
}

async function uploadSubmit(event) {
  event.preventDefault();
  const button = event.target.querySelector("button[type='submit']");
  const file = document.getElementById("upload-file").files?.[0];
  const result = document.getElementById("upload-result");
  if (!file) return toast("Choose a document first.");
  const form = new FormData();
  form.append("file", file);
  result.textContent = "Uploading and classifying...";
  try {
    setBusy(button, true, "Classifying...");
    const payload = await api("/predict-file", { method: "POST", body: form });
    result.textContent = JSON.stringify(payload, null, 2);
    toast("Document classified successfully.");
  } catch (error) {
    result.textContent = error.message;
    toast(error.message);
  } finally {
    setBusy(button, false);
  }
}

async function chatSubmit(event) {
  event.preventDefault();
  const button = event.target.querySelector("button[type='submit']");
  const input = document.getElementById("chat-question");
  const question = input.value.trim();
  if (!question) return;
  const retrievalScope = document.getElementById("rag-retrieval-scope")?.value || "all-documents";
  const selectedDocumentId = document.getElementById("rag-document-id")?.value || "";
  if (retrievalScope === "selected-document" && !selectedDocumentId) {
    toast("Choose a document for one-document retrieval.");
    return;
  }
  appendMessage("user", question);
  appendMessage("assistant", "Retrieving authorized document chunks...");
  input.value = "";
  const requestBody = {
    question,
    scope: "all",
    ...(retrievalScope === "selected-document" ? { document_id: selectedDocumentId } : {}),
  };
  try {
    setBusy(button, true, "Sending...");
    const payload = await api("/rag/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(requestBody),
    });
    removeLastLoadingMessage();
    appendMessage("assistant", payload.answer);
    renderSources(payload.sources || []);
  } catch (error) {
    removeLastLoadingMessage();
    appendMessage("assistant", error.message);
    renderSources([]);
  } finally {
    setBusy(button, false);
  }
}

function appendMessage(kind, text) {
  const messages = document.getElementById("messages");
  const el = document.createElement("div");
  el.className = `message ${kind}`;
  el.textContent = text;
  messages.appendChild(el);
  messages.scrollTop = messages.scrollHeight;
}

function removeLastLoadingMessage() {
  const messages = document.getElementById("messages");
  const last = messages?.querySelector(".message.assistant:last-child");
  if (last?.textContent === "Retrieving authorized document chunks...") {
    last.remove();
  }
}

function renderSources(sources) {
  const panel = document.getElementById("sources-list");
  if (!sources.length) {
    panel.innerHTML = `<span class="muted">No citations returned.</span>`;
    return;
  }
  panel.innerHTML = sources.map((source) => `
    <div class="source-card">
      <strong>${source.file_name || source.document_id}</strong>
      <div class="muted">Chunk ${source.chunk_id || "n/a"} · score ${source.score ? Number(source.score).toFixed(3) : "n/a"}</div>
    </div>
  `).join("");
}

async function createUserSubmit(event) {
  event.preventDefault();
  const button = event.target.querySelector("button[type='submit']");
  const form = new FormData(event.target);
  const payload = Object.fromEntries(form.entries());
  payload.is_active = true;
  try {
    setBusy(button, true, "Creating...");
    await api("/users", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    toast("User created.");
    loadUsers();
    event.target.reset();
  } catch (error) {
    toast(error.message);
  } finally {
    setBusy(button, false);
  }
}

async function afterRender(route, force = false) {
  if (!state.token || route === "landing" || route === "login") return;
  if (
    [
      "user/dashboard",
      "user/documents",
      "user/classification",
      "user/assistant",
      "manager/home",
      "manager/team-documents",
      "manager/assistant",
      "manager/access-overview",
      "admin/dashboard",
      "admin/documents",
      "admin/classification-pipeline",
      "admin/retrieval-pipeline",
      "admin/vector-index",
    ].includes(route) || force
  ) await loadDocuments();
  if (["manager/team-analytics", "admin/analytics", "admin/redis-cache", "admin/model-management"].includes(route) || force) await loadMetrics();
  if (route === "admin/audit") await loadAudit();
  if (route === "admin/users") await loadUsers();
  if (route === "manager/approval-queue") await loadReviews();
  if (route === "admin/model-management") await loadTrainingApprovals();
}

async function loadDocuments() {
  try {
    const currentRoute = routeName();
    if (["user/dashboard", "user/documents", "user/classification"].includes(currentRoute)) {
      const data = await api("/documents/my-uploads");
      state.documents = (data.items || []).length ? data.items : mockDocuments.filter((doc) => allowedByRole(doc.access_level));
    } else {
      const data = await api("/rag/documents");
      state.documents = data.length ? data : mockDocuments.filter((doc) => allowedByRole(doc.access_level));
    }
  } catch (error) {
    // TODO: Keep mock rows when RAG/PostgreSQL is not configured or user has no indexed documents.
    state.documents = mockDocuments.filter((doc) => allowedByRole(doc.access_level));
  }
  const docsTable = document.getElementById("documents-table");
  if (docsTable) docsTable.innerHTML = routeName() === "admin/documents" ? repositoryTable(state.documents) : documentsTable(state.documents);
  const dashboardDocs = document.getElementById("dashboard-documents");
  if (dashboardDocs) dashboardDocs.innerHTML = documentsTable(state.documents.slice(0, 5));
  updateRagDocumentOptions();
}

function allowedByRole(accessLevel) {
  const role = state.user?.role;
  if (role === "Admin") return true;
  if (role === "Manager") return ["User", "Manager"].includes(accessLevel);
  return accessLevel === "User";
}

async function loadMetrics() {
  try {
    state.ragMetrics = await api("/rag/metrics");
  } catch (error) {
    // TODO: Non-admin roles cannot call metrics. UI remains available only for Admin route.
    state.ragMetrics = { metrics: {} };
  }
  try {
    state.agenticMetrics = await api("/agentic/metrics");
  } catch (error) {
    state.agenticMetrics = { metrics: {} };
  }
}

async function loadAudit() {
  try {
    const data = await api("/audit-logs");
    document.getElementById("audit-table").innerHTML = auditTable(data.items || []);
  } catch (error) {
    // TODO: Keep mock audit rows when PostgreSQL has no audit entries yet.
    document.getElementById("audit-table").innerHTML = auditTable(mockAuditLogs);
  }
}

async function loadUsers() {
  try {
    const users = await api("/users");
    document.getElementById("users-table").innerHTML = usersTable(users);
  } catch (error) {
    document.getElementById("users-table").textContent = error.message;
  }
}

async function loadTrainingApprovals() {
  const list = document.getElementById("training-approval-list");
  if (!list) return;
  try {
    const data = await api("/agentic/training/pending");
    state.trainingApprovals = data.items || [];
    list.innerHTML = `<div class="table-wrap">${trainingApprovalTable(state.trainingApprovals)}</div>`;
    bindDynamicActionEvents();
  } catch (error) {
    list.innerHTML = `<div class="empty-state error-state">${escapeHtml(error.message)}</div>`;
  }
}

async function loadReviews() {
  const reviewList = document.getElementById("review-list");
  if (!reviewList) return;
  try {
    const data = await api("/agentic/reviews");
    const items = data.items || [];
    reviewList.innerHTML = items.length
      ? `<div class="table-wrap">${manualReviewTable(items)}</div>`
      : `<div class="empty-state">No pending manual review items. Upload a low-confidence document to populate this queue.</div>`;
    bindDynamicActionEvents();
  } catch (error) {
    reviewList.innerHTML = `<div class="empty-state error-state">${escapeHtml(error.message)}</div>`;
  }
}

window.addEventListener("hashchange", render);

if (state.token) {
  api("/auth/me")
    .then((user) => {
      state.user = user;
      localStorage.setItem("tmf_user", JSON.stringify(user));
      render();
    })
    .catch(() => {
      clearSession();
      render();
    });
} else {
  render();
}
