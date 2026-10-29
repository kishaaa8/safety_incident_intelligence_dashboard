/**
 * Greenko ESG & Safety Incident Intelligence Platform
 * Frontend Controller & API Integration Engine
 */

let allIncidents = [];
let debounceTimer = null;

const PRESETS = [
  "Contractor slipped near wet walkway at solar block, minor injury, housekeeping issue",
  "Oil seepage observed near wind gearbox maintenance bay, contained with spill kit",
  "High voltage arc flash warning alarm triggered in 33kV switchyard due to moisture ingress",
  "Battery energy storage system (BESS) container temperature alarm tripped at 42C due to HVAC failure",
  "Excavator damaged buried 11kV auxiliary power cable due to missing route marker stakes",
  "Repeated waste segregation deviation at contractor storage area",
  "Worker observed working at height (65m nacelle) without dual lanyard clip attachment"
];

document.addEventListener("DOMContentLoaded", () => {
  // Initial load
  loadPreset(0);
  runTriage();
  fetchIncidents();
  fetchAnalytics();
  fetchEvaluation();
  searchRAG();
});

// Tab Switcher
function switchTab(tabId, btn) {
  document.querySelectorAll(".tab-pane").forEach(pane => pane.classList.remove("active"));
  document.querySelectorAll(".tab-link").forEach(b => b.classList.remove("active"));
  
  const target = document.getElementById("tab-" + tabId);
  if (target) {
    target.classList.add("active");
  }
  if (btn) {
    btn.classList.add("active");
  }

  // Refresh data on specific tabs
  if (tabId === "incidents") fetchIncidents();
  if (tabId === "analytics") fetchAnalytics();
  if (tabId === "evaluation") fetchEvaluation();
}

// Preset Loader
function loadPreset(idx) {
  const narrative = PRESETS[idx] || PRESETS[0];
  const inputEl = document.getElementById("triage-input");
  if (inputEl) {
    inputEl.value = narrative;
  }
}

// Live AI Triage Action
async function runTriage() {
  const inputEl = document.getElementById("triage-input");
  const narrative = (inputEl ? inputEl.value : "").trim();
  if (!narrative) {
    alert("Please provide an incident narrative to triage.");
    return;
  }

  const outputBody = document.getElementById("triage-output-body");
  const tsEl = document.getElementById("triage-timestamp");
  if (tsEl) tsEl.textContent = "Analyzing via NLP & RAG...";

  try {
    const res = await fetch("/api/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: narrative })
    });

    if (!res.ok) {
      throw new Error(`API error: ${res.statusText}`);
    }

    const data = await res.json();
    renderTriageResults(data);
    if (tsEl) tsEl.textContent = `Triaged in ${new Date().toLocaleTimeString()}`;
  } catch (err) {
    console.error(err);
    if (outputBody) {
      outputBody.innerHTML = `
        <div style="color:#f43f5e; padding:20px; text-align:center;">
          <p><strong>Error triaging incident narrative:</strong></p>
          <p style="font-size:0.85rem; margin-top:6px;">${err.message}</p>
        </div>
      `;
    }
  }
}

function renderTriageResults(data) {
  const pred = data.prediction;
  const reasoning = data.reasoning;
  const similar = data.similar_incidents || [];
  const domain = pred.domain_features || {};

  const catClass = `badge-${pred.category}`;
  const sevClass = `sev-${pred.severity}`;

  // Build similar incidents list
  let precedentsHtml = "";
  if (similar.length > 0) {
    precedentsHtml = similar.map(s => `
      <div class="precedent-card">
        <div class="precedent-header">
          <strong style="color:var(--text-main); font-size:0.85rem;">${s.incident_id} • ${s.location || s.asset_type}</strong>
          <span class="sim-score">${Math.round(s.similarity_score * 100)}% Match</span>
        </div>
        <p style="font-size:0.8rem; color:#cbd5e1; margin-bottom:6px;">"${s.narrative}"</p>
        <div style="font-size:0.75rem; color:#34d399;"><strong>Prior CAPA:</strong> ${s.capa_action}</div>
      </div>
    `).join("");
  } else {
    precedentsHtml = `<p style="font-size:0.8rem; color:var(--text-dim);">No past precedents above similarity threshold.</p>`;
  }

  const container = document.getElementById("triage-output-body");
  container.innerHTML = `
    <!-- Top Badges & Risk Index -->
    <div class="badge-row">
      <span class="category-badge ${catClass}">
        ${pred.category.replace('_', ' ')}
        <span style="opacity:0.7; font-size:0.75rem;">(${Math.round(pred.category_confidence * 100)}%)</span>
      </span>
      <span class="severity-pill ${sevClass}">
        ${pred.severity} Priority
      </span>
      <span class="risk-score-badge">
        Risk Index: <strong style="color:#fbbf24; font-size:0.95rem;">${pred.risk_score}</strong> / 10
      </span>
      <span class="risk-score-badge" style="border-color:rgba(16,185,129,0.3);">
        Pillar: <strong style="color:#34d399;">${domain.esg_pillar || 'Social (Safety)'}</strong>
      </span>
    </div>

    <!-- Executive Briefing -->
    <div class="briefing-box">
      <div style="font-size:0.75rem; text-transform:uppercase; color:var(--brand-green); margin-bottom:4px; font-weight:700; letter-spacing:0.05em;">
        Executive Briefing ${reasoning.llm_enhanced ? '• GenAI Enhanced' : '• Expert Rule Synthesis'}
      </div>
      <div>${reasoning.executive_summary}</div>
    </div>

    <!-- Root Cause & Containment Grid -->
    <div class="mini-meta-grid">
      <div class="meta-field">
        <div class="meta-field-label">Root Cause Hypothesis</div>
        <div style="color:#e2e8f0; font-size:0.83rem;">${reasoning.root_cause_hypothesis}</div>
      </div>
      <div class="meta-field">
        <div class="meta-field-label">Immediate Containment (1h)</div>
        <div style="color:#fde68a; font-size:0.83rem;">${reasoning.immediate_containment}</div>
      </div>
    </div>

    <!-- Long Term CAPA -->
    <div class="capa-box">
      <h4>Systemic Corrective & Preventive Action (CAPA)</h4>
      <p>${reasoning.recommended_capa}</p>
      <div style="margin-top:8px; font-size:0.75rem; color:#94a3b8;">
        <strong>Governing Standard:</strong> ${reasoning.compliance_framework}
      </div>
    </div>

    <!-- Historical Precedents Retrieved -->
    <div style="margin-top:6px;">
      <h4 style="font-size:0.8rem; text-transform:uppercase; color:var(--text-muted); margin-bottom:8px; letter-spacing:0.04em;">
        Organizational Safety Memory (RAG Precedents)
      </h4>
      <div class="precedent-list">
        ${precedentsHtml}
      </div>
    </div>
  `;
}

// Fetch Incidents for Tab 2
function debounceFetchIncidents() {
  clearTimeout(debounceTimer);
  debounceTimer = setTimeout(fetchIncidents, 250);
}

async function fetchIncidents() {
  const cat = document.getElementById("filter-category")?.value || "all";
  const sev = document.getElementById("filter-severity")?.value || "all";
  const search = document.getElementById("filter-search")?.value || "";

  const params = new URLSearchParams();
  if (cat !== "all") params.append("category", cat);
  if (sev !== "all") params.append("severity", sev);
  if (search) params.append("search", search);

  const tbody = document.getElementById("incidents-tbody");
  const countLabel = document.getElementById("table-count-label");

  try {
    const res = await fetch(`/api/incidents?${params.toString()}`);
    const data = await res.json();
    allIncidents = data.incidents || [];

    if (countLabel) {
      countLabel.textContent = `Showing ${allIncidents.length} records`;
    }

    if (allIncidents.length === 0) {
      tbody.innerHTML = `<tr><td colspan="8" style="text-align:center; padding:24px; color:var(--text-dim);">No matching incidents found.</td></tr>`;
      return;
    }

    tbody.innerHTML = allIncidents.map(inc => {
      const catClass = `badge-${inc.category}`;
      const sevClass = `sev-${inc.severity}`;
      return `
        <tr>
          <td><strong style="color:var(--text-main);">${inc.incident_id}</strong></td>
          <td style="max-width:320px; font-size:0.83rem;">${inc.narrative}</td>
          <td><span class="category-badge ${catClass}" style="padding:3px 10px; font-size:0.75rem;">${inc.category.replace('_', ' ')}</span></td>
          <td><span class="severity-pill ${sevClass}" style="padding:3px 10px; font-size:0.75rem;">${inc.severity}</span></td>
          <td>${inc.asset_type || 'Solar/Wind'}</td>
          <td>${inc.location || 'Site'}</td>
          <td>${inc.shift || 'General'}</td>
          <td>
            <button class="dataset-btn" onclick="inspectIncident('${inc.incident_id}')" style="padding:4px 10px; font-size:0.75rem;">
              Inspect
            </button>
          </td>
        </tr>
      `;
    }).join("");
  } catch (err) {
    console.error("Failed to fetch incidents:", err);
    tbody.innerHTML = `<tr><td colspan="8" style="text-align:center; color:#f43f5e;">Failed to load records.</td></tr>`;
  }
}

function inspectIncident(id) {
  const item = allIncidents.find(i => i.incident_id === id);
  if (!item) return;

  const modalTitle = document.getElementById("modal-inspect-title");
  const modalBody = document.getElementById("modal-inspect-body");

  if (modalTitle) modalTitle.textContent = `Incident Record: ${item.incident_id}`;
  if (modalBody) {
    modalBody.innerHTML = `
      <div style="margin-bottom:14px; background:rgba(255,255,255,0.03); padding:12px; border-radius:8px;">
        <div style="font-size:0.75rem; text-transform:uppercase; color:var(--text-dim);">Observation Narrative</div>
        <div style="font-size:0.95rem; color:#f8fafc; margin-top:4px;">"${item.narrative}"</div>
      </div>
      <div style="display:grid; grid-template-columns:1fr 1fr; gap:12px; margin-bottom:14px;">
        <div class="meta-field">
          <div class="meta-field-label">Category & Severity</div>
          <div><strong style="text-transform:capitalize;">${item.category.replace('_', ' ')}</strong> (${item.severity.toUpperCase()})</div>
        </div>
        <div class="meta-field">
          <div class="meta-field-label">Asset & Location</div>
          <div>${item.asset_type} • ${item.location} (${item.shift} Shift)</div>
        </div>
      </div>
      <div class="meta-field" style="margin-bottom:12px;">
        <div class="meta-field-label">Root Cause Analysis</div>
        <div style="color:#e2e8f0;">${item.root_cause || 'Pending Root Cause Assessment'}</div>
      </div>
      <div class="capa-box" style="margin-bottom:12px;">
        <h4>Assigned CAPA Action</h4>
        <p>${item.capa_action || 'Implement standard hazard barrier and retraining.'}</p>
      </div>
      <div style="font-size:0.75rem; color:var(--text-dim);">
        <strong>Regulatory Compliance:</strong> ${item.regulatory_framework || 'ISO 45001 / OSHA 1910'}
      </div>
    `;
  }

  const modal = document.getElementById("modal-incident-inspect");
  if (modal) modal.classList.add("active");
}

// RAG Precedent Search Action (Tab 3)
async function searchRAG() {
  const qEl = document.getElementById("rag-search-input");
  const query = (qEl ? qEl.value : "").trim();
  if (!query) return;

  const container = document.getElementById("rag-results-container");
  if (container) container.innerHTML = `<p style="color:var(--text-muted); font-size:0.85rem;">Searching organizational memory...</p>`;

  try {
    const res = await fetch(`/api/similar?query=${encodeURIComponent(query)}&top_k=4`);
    const data = await res.json();
    const results = data.results || [];

    if (results.length === 0) {
      container.innerHTML = `<p style="color:var(--text-dim); font-size:0.85rem;">No historical precedents found matching query.</p>`;
      return;
    }

    container.innerHTML = results.map(r => `
      <div class="precedent-card" style="padding:18px;">
        <div class="precedent-header">
          <span style="font-size:0.9rem; font-weight:700; color:var(--text-main);">
            ${r.incident_id} • ${r.asset_type} (${r.location})
          </span>
          <span class="sim-score" style="font-size:0.82rem; padding:3px 10px;">
            ${Math.round(r.similarity_score * 100)}% Similarity
          </span>
        </div>
        <p style="font-size:0.85rem; color:#f1f5f9; margin-bottom:10px;">"${r.narrative}"</p>
        <div style="display:grid; grid-template-columns:1fr 1fr; gap:10px; font-size:0.8rem;">
          <div style="background:rgba(255,255,255,0.02); padding:8px 12px; border-radius:6px; border:1px solid var(--border-color);">
            <div style="color:var(--text-dim); font-size:0.72rem; text-transform:uppercase;">Root Cause</div>
            <div style="color:#e2e8f0;">${r.root_cause}</div>
          </div>
          <div style="background:rgba(16,185,129,0.05); padding:8px 12px; border-radius:6px; border:1px solid rgba(16,185,129,0.2);">
            <div style="color:var(--brand-green); font-size:0.72rem; text-transform:uppercase;">Historical CAPA</div>
            <div style="color:#cbd5e1;">${r.capa_action}</div>
          </div>
        </div>
      </div>
    `).join("");

  } catch (err) {
    console.error("RAG search error:", err);
    container.innerHTML = `<p style="color:#f43f5e;">Failed to execute precedent search.</p>`;
  }
}

// Portfolio Analytics Action (Tab 4)
async function fetchAnalytics() {
  try {
    const res = await fetch("/api/analytics");
    const data = await res.json();

    // Update KPI values
    const kpiTotal = document.getElementById("kpi-total");
    const kpiHighCrit = document.getElementById("kpi-high-crit");
    const kpiEsgIndex = document.getElementById("kpi-esg-index");
    const kpiDatasetLabel = document.getElementById("kpi-dataset-label");
    const headerCount = document.getElementById("header-records-count");

    if (kpiTotal) kpiTotal.textContent = data.total_incidents || 0;
    if (kpiHighCrit) {
      kpiHighCrit.innerHTML = `${data.high_critical_incidents} <span style="font-size:1rem; opacity:0.8;">(${data.high_critical_percentage}%)</span>`;
    }
    if (kpiEsgIndex) kpiEsgIndex.textContent = `${data.esg_compliance_index}%`;
    if (kpiDatasetLabel) {
      kpiDatasetLabel.textContent = `Active: ${data.active_dataset === 'expanded' ? 'Enriched Benchmark (60)' : 'Bundled Dummy (6)'}`;
    }
    if (headerCount) headerCount.textContent = `${data.total_incidents} Records Indexed`;

    // Render Distribution Bars
    renderBars("analytics-category-bars", data.category_distribution, data.total_incidents, {
      safety: "#60a5fa",
      environment: "#34d399",
      esg_compliance: "#c084fc",
      operational_hazard: "#fbbf24"
    });

    renderBars("analytics-severity-bars", data.severity_distribution, data.total_incidents, {
      low: "#34d399",
      medium: "#fbbf24",
      high: "#fb923c",
      critical: "#f43f5e"
    });

    renderBars("analytics-asset-bars", data.asset_distribution || {}, data.total_incidents, {
      Solar: "#f59e0b",
      Wind: "#06b6d4",
      Hydro: "#3b82f6",
      Storage: "#10b981",
      Substation: "#8b5cf6",
      Corporate: "#94a3b8"
    });

    // Render Systemic Pattern Alerts in Tab 3 & Tab 4
    const patterns = data.recurring_patterns?.top_recurring_themes || [];
    const systemicBox = document.getElementById("analytics-systemic-text");
    if (systemicBox) {
      systemicBox.textContent = data.recurring_patterns?.systemic_alert || "All plant systems operating within normal parameters.";
    }

    const ragPatternsList = document.getElementById("rag-patterns-list");
    if (ragPatternsList) {
      ragPatternsList.innerHTML = patterns.map(p => `
        <div style="background:rgba(255,255,255,0.03); border:1px solid var(--border-color); border-radius:8px; padding:14px;">
          <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
            <span style="font-size:0.75rem; padding:2px 8px; border-radius:4px; font-weight:700; background:${p.risk_alert==='HIGH'?'rgba(244,63,94,0.2)':'rgba(245,158,11,0.2)'}; color:${p.risk_alert==='HIGH'?'#f43f5e':'#fbbf24'};">
              ${p.risk_alert} CLUSTER
            </span>
            <strong style="color:var(--text-main); font-size:0.9rem;">${p.incident_count} Incidents</strong>
          </div>
          <div style="font-size:0.83rem; color:#e2e8f0; font-weight:500;">${p.theme}</div>
        </div>
      `).join("");
    }

  } catch (err) {
    console.error("Failed to fetch analytics:", err);
  }
}

function renderBars(containerId, dataMap, total, colorMap) {
  const container = document.getElementById(containerId);
  if (!container) return;

  const entries = Object.entries(dataMap || {});
  if (entries.length === 0) {
    container.innerHTML = `<p style="font-size:0.8rem; color:var(--text-dim);">No distribution data available.</p>`;
    return;
  }

  container.innerHTML = entries.map(([key, count]) => {
    const pct = total > 0 ? Math.round((count / total) * 100) : 0;
    const color = colorMap[key] || "#94a3b8";
    const label = key.replace('_', ' ').replace(/\b\w/g, c => c.toUpperCase());
    return `
      <div class="bar-row">
        <div class="bar-labels">
          <span style="color:var(--text-main); font-weight:500;">${label}</span>
          <span style="color:var(--text-muted);">${count} (${pct}%)</span>
        </div>
        <div class="bar-track">
          <div class="bar-fill" style="width:${pct}%; background:${color};"></div>
        </div>
      </div>
    `;
  }).join("");
}

// Model Evaluation & Audit Action (Tab 5)
async function fetchEvaluation() {
  try {
    const res = await fetch("/api/evaluation");
    const data = await res.json();
    if (data.error) return;

    // Accuracy & F1
    const catAccEl = document.getElementById("eval-cat-acc");
    const catF1El = document.getElementById("eval-cat-f1");
    const sevAccEl = document.getElementById("eval-sev-acc");
    const sevF1El = document.getElementById("eval-sev-f1");

    if (catAccEl) catAccEl.textContent = `${Math.round(data.category_metrics.accuracy * 100)}%`;
    if (catF1El) catF1El.textContent = `Macro F1: ${data.category_metrics.macro_f1} | Weighted F1: ${data.category_metrics.weighted_f1}`;

    if (sevAccEl) sevAccEl.textContent = `${Math.round(data.severity_metrics.accuracy * 100)}%`;
    if (sevF1El) sevF1El.textContent = `Macro F1: ${data.severity_metrics.macro_f1} | Weighted F1: ${data.severity_metrics.weighted_f1}`;

    // Render Confusion Matrices
    renderConfusionMatrix("cat-cm-container", data.category_metrics.confusion_matrix, data.category_metrics.classes);
    renderConfusionMatrix("sev-cm-container", data.severity_metrics.confusion_matrix, data.severity_metrics.classes);

    // Assumptions & Failure Cases
    const assumptionsList = document.getElementById("eval-assumptions-list");
    if (assumptionsList) {
      assumptionsList.innerHTML = (data.assumptions || []).map(a => `<li>${a}</li>`).join("");
      const failureCases = data.failure_cases_and_edge_conditions || [];
      if (failureCases.length > 0) {
        assumptionsList.innerHTML += `
          <div style="margin-top:12px; font-weight:700; color:#fbbf24;">Known Edge Cases & Mitigations:</div>
          ${failureCases.map(f => `<li style="margin-top:6px;"><strong>${f.case}:</strong> ${f.description} <br/><span style="color:#34d399;">Mitigation:</span> ${f.mitigation}</li>`).join("")}
        `;
      }
    }

    // Roadmap Gaps
    const roadmapList = document.getElementById("eval-roadmap-list");
    if (roadmapList) {
      roadmapList.innerHTML = (data.production_readiness_gaps || []).map(g => `
        <li style="margin-bottom:8px;">
          <strong style="color:var(--text-main);">${g.gap}:</strong><br/>
          <span style="color:#94a3b8;">${g.solution}</span>
        </li>
      `).join("");
    }

  } catch (err) {
    console.error("Failed to load evaluation:", err);
  }
}

function renderConfusionMatrix(containerId, cm, classes) {
  const container = document.getElementById(containerId);
  if (!container || !cm) return;

  const labels = classes.map(c => c.replace('_', ' ').substring(0, 7).toUpperCase());

  let tableHtml = `<table style="width:100%; border-collapse:collapse; font-size:0.75rem; text-align:center;">`;
  tableHtml += `<thead><tr><th style="padding:6px; color:var(--text-dim);">Pred &rarr;<br/>True &darr;</th>`;
  labels.forEach(l => {
    tableHtml += `<th style="padding:6px; color:var(--text-muted);">${l}</th>`;
  });
  tableHtml += `</tr></thead><tbody>`;

  cm.forEach((row, rIdx) => {
    tableHtml += `<tr><th style="padding:6px; color:var(--text-muted); text-align:left;">${labels[rIdx]}</th>`;
    row.forEach((val, cIdx) => {
      const isDiag = rIdx === cIdx;
      const highlight = isDiag ? "background:rgba(16,185,129,0.25); color:#34d399; font-weight:700;" : "color:#64748b;";
      tableHtml += `<td style="padding:10px 4px; border:1px solid var(--border-color); ${highlight}">${val}</td>`;
    });
    tableHtml += `</tr>`;
  });

  tableHtml += `</tbody></table>`;
  container.innerHTML = tableHtml;
}

// Dataset Switcher
async function switchDataset(name) {
  try {
    const res = await fetch("/api/dataset/switch", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ dataset_name: name })
    });
    const data = await res.json();

    // Toggle button UI
    const btnExp = document.getElementById("btn-ds-expanded");
    const btnDum = document.getElementById("btn-ds-dummy");
    if (name === "expanded") {
      btnExp?.classList.add("active");
      btnDum?.classList.remove("active");
    } else {
      btnDum?.classList.add("active");
      btnExp?.classList.remove("active");
    }

    // Refresh all data
    fetchIncidents();
    fetchAnalytics();
    fetchEvaluation();
    searchRAG();
    runTriage();
  } catch (err) {
    console.error("Dataset toggle error:", err);
  }
}

// Modal Controllers
function openNewIncidentModal() {
  const modal = document.getElementById("modal-new-incident");
  if (modal) modal.classList.add("active");
}

function closeModal(modalId) {
  const modal = document.getElementById(modalId);
  if (modal) modal.classList.remove("active");
}

async function submitNewIncident(e) {
  e.preventDefault();
  const narrative = document.getElementById("new-incident-narrative")?.value;
  const asset = document.getElementById("new-incident-asset")?.value;
  const location = document.getElementById("new-incident-location")?.value;
  const shift = document.getElementById("new-incident-shift")?.value;

  if (!narrative) return;

  try {
    const res = await fetch("/api/incidents", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        narrative,
        asset_type: asset,
        location,
        shift
      })
    });

    const data = await res.json();
    if (data.status === "success") {
      alert(`Incident ${data.record.incident_id} successfully triaged and indexed!`);
      closeModal("modal-new-incident");
      document.getElementById("new-incident-narrative").value = "";
      
      // Refresh views
      fetchIncidents();
      fetchAnalytics();
      searchRAG();
      
      // Load into triage viewer
      renderTriageResults(data.triage);
      switchTab("triage", document.querySelectorAll(".tab-link")[0]);
    }
  } catch (err) {
    console.error("Failed to submit incident:", err);
    alert("Error logging incident: " + err.message);
  }
}
