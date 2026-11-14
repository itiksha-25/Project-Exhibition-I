/**
 * Adaptive Multi-SLM AI Orchestrator - Client Controller
 * Intelligent Multi-Model Routing, Memory Management & MCP Testbed
 */

document.addEventListener("DOMContentLoaded", () => {
  // ==========================================
  // STATE MANAGEMENT
  // ==========================================
  const state = {
    activeTab: "tab-chat",
    activeModel: "Detecting...",
    ramPercent: 0,
    isStreaming: false,
    domainsMeta: {}
  };

  // ==========================================
  // DOM ELEMENT REFERENCES
  // ==========================================
  const navTabs = document.querySelectorAll(".nav-tab");
  const tabPanes = document.querySelectorAll(".tab-pane");

  // Header Telemetry
  const telemetryActiveModel = document.getElementById("telemetry-active-model");
  const telemetryRamVal = document.getElementById("telemetry-ram-val");
  const telemetryRamBar = document.getElementById("telemetry-ram-bar");

  // Chat Elements
  const chatMessages = document.getElementById("chat-messages");
  const chatInput = document.getElementById("chat-input");
  const chatSubmitBtn = document.getElementById("chat-submit-btn");
  const selectDomainOverride = document.getElementById("select-domain-override");
  const btnClearChat = document.getElementById("btn-clear-chat");

  // Chat Inspector Elements
  const insDomain = document.getElementById("ins-domain");
  const insModel = document.getElementById("ins-model");
  const insQuant = document.getElementById("ins-quant");
  const insRouteLatency = document.getElementById("ins-route-latency");
  const insMemLatency = document.getElementById("ins-mem-latency");
  const insTtftTps = document.getElementById("ins-ttft-tps");

  const latEvictVal = document.getElementById("lat-evict-val");
  const latLoadVal = document.getElementById("lat-load-val");
  const latInferVal = document.getElementById("lat-infer-val");
  const barEvict = document.getElementById("bar-evict");
  const barLoad = document.getElementById("bar-load");
  const barInfer = document.getElementById("bar-infer");
  const badgeSwitchState = document.getElementById("badge-switch-state");

  const sidebarRamPercent = document.getElementById("sidebar-ram-percent");
  const sidebarRamUsed = document.getElementById("sidebar-ram-used");
  const sidebarRamFree = document.getElementById("sidebar-ram-free");
  const sidebarWarmCount = document.getElementById("sidebar-warm-count");

  // Router Lab Elements
  const routerTestInput = document.getElementById("router-test-input");
  const btnTestRoute = document.getElementById("btn-test-route");
  const routerResultBox = document.getElementById("router-result-box");
  const btnRunRouterBench = document.getElementById("btn-run-router-bench");
  const matrixContainer = document.getElementById("matrix-container");
  const taxonomyTbody = document.getElementById("taxonomy-tbody");

  // Memory Orchestrator Simulator Elements
  const simSize = document.getElementById("sim-size");
  const simSizeVal = document.getElementById("sim-size-val");
  const simBus = document.getElementById("sim-bus");
  const calcEvict = document.getElementById("calc-evict");
  const calcLoad = document.getElementById("calc-load");
  const calcInfer = document.getElementById("calc-infer");
  const calcTotal = document.getElementById("calc-total");
  const lruQueueDisplay = document.getElementById("lru-queue-display");
  const hardwareSpecBox = document.getElementById("hardware-spec-box");
  const btnForceEvict = document.getElementById("btn-force-evict");

  // Vector DB Elements
  const memorySearchInput = document.getElementById("memory-search-input");
  const btnMemorySearch = document.getElementById("btn-memory-search");
  const memoryResultsTray = document.getElementById("memory-results-tray");
  const factInputText = document.getElementById("fact-input-text");
  const factInputDomain = document.getElementById("fact-input-domain");
  const factInputImportance = document.getElementById("fact-input-importance");
  const btnSaveFact = document.getElementById("btn-save-fact");

  // Benchmark Elements
  const quantMatrixContainer = document.getElementById("quant-matrix-container");
  const selectBenchDomain = document.getElementById("select-bench-domain");
  const btnRunLiveBench = document.getElementById("btn-run-live-bench");
  const benchResultsTray = document.getElementById("bench-results-tray");

  // MCP Testbed Elements
  const mcpSelectTool = document.getElementById("mcp-select-tool");
  const mcpArgsInput = document.getElementById("mcp-args-input");
  const btnInvokeMcp = document.getElementById("btn-invoke-mcp");
  const mcpResponsePreview = document.getElementById("mcp-response-preview");
  const mcpResourcePreview = document.getElementById("mcp-resource-preview");
  const resourceTabs = document.querySelectorAll(".resource-tabs button");

  // Specs Elements
  const specsCardsContainer = document.getElementById("specs-cards-container");

  // ==========================================
  // TAB NAVIGATION
  // ==========================================
  navTabs.forEach((tab) => {
    tab.addEventListener("click", () => {
      navTabs.forEach((t) => t.classList.remove("active"));
      tabPanes.forEach((p) => p.classList.remove("active"));

      tab.classList.add("active");
      const targetId = tab.dataset.tab;
      const targetPane = document.getElementById(targetId);
      if (targetPane) targetPane.classList.add("active");
      state.activeTab = targetId;

      // Lazy load tab data
      if (targetId === "tab-router" && !taxonomyTbody.hasChildNodes()) loadTaxonomy();
      if (targetId === "tab-benchmarks" && quantMatrixContainer.querySelector(".loader-box")) loadQuantMatrix();
      if (targetId === "tab-specs") loadSystemSpecs();
    });
  });

  // ==========================================
  // TELEMETRY & SYSTEM MONITORING POLLING
  // ==========================================
  async function pollSystemStats() {
    try {
      const res = await fetch("/api/system/stats");
      if (!res.ok) return;
      const data = await res.json();

      const mem = data.memory;
      state.activeModel = data.active_model || "None (Cold)";
      state.ramPercent = mem.ram_usage_percent;

      // Update Header
      telemetryActiveModel.textContent = state.activeModel;
      telemetryRamVal.textContent = `${mem.ram_usage_percent}%`;
      telemetryRamBar.style.width = `${Math.min(100, mem.ram_usage_percent)}%`;

      // Update Sidebar
      if (sidebarRamPercent) sidebarRamPercent.textContent = `${mem.ram_usage_percent}%`;
      if (sidebarRamUsed) sidebarRamUsed.textContent = `${(mem.used_ram_mb / 1024).toFixed(1)} GB`;
      if (sidebarRamFree) sidebarRamFree.textContent = `${(mem.free_ram_mb / 1024).toFixed(1)} GB`;
      if (sidebarWarmCount) sidebarWarmCount.textContent = data.lru_queue.length;

      // Render LRU Queue in Tab 3
      renderLruQueue(data.lru_queue, data.models_catalog);
      renderHardwareSpec(mem, data.telemetry);
    } catch (e) {
      console.warn("Telemetry polling warning:", e);
    }
  }

  setInterval(pollSystemStats, 4000);
  pollSystemStats();

  // ==========================================
  // CHAT STUDIO LOGIC (STREAMING & ROUTING)
  // ==========================================

  // Quick Prompt Chips
  document.querySelectorAll(".chip-btn").forEach((chip) => {
    chip.addEventListener("click", () => {
      chatInput.value = chip.dataset.query;
      chatInput.focus();
    });
  });

  if (chatSubmitBtn) {
    chatSubmitBtn.addEventListener("click", () => handleSendMessage());
  }

  if (chatInput) {
    chatInput.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        handleSendMessage();
      }
    });
  }

  if (btnClearChat) {
    btnClearChat.addEventListener("click", () => {
      chatMessages.innerHTML = `
        <div class="message-row assistant intro-card">
          <div class="msg-avatar">🤖</div>
          <div class="msg-body">
            <div class="msg-header"><span class="agent-title">Adaptive SLM Orchestrator</span></div>
            <div class="msg-text"><p>Dialogue cleared. Ask a question to begin a fresh multi-model session.</p></div>
          </div>
        </div>`;
    });
  }

  async function handleSendMessage() {
    const text = chatInput.value.trim();
    if (!text || state.isStreaming) return;

    const domainOverride = selectDomainOverride.value || null;
    chatInput.value = "";

    // 1. Render User Message
    appendUserMessage(text);

    // 2. Prepare Assistant Response Placeholder
    const assistantRow = appendAssistantPlaceholder();
    const assistantTextEl = assistantRow.querySelector(".msg-text");
    const assistantBadgesBar = assistantRow.querySelector(".msg-badges-bar");

    state.isStreaming = true;
    chatSubmitBtn.disabled = true;

    try {
      const response = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query: text,
          domain_override: domainOverride,
          stream: true
        })
      });

      if (!response.ok) {
        throw new Error(`Server returned HTTP ${response.status}`);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      let fullTokens = "";

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n\n");
        buffer = lines.pop(); // Keep incomplete chunk

        
for (const line of lines) {
  const trimmed = line.trim();
  if (!trimmed.startsWith("data: ")) continue;

  const jsonStr = trimmed.substring(6);

  try {
    const event = JSON.parse(jsonStr);

    if (event.type === "meta") {
      const r = event.routing || {};
      const mem = event.memory || {};
      const provider = r.provider || "unknown";
      const isOllama = provider === "ollama";

      const escapeHtml = (value) =>
        String(value ?? "").replace(/[&<>"']/g, (char) => ({
          "&": "&amp;",
          "<": "&lt;",
          ">": "&gt;",
          '"': "&quot;",
          "'": "&#39;"
        })[char]);

      const domain = escapeHtml(r.domain_name || "General");
      const model = escapeHtml(r.target_model || "Unknown model");
      const providerLabel = escapeHtml(provider.toUpperCase());
      const confidence = Number(r.confidence || 0);
      const fallbacks = (event.provider_fallbacks || [])
        .map(escapeHtml);

      const modelLabel = isOllama
        ? `${model} [${escapeHtml(r.recommended_quant || "N/A")}]`
        : `${providerLabel}: ${model}`;

      assistantBadgesBar.innerHTML = `
        <span class="badge-chip domain">
          ${domain} (${(confidence * 100).toFixed(0)}%)
        </span>
        <span class="badge-chip model">${modelLabel}</span>
        <span class="badge-chip latency">
          ⚡ Router: ${escapeHtml(r.latency_ms ?? "N/A")}ms
        </span>
        ${mem.injected_count > 0
          ? `<span class="badge-chip hot">
              🧠 Recalled ${Number(mem.injected_count)} Context Facts
            </span>`
          : ""}
        ${fallbacks.length > 0
          ? `<span class="badge-chip latency">
              Fallback: ${fallbacks.join(" → ")}
            </span>`
          : ""}
      `;

      // Update sidebar inspector
      insDomain.textContent =
        `${r.domain_name || "General"} (${(confidence * 100).toFixed(0)}%)`;

      insModel.textContent = r.target_model || "Unknown";

      insQuant.textContent = isOllama
        ? `${r.recommended_quant || "N/A"} (Min ${r.min_precision_bits ?? "N/A"}-bit)`
        : "Cloud inference (quantization not reported)";

      insRouteLatency.textContent =
        `${r.latency_ms ?? "N/A"} ms (${r.decision_tier || "N/A"})`;

      insMemLatency.textContent =
        `${mem.lookup_latency_ms ?? "N/A"} ms`;

      // Reset local-model swap metrics for cloud inference
      if (!isOllama) {
        latEvictVal.textContent = "N/A";
        latLoadVal.textContent = "N/A";
        latInferVal.textContent = "Cloud inference";

        barEvict.style.width = "0%";
        barLoad.style.width = "0%";
        barInfer.style.width = "0%";

        badgeSwitchState.textContent = "CLOUD PROVIDER";
        badgeSwitchState.className = "badge-tag hot";

        insTtftTps.textContent = "N/A / N/A";
      }
    }

    else if (event.type === "staging") {
      const swap = event.swap;

      latEvictVal.textContent = `${swap.t_evict_ms} ms`;
      latLoadVal.textContent = `${swap.t_load_ms} ms`;
      latInferVal.textContent = `${swap.t_infer_ms} ms`;

      const maxMs = Math.max(1000, swap.t_total_ms);

      barEvict.style.width =
        `${(swap.t_evict_ms / maxMs) * 100}%`;

      barLoad.style.width =
        `${(swap.t_load_ms / maxMs) * 100}%`;

      barInfer.style.width =
        `${(swap.t_infer_ms / maxMs) * 100}%`;

      if (swap.is_hot_switch) {
        badgeSwitchState.textContent = "HOT CACHE (0ms swap)";
        badgeSwitchState.className = "badge-tag hot";
      } else {
        badgeSwitchState.textContent =
          `COLD SWITCH (${swap.t_load_ms}ms I/O)`;
        badgeSwitchState.className = "badge-tag cold";
      }
    }

    else if (event.type === "ttft") {
      insTtftTps.textContent = `${event.ttft_ms} ms / -- tps`;
    }

    else if (event.type === "token") {
      fullTokens += event.content || "";
      assistantTextEl.innerHTML = formatMarkdown(fullTokens);
      chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    else if (event.type === "done") {
      const previousTtft =
        insTtftTps.textContent.split("/")[0].trim();

      insTtftTps.textContent =
        `${previousTtft} / ${event.tps ?? "N/A"} tps`;
    }

    else if (event.type === "error") {
      const errorMessage =
        event.message || "Inference failed. Please try again.";

      assistantTextEl.textContent =
        fullTokens
          ? `${fullTokens}\n\nError: ${errorMessage}`
          : `Error: ${errorMessage}`;

      console.error("[Inference Error]", errorMessage);
    }

  } catch (err) {
    console.error("SSE parse error:", err);
  }
}

      }
    } catch (e) {
      assistantTextEl.innerHTML = `<p style="color: var(--accent-rose);">Error connecting to local SLM engine: ${e.message}</p>`;
    } finally {
      state.isStreaming = false;
      chatSubmitBtn.disabled = false;
      pollSystemStats();
    }
  }

  function appendUserMessage(text) {
    const row = document.createElement("div");
    row.className = "message-row user";
    row.innerHTML = `
      <div class="msg-avatar">👤</div>
      <div class="msg-body">
        <div class="msg-text">${escapeHtml(text)}</div>
      </div>
    `;
    chatMessages.appendChild(row);
    chatMessages.scrollTop = chatMessages.scrollHeight;
  }

  function appendAssistantPlaceholder() {
    const row = document.createElement("div");
    row.className = "message-row assistant";
    row.innerHTML = `
      <div class="msg-avatar">🤖</div>
      <div class="msg-body">
        <div class="msg-badges-bar">
          <span class="badge-chip">⚡ Evaluating intent & context...</span>
        </div>
        <div class="msg-text"><span class="cursor-typing">▊</span></div>
      </div>
    `;
    chatMessages.appendChild(row);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    return row;
  }

  // ==========================================
  // OM'S INTENT ROUTER LAB LOGIC
  // ==========================================

  // Preset chips in tab 2
  document.querySelectorAll(".test-chip").forEach((chip) => {
    chip.addEventListener("click", () => {
      routerTestInput.value = chip.dataset.test;
      btnTestRoute.click();
    });
  });

  if (btnTestRoute) {
    btnTestRoute.addEventListener("click", async () => {
      const q = routerTestInput.value.trim();
      if (!q) return;

      routerResultBox.innerHTML = "<div class='result-placeholder'>Executing cascade classification...</div>";

      try {
        const res = await fetch("/api/route", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ query: q })
        });
        const data = await res.json();

        // Render colorful breakdown
        let probBars = "";
        for (const [dom, p] of Object.entries(data.domain_probabilities)) {
          const pct = (p * 100).toFixed(1);
          probBars += `
            <div style="margin-bottom: 8px;">
              <div style="display: flex; justify-content: space-between; font-size: 11px; margin-bottom: 3px;">
                <span>${dom.toUpperCase()}</span>
                <strong>${pct}%</strong>
              </div>
              <div class="progress-track">
                <div class="progress-fill infer" style="width: ${pct}%;"></div>
              </div>
            </div>`;
        }

        routerResultBox.innerHTML = `
          <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px;">
            <div>
              <div style="font-size: 11px; color: var(--text-dim);">CLASSIFICATION DECISION:</div>
              <h3 style="font-size: 20px; color: var(--accent-cyan); margin: 4px 0 10px 0;">${data.domain_name}</h3>
              <div class="detail-line"><span>Confidence:</span> <strong>${(data.confidence * 100).toFixed(1)}%</strong></div>
              <div class="detail-line"><span>Target SLM:</span> <strong style="color: var(--accent-purple);">${data.target_model}</strong></div>
              <div class="detail-line"><span>Recommended Quant:</span> <strong>${data.recommended_quant} (${data.min_precision_bits}-bit)</strong></div>
              <div class="detail-line"><span>Decision Tier:</span> <strong>${data.decision_tier}</strong></div>
              <div class="detail-line"><span>Execution Latency:</span> <strong style="color: var(--accent-emerald);">${data.latency_ms} ms (&lt; 25ms budget)</strong></div>
              <p style="font-size: 12px; color: var(--text-muted); margin-top: 10px; background: rgba(255,255,255,0.03); padding: 8px; border-radius: 6px;">
                <strong>Rationale:</strong> ${data.decision_rationale}
              </p>
            </div>
            <div>
              <div style="font-size: 11px; color: var(--text-dim); margin-bottom: 8px;">HYBRID CENTROID PROBABILITIES:</div>
              ${probBars}
            </div>
          </div>
        `;
      } catch (err) {
        routerResultBox.innerHTML = `<div style="color: var(--accent-rose);">Error: ${err.message}</div>`;
      }
    });
  }

  if (btnRunRouterBench) {
    btnRunRouterBench.addEventListener("click", async () => {
      matrixContainer.innerHTML = "<div class='loader-box'>Evaluating test dataset across all domains...</div>";
      try {
        const res = await fetch("/api/router/benchmark");
        const data = await res.json();

        let tableHtml = `
          <div style="margin-bottom: 12px; display: flex; gap: 20px;">
            <div><strong>Accuracy:</strong> ${(data.accuracy * 100).toFixed(1)}%</div>
            <div><strong>Avg Latency:</strong> ${data.average_latency_ms} ms</div>
            <div><strong>Max Latency:</strong> ${data.max_latency_ms} ms</div>
            <div><strong>Budget Adherence:</strong> <span style="color: var(--accent-emerald);">${data.within_budget ? "PASSED (< 25ms)" : "EXCEEDED"}</span></div>
          </div>
          <table class="data-table">
            <thead>
              <tr>
                <th>True \\ Pred</th>
                <th>Math</th>
                <th>Medical</th>
                <th>Code</th>
                <th>Science</th>
                <th>Writing</th>
                <th>General</th>
              </tr>
            </thead>
            <tbody>`;

        for (const [trueDom, row] of Object.entries(data.confusion_matrix)) {
          tableHtml += `<tr><td><strong>${trueDom}</strong></td>`;
          for (const predDom of ["mathematics", "medical", "code", "science", "writing", "general"]) {
            const count = row[predDom] || 0;
            const isMatch = trueDom === predDom;
            const bg = isMatch ? (count > 0 ? "rgba(52, 211, 153, 0.2)" : "") : (count > 0 ? "rgba(244, 63, 94, 0.2)" : "");
            tableHtml += `<td style="background: ${bg}; text-align: center;">${count}</td>`;
          }
          tableHtml += `</tr>`;
        }

        tableHtml += `</tbody></table>`;
        matrixContainer.innerHTML = tableHtml;
      } catch (e) {
        matrixContainer.innerHTML = `<div style="color: var(--accent-rose);">Benchmark error: ${e.message}</div>`;
      }
    });
  }

  async function loadTaxonomy() {
    try {
      const res = await fetch("/api/models");
      const data = await res.json();
      state.domainsMeta = data.domain_assignments;

      let rows = "";
      for (const [id, d] of Object.entries(data.domain_assignments)) {
        rows += `
          <tr>
            <td><strong style="color: ${d.color};">${d.icon} ${d.name}</strong></td>
            <td><code>${d.primary_model}</code></td>
            <td><span class="badge-tag hot">${d.recommended_quant}</span></td>
            <td>${d.description}</td>
          </tr>
        `;
      }
      taxonomyTbody.innerHTML = rows;
    } catch (e) {
      console.warn("Taxonomy load warning:", e);
    }
  }

  // ==========================================
  // MEMORY & SWAPPING SIMULATOR
  // ==========================================
  function updateSwapSimulator() {
    const sizeMb = parseFloat(simSize.value);
    const busSpeed = parseFloat(simBus.value);
    const isHot = document.querySelector("input[name='sim-state']:checked").value === "hot";

    simSizeVal.textContent = `${sizeMb} MB`;

    if (isHot) {
      calcEvict.textContent = "0.0 ms";
      calcLoad.textContent = "0.0 ms";
      calcInfer.textContent = "450.0 ms";
      calcTotal.textContent = "450.0 ms";
    } else {
      const tLoadSec = sizeMb / busSpeed;
      const tLoadMs = tLoadSec * 1000.0;
      const tEvictMs = 45.0;
      const tInferMs = 500.0;
      const tTotal = tEvictMs + tLoadMs + tInferMs;

      calcEvict.textContent = `${tEvictMs.toFixed(1)} ms`;
      calcLoad.textContent = `${tLoadMs.toFixed(1)} ms`;
      calcInfer.textContent = `${tInferMs.toFixed(1)} ms`;
      calcTotal.textContent = `${tTotal.toFixed(1)} ms`;
    }
  }

  if (simSize) simSize.addEventListener("input", updateSwapSimulator);
  if (simBus) simBus.addEventListener("change", updateSwapSimulator);
  document.querySelectorAll("input[name='sim-state']").forEach((r) => r.addEventListener("change", updateSwapSimulator));

  if (btnForceEvict) {
    btnForceEvict.addEventListener("click", async () => {
      try {
        await fetch("/api/models/evict", { method: "POST" });
        pollSystemStats();
        alert("Zero-leakage memory eviction completed across all models.");
      } catch (e) {
        alert("Eviction error: " + e.message);
      }
    });
  }

  function renderLruQueue(queue, catalog) {
    if (!lruQueueDisplay) return;
    if (!queue || queue.length === 0) {
      lruQueueDisplay.innerHTML = "<div class='result-placeholder'>No models currently warm in RAM (Zero active overhead).</div>";
      return;
    }

    let html = "<div style='font-size: 11px; color: var(--text-dim); margin-bottom: 8px;'>ACTIVE LRU QUEUE (Capacity: 3 Models):</div>";
    queue.forEach((m, idx) => {
      const meta = catalog[m] || {};
      html += `
        <div style="background: var(--bg-deep); border: 1px solid var(--border-subtle); padding: 10px; border-radius: 8px; margin-bottom: 8px; display: flex; justify-content: space-between; align-items: center;">
          <div>
            <strong style="color: var(--accent-cyan); font-size: 13px;">#${idx + 1} ${m}</strong>
            <div style="font-size: 11px; color: var(--text-dim);">${meta.size_mb || 2048} MB • ${meta.quantization || "Q4_K_M"}</div>
          </div>
          <span class="badge-tag hot">WARM IN RAM</span>
        </div>
      `;
    });
    lruQueueDisplay.innerHTML = html;
  }

  function renderHardwareSpec(mem, telemetry) {
    if (!hardwareSpecBox) return;
    hardwareSpecBox.innerHTML = `
      <div class="detail-line"><span>Total Physical RAM:</span> <strong>${(mem.total_ram_mb / 1024).toFixed(1)} GB</strong></div>
      <div class="detail-line"><span>Available / Free:</span> <strong>${(mem.free_ram_mb / 1024).toFixed(1)} GB</strong></div>
      <div class="detail-line"><span>RAM Pressure Threshold:</span> <strong>${mem.threshold_percent}%</strong></div>
      <div class="detail-line"><span>Cache Hit Ratio:</span> <strong style="color: var(--accent-emerald);">${telemetry.cache_hit_ratio_percent || 100}%</strong></div>
      <div class="detail-line"><span>Average Model TTFT:</span> <strong>${telemetry.avg_ttft_ms || 0} ms</strong></div>
      <div class="detail-line"><span>Average Generation TPS:</span> <strong>${telemetry.avg_tps || 0} tokens/sec</strong></div>
    `;
  }

  // ==========================================
  // VECTOR DB & CONTEXT EXPLORER
  // ==========================================
  if (btnMemorySearch) {
    btnMemorySearch.addEventListener("click", async () => {
      const q = memorySearchInput.value.trim();
      if (!q) return;

      memoryResultsTray.innerHTML = "<div class='loader-box'>Searching semantic vectors...</div>";
      try {
        const res = await fetch(`/api/memory/search?q=${encodeURIComponent(q)}&limit=4`);
        const data = await res.json();

        if (data.results.length === 0) {
          memoryResultsTray.innerHTML = "<div class='result-placeholder'>No matching semantic records found.</div>";
          return;
        }

        let html = `<div style="font-size: 11px; color: var(--text-dim); margin-bottom: 8px;">Query Latency: ${data.latency_ms} ms (${data.count} results):</div>`;
        data.results.forEach((item) => {
          html += `
            <div style="background: var(--bg-deep); border: 1px solid var(--border-subtle); padding: 10px; border-radius: 8px; margin-bottom: 8px;">
              <div style="display: flex; justify-content: space-between; font-size: 11px; margin-bottom: 4px;">
                <span style="color: var(--accent-cyan); font-weight: 600;">Domain: ${item.domain}</span>
                <span>Similarity: ${(item.similarity * 100).toFixed(1)}%</span>
              </div>
              <div style="font-size: 13px; color: var(--text-main);">${escapeHtml(item.text)}</div>
            </div>`;
        });
        memoryResultsTray.innerHTML = html;
      } catch (e) {
        memoryResultsTray.innerHTML = `<div style="color: var(--accent-rose);">Error: ${e.message}</div>`;
      }
    });
  }

  if (btnSaveFact) {
    btnSaveFact.addEventListener("click", async () => {
      const fact = factInputText.value.trim();
      if (!fact) return;

      try {
        const res = await fetch("/api/memory/add", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            fact: fact,
            domain: factInputDomain.value,
            importance: parseFloat(factInputImportance.value)
          })
        });
        const data = await res.json();
        factInputText.value = "";
        alert(`Fact stored into vector store! Assigned ID: ${data.item_id}`);
      } catch (e) {
        alert("Failed to store fact: " + e.message);
      }
    });
  }

  // ==========================================
  // BENCHMARK LAB
  // ==========================================
  async function loadQuantMatrix() {
    try {
      const res = await fetch("/api/benchmarks/matrix");
      const data = await res.json();

      let tableHtml = `
        <table class="data-table">
          <thead>
            <tr>
              <th>Quantization Level</th>
              <th>Format</th>
              <th>Bits / Weight</th>
              <th>7B Footprint</th>
              <th>Avg TPS</th>
              <th>Math Ret.</th>
              <th>Med Ret.</th>
              <th>Writing Ret.</th>
            </tr>
          </thead>
          <tbody>`;

      data.profiles.forEach((p) => {
        tableHtml += `
          <tr>
            <td><strong style="color: var(--accent-cyan);">${p.quant_level}</strong></td>
            <td>${p.format}</td>
            <td>${p.bits_per_weight}</td>
            <td>${p.size_7b_gb} GB</td>
            <td><strong style="color: var(--accent-emerald);">${p.avg_tps} tps</strong></td>
            <td>${(p.retention.mathematics * 100).toFixed(1)}%</td>
            <td>${(p.retention.medical * 100).toFixed(1)}%</td>
            <td>${(p.retention.writing * 100).toFixed(1)}%</td>
          </tr>
        `;
      });

      tableHtml += `</tbody></table>
        <p style="font-size: 12px; color: var(--text-muted); margin-top: 14px; background: rgba(255,255,255,0.03); padding: 10px; border-radius: 8px;">
          <strong>Key Research Finding:</strong> ${data.key_finding}
        </p>
      `;
      quantMatrixContainer.innerHTML = tableHtml;
    } catch (e) {
      quantMatrixContainer.innerHTML = `<div style="color: var(--accent-rose);">Failed to load matrix: ${e.message}</div>`;
    }
  }

  if (btnRunLiveBench) {
    btnRunLiveBench.addEventListener("click", async () => {
      const domain = selectBenchDomain.value;
      benchResultsTray.innerHTML = "<div class='loader-box'>Executing empirical benchmark on local SLM...</div>";

      try {
        const res = await fetch(`/api/benchmarks/run?domain=${domain}&limit=1`);
        const data = await res.json();

        let detailsHtml = "";
        data.test_details.forEach((t) => {
          detailsHtml += `
            <div style="background: var(--bg-deep); border: 1px solid var(--border-subtle); padding: 14px; border-radius: 8px; margin-top: 10px;">
              <div style="font-size: 11px; color: var(--text-dim); margin-bottom: 4px;">SOURCE: ${t.source}</div>
              <div style="font-size: 13px; font-weight: 600; margin-bottom: 8px;">Prompt: ${escapeHtml(t.prompt)}</div>
              <div style="font-size: 12px; color: var(--accent-cyan); margin-bottom: 6px;">Ground Truth: ${t.ground_truth}</div>
              <div style="font-size: 12px; background: rgba(0,0,0,0.3); padding: 10px; border-radius: 6px;">
                <strong>Output:</strong> ${escapeHtml(t.model_output)}
              </div>
            </div>`;
        });

        benchResultsTray.innerHTML = `
          <div style="display: flex; gap: 20px; font-size: 13px; border-bottom: 1px solid var(--border-subtle); padding-bottom: 10px;">
            <div>Tested Model: <strong style="color: var(--accent-purple);">${data.model_tested}</strong></div>
            <div>Avg TTFT: <strong style="color: var(--accent-emerald);">${data.average_ttft_ms} ms</strong></div>
            <div>Avg TPS: <strong style="color: var(--accent-cyan);">${data.average_tps} tokens/s</strong></div>
            <div>Quality Score: <strong>${(data.domain_quality_score * 100).toFixed(0)}%</strong></div>
          </div>
          ${detailsHtml}
        `;
      } catch (e) {
        benchResultsTray.innerHTML = `<div style="color: var(--accent-rose);">Benchmark failed: ${e.message}</div>`;
      }
    });
  }

  // ==========================================
  // MCP PROTOCOL TESTBED LOGIC
  // ==========================================
  const toolArgPresets = {
    route_and_infer: '{"query": "Compute the eigenvalues of a 2x2 identity matrix."}',
    classify_intent: '{"query": "What are the common causes of acute respiratory distress syndrome?"}',
    query_semantic_memory: '{"query": "diabetes", "top_k": 2}',
    add_memory_fact: '{"fact": "Standard atmospheric pressure is 101.325 kPa.", "domain": "science", "importance": 0.9}',
    get_system_telemetry: '{}',
    run_quantization_benchmark: '{"domain": "code"}'
  };

  if (mcpSelectTool) {
    mcpSelectTool.addEventListener("change", () => {
      const tool = mcpSelectTool.value;
      if (toolArgPresets[tool]) {
        mcpArgsInput.value = toolArgPresets[tool];
      }
    });
  }

  if (btnInvokeMcp) {
    btnInvokeMcp.addEventListener("click", async () => {
      const tool = mcpSelectTool.value;
      let args = {};
      try {
        args = JSON.parse(mcpArgsInput.value);
      } catch (err) {
        alert("Invalid JSON in Tool Arguments: " + err.message);
        return;
      }

      mcpResponsePreview.textContent = "// Invoking MCP Tool via JSON-RPC...";

      try {
        const res = await fetch("/api/mcp/invoke", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            tool_name: tool,
            arguments: args
          })
        });
        const data = await res.json();
        mcpResponsePreview.textContent = JSON.stringify(data, null, 2);
      } catch (e) {
        mcpResponsePreview.textContent = `// MCP Error: ${e.message}`;
      }
    });
  }

  resourceTabs.forEach((btn) => {
    btn.addEventListener("click", async () => {
      resourceTabs.forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      const uri = btn.dataset.res;

      mcpResourcePreview.textContent = `// Fetching resource ${uri}...`;
      try {
        const res = await fetch(`/api/mcp/resources?uri=${encodeURIComponent(uri)}`);
        const data = await res.json();
        mcpResourcePreview.textContent = JSON.stringify(data, null, 2);
      } catch (e) {
        mcpResourcePreview.textContent = `// Failed to read resource: ${e.message}`;
      }
    });
  });

  // ==========================================
  // SYSTEM SPECIFICATIONS & ARCHITECTURE
  // ==========================================
  async function loadSystemSpecs() {
    try {
      const res = await fetch("/api/specs");
      const data = await res.json();

      let html = "";
      const specsList = data.specifications || [];
      specsList.forEach((spec) => {
        html += `
          <div class="spec-card">
            <div class="tm-name">${spec.title}</div>
            <div class="tm-reg">${spec.latency_budget || spec.architecture_type || "Active Subsystem"}</div>
            <div class="tm-role">${spec.subsystem}</div>
            <div class="tm-deliverable">${spec.description}</div>
          </div>
        `;
      });
      if (specsCardsContainer) specsCardsContainer.innerHTML = html;
    } catch (e) {
      console.warn("System specs load error:", e);
    }
  }

  // Pre-fetch system specs
  loadSystemSpecs();

  // ==========================================
  // HELPERS
  // ==========================================
  function escapeHtml(str) {
    return str
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;");
  }

  
function formatMarkdown(text) {
  if (typeof text !== "string") return "";

  let html = escapeHtml(text);

  // Preserve fenced code blocks before processing other Markdown.
  const codeBlocks = [];

  html = html.replace(
    /```([a-zA-Z0-9_+-]*)\s*\n([\s\S]*?)```/g,
    (match, language, code) => {
      const index = codeBlocks.length;
      const lang = language || "text";

      codeBlocks.push(
        `<pre class="code-preview font-mono"><code class="language-${lang}">${code.replace(/\n$/, "")}</code></pre>`
      );

      return `@@CODEBLOCK_${index}@@`;
    }
  );

  // Preserve inline code.
  const inlineCodes = [];

  html = html.replace(/`([^`\n]+)`/g, (match, code) => {
    const index = inlineCodes.length;

    inlineCodes.push(
      `<code class="inline-code">${code}</code>`
    );

    return `@@INLINECODE_${index}@@`;
  });

  // Convert common LaTeX commands into readable symbols.
  const mathReplacements = [
    [/\\times/g, "×"],
    [/\\cdot/g, "·"],
    [/\\geq?/g, "≥"],
    [/\\leq?/g, "≤"],
    [/\\neq/g, "≠"],
    [/\\approx/g, "≈"],
    [/\\Delta/g, "Δ"],
    [/\\lambda/g, "λ"],
    [/\\pi/g, "π"],
    [/\\theta/g, "θ"],
    [/\\sqrt\{([^{}]+)\}/g, "√($1)"],
    [/\\frac\{([^{}]+)\}\{([^{}]+)\}/g, "($1 / $2)"],
    [/\\text\{([^{}]+)\}/g, "$1"],
    [/\\left/g, ""],
    [/\\right/g, ""],
    [/\\,/g, " "],
    [/\\;/g, " "],
    [/\\!/g, ""]
  ];

  for (const [pattern, replacement] of mathReplacements) {
    html = html.replace(pattern, replacement);
  }

  // Convert common math delimiters into readable math spans.
  html = html.replace(
    /\\\[([\s\S]*?)\\\]/g,
    '<div class="math-block">$1</div>'
  );

  html = html.replace(
    /\\\(([\s\S]*?)\\\)/g,
    '<span class="math-inline">$1</span>'
  );

  html = html.replace(
    /\$\$([\s\S]*?)\$\$/g,
    '<div class="math-block">$1</div>'
  );

  html = html.replace(
    /\$([^$\n]+)\$/g,
    '<span class="math-inline">$1</span>'
  );

  // Convert headings.
  html = html.replace(
    /^###\s+(.+)$/gm,
    "<h4>$1</h4>"
  );

  html = html.replace(
    /^##\s+(.+)$/gm,
    "<h3>$1</h3>"
  );

  html = html.replace(
    /^#\s+(.+)$/gm,
    "<h2>$1</h2>"
  );

  // Horizontal rules.
  html = html.replace(
    /^\s*[-*_]{3,}\s*$/gm,
    "<hr>"
  );

  // Bold and italic.
  html = html.replace(
    /\*\*(.+?)\*\*/g,
    "<strong>$1</strong>"
  );

  html = html.replace(
    /\*(.+?)\*/g,
    "<em>$1</em>"
  );

  // Convert unordered lists.
  html = html.replace(
    /(?:^|\n)((?:\s*[-*+]\s+.+(?:\n|$))+)/g,
    (match, list) => {
      const items = list
        .trim()
        .split("\n")
        .map(item =>
          `<li>${item.replace(/^\s*[-*+]\s+/, "")}</li>`
        )
        .join("");

      return `\n<ul>${items}</ul>\n`;
    }
  );

  // Convert numbered lists.
  html = html.replace(
    /(?:^|\n)((?:\s*\d+\.\s+.+(?:\n|$))+)/g,
    (match, list) => {
      const items = list
        .trim()
        .split("\n")
        .map(item =>
          `<li>${item.replace(/^\s*\d+\.\s+/, "")}</li>`
        )
        .join("");

      return `\n<ol>${items}</ol>\n`;
    }
  );

  // Paragraphs and line breaks.
  const blocks = html
    .split(/\n{2,}/)
    .map(block => block.trim())
    .filter(Boolean);

  html = blocks.map(block => {
    if (
      /^<(h[2-4]|ul|ol|pre|div|hr)/.test(block) ||
      /^@@CODEBLOCK_\d+@@$/.test(block)
    ) {
      return block;
    }

    return `<p>${block.replace(/\n/g, "<br>")}</p>`;
  }).join("\n");

  // Restore inline code and fenced code blocks.
  inlineCodes.forEach((code, index) => {
    html = html.replace(`@@INLINECODE_${index}@@`, code);
  });

  codeBlocks.forEach((code, index) => {
    html = html.replace(`@@CODEBLOCK_${index}@@`, code);
  });

  return html;
}
